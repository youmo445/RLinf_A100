#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from rlinf.envs.world_model.world_model_wan_multi_view_env import WanmultiviewEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate a WAN multi-view world-model rollout video and per-frame "
            "binary rewards using a chosen reward-model checkpoint."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/data/zsq/RLinf/examples/embodiment/config"),
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="wanmultiview_arxx5_grpo_pi05",
        help="Hydra config name, with or without .yaml.",
    )
    parser.add_argument(
        "--env-section",
        type=str,
        default="train",
        choices=("train", "eval"),
        help="Use cfg.env.train or cfg.env.eval as the world-model env config.",
    )
    parser.add_argument("--episode-index", type=int, default=20)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Rollout dataset dir. Defaults to cfg.env.<section>.initial_image_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/zsq/RLinf/outputs/wanmultiview_rollout_latest_reward"),
    )
    parser.add_argument(
        "--reward-ckpt",
        type=Path,
        default=Path(
            "/data/zsq/stack_blocks_bimanual_0513/reward_model_runs/resnet_multi/latest.pth"
        ),
        help="Reward model checkpoint. Use best.pth/latest.pth as needed.",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=None,
        help="Override checkpoint threshold. Omit to use threshold stored in ckpt.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit on rollout action steps after the reset frame.",
    )
    return parser.parse_args()


def load_env_cfg(args):
    os.environ.setdefault("EMBODIED_PATH", str(args.config_dir.parent))
    GlobalHydra.instance().clear()
    config_name = args.config_name.removesuffix(".yaml")
    with initialize_config_dir(config_dir=str(args.config_dir), version_base="1.1"):
        full_cfg = compose(config_name=config_name)

    env_cfg = full_cfg.env[args.env_section]
    with open_dict(env_cfg):
        if args.dataset_dir is not None:
            env_cfg.initial_image_path = str(args.dataset_dir)
        env_cfg.total_num_envs = 1
        env_cfg.group_size = 1
        env_cfg.auto_reset = False
        env_cfg.use_fixed_reset_state_ids = False
        env_cfg.reward_model.type = "ResnetRewModel"
        env_cfg.reward_model.from_pretrained = str(args.reward_ckpt)
        env_cfg.reward_model.threshold = args.reward_threshold
    return env_cfg, full_cfg


def tensor_frames_to_uint8(frames_tensor: torch.Tensor) -> np.ndarray:
    x = frames_tensor.detach().float().cpu()
    if x.ndim != 4:
        raise ValueError(f"Expected [C, T, H, W], got {tuple(x.shape)}")
    x = x.permute(1, 2, 3, 0)
    x = (x + 1.0) / 2.0 * 255.0
    return torch.clamp(x, 0, 255).numpy().astype(np.uint8)


def annotate_frame(frame: np.ndarray, frame_idx: int, reward_value: float) -> np.ndarray:
    annotated = np.ascontiguousarray(frame.copy())
    text = f"frame={frame_idx:04d} reward={reward_value:.3f}"
    color = (0, 255, 0) if reward_value >= 0.5 else (255, 80, 80)
    cv2.rectangle(annotated, (6, 6), (360, 38), (0, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return annotated


def load_episode_actions(env: WanmultiviewEnv, episode_index: int) -> np.ndarray:
    trajectory = np.load(env.dataset.npy_files[episode_index], allow_pickle=True)
    actions = np.asarray(
        [frame.get("action", frame.get("state")) for frame in trajectory],
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected episode actions [T, 14], got {actions.shape}")
    return actions


@torch.no_grad()
def main():
    args = parse_args()
    env_cfg, full_cfg = load_env_cfg(args)
    dataset_dir = Path(env_cfg.initial_image_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Rollout dataset not found: {dataset_dir}")
    if not args.reward_ckpt.exists():
        raise FileNotFoundError(f"Reward checkpoint not found: {args.reward_ckpt}")

    env = WanmultiviewEnv(env_cfg, num_envs=1, seed_offset=0, total_num_processes=1)
    if args.episode_index < 0 or args.episode_index >= len(env.dataset):
        raise ValueError(
            f"--episode-index {args.episode_index} outside dataset range "
            f"[0, {len(env.dataset) - 1}]"
        )

    actions = load_episode_actions(env, args.episode_index)
    rollout_actions = actions[1:]
    if args.max_steps is not None:
        rollout_actions = rollout_actions[: args.max_steps]

    with open_dict(env_cfg):
        env_cfg.max_episode_steps = int(len(rollout_actions))

    obs, info = env.reset(episode_indices=np.asarray([args.episode_index]))
    del obs, info

    generated_frames = [
        tensor_frames_to_uint8(env.current_obs[0, :, 0, -1:, :, :])[0]
    ]
    rewards = [0.0]

    num_chunks = int(np.ceil(len(rollout_actions) / env_cfg.chunk))
    print(
        f"Rolling out episode={args.episode_index}, steps={len(rollout_actions)}, "
        f"chunks={num_chunks}, chunk={env_cfg.chunk}"
    )
    print(f"dataset={dataset_dir}")
    print(f"reward_ckpt={args.reward_ckpt}")
    print(f"reward_threshold={env.reward_model.threshold}")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * env_cfg.chunk
        end = min(start + env_cfg.chunk, len(rollout_actions))
        chunk = rollout_actions[start:end]
        valid_len = len(chunk)
        if valid_len == 0:
            continue
        if valid_len < env_cfg.chunk:
            pad = np.repeat(chunk[-1:], env_cfg.chunk - valid_len, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

        chunk_tensor = torch.from_numpy(chunk[None]).to(
            env.device, dtype=torch.float32
        )
        _, chunk_rewards, _, _, _ = env.chunk_step(chunk_tensor)
        chunk_reward_np = chunk_rewards[0, :valid_len].detach().cpu().numpy()
        chunk_frames = tensor_frames_to_uint8(
            env.current_obs[0, :, 0, -env_cfg.chunk :, :, :]
        )[:valid_len]

        generated_frames.extend(list(chunk_frames))
        rewards.extend([float(x) for x in chunk_reward_np])
        print(
            f"chunk {chunk_idx + 1}/{num_chunks}: "
            f"frames={valid_len}, rewards={chunk_reward_np.tolist()}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = (
        args.output_dir / f"episode_{args.episode_index:06d}_wm_rollout_reward.mp4"
    )
    rewards_path = (
        args.output_dir / f"episode_{args.episode_index:06d}_wm_rollout_rewards.json"
    )
    config_path = args.output_dir / f"episode_{args.episode_index:06d}_config.yaml"

    annotated_frames = [
        annotate_frame(frame, idx, reward)
        for idx, (frame, reward) in enumerate(zip(generated_frames, rewards))
    ]
    imageio.mimsave(video_path, annotated_frames, fps=args.fps, macro_block_size=1)
    rewards_path.write_text(
        json.dumps(
            {
                "episode_index": args.episode_index,
                "dataset": str(dataset_dir),
                "video": str(video_path),
                "fps": args.fps,
                "reward_ckpt": str(args.reward_ckpt),
                "reward_threshold": float(env.reward_model.threshold),
                "rewards": rewards,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    OmegaConf.save(full_cfg, config_path)
    print(f"Saved rollout video: {video_path}")
    print(f"Saved rewards: {rewards_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
