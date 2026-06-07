#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from rlinf.envs.world_model.world_model_wan_multi_view_env import WanmultiviewEnv
from rlinf.models import get_model


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
        default=Path("/data/zsq/RLinf/examples/embodiment/config/test_data/wanmultiview_arxx5_0521_rollout"),
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
    parser.add_argument(
        "--rollout-source",
        type=str,
        default="vla",
        choices=("vla", "dataset"),
        help="Use PI05 to interact with WM, or replay dataset actions.",
    )
    parser.add_argument(
        "--policy-model-path",
        type=Path,
        default=Path("/data/zsq/openpi/checkpoints/pi05_arxx5/arxx5_h=16_hasstate_0520/10000torch"),
        help="OpenPI/RLinf policy checkpoint root used when --rollout-source=vla.",
    )
    parser.add_argument(
        "--policy-full-weights",
        type=Path,
        default=None,
        help="Optional RLinf full_weights.pt to load after constructing the policy.",
    )
    parser.add_argument(
        "--num-action-chunks",
        type=int,
        default=16,
        help="PI05 action chunk length. Must be a multiple of the WM chunk.",
    )
    parser.add_argument("--device", type=str, default="cuda")
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
        env_cfg.wm_chunk = 8
        env_cfg.context_frames = 5
        env_cfg.wm_num_frames = 13

    with open_dict(full_cfg):
        if "actor" in full_cfg and "model" in full_cfg.actor:
            full_cfg.actor.model.model_path = str(args.policy_model_path)
            full_cfg.actor.model.use_proprio = True
            full_cfg.actor.model.num_action_chunks = int(args.num_action_chunks)
            if "openpi" in full_cfg.actor.model:
                full_cfg.actor.model.openpi.action_chunk = int(args.num_action_chunks)
        if "rollout" in full_cfg and "model" in full_cfg.rollout:
            full_cfg.rollout.model.model_path = str(args.policy_model_path)
            full_cfg.rollout.model.use_proprio = True
            full_cfg.rollout.model.num_action_chunks = int(args.num_action_chunks)
            if "openpi" in full_cfg.rollout.model:
                full_cfg.rollout.model.openpi.action_chunk = int(args.num_action_chunks)
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


def unwrap_obs(obs):
    if isinstance(obs, list):
        if len(obs) != 1:
            raise ValueError(f"Expected one env obs, got {len(obs)}")
        return obs[0]
    return obs


def patch_arxx5_state_obs_processor(policy):
    original_obs_processor = policy.obs_processor

    def obs_processor(self, env_obs):
        if "arxx5" not in self.config.config_name:
            return original_obs_processor(env_obs)

        wrist_images = env_obs["wrist_images"]
        if wrist_images is None:
            raise ValueError("ARXX5 OpenPI requires env_obs['wrist_images']")
        processed_obs = {
            "image": {
                "base_0_rgb": env_obs["main_images"],
                "left_wrist_0_rgb": wrist_images[:, 0],
                "right_wrist_0_rgb": wrist_images[:, 1],
            },
            "prompt": env_obs["task_descriptions"],
        }
        if "states" in env_obs and env_obs["states"] is not None:
            processed_obs["state"] = env_obs["states"]
        return processed_obs

    policy.obs_processor = types.MethodType(obs_processor, policy)


def make_policy(full_cfg, args):
    policy = get_model(full_cfg.actor.model)
    if args.policy_full_weights is not None:
        state = torch.load(args.policy_full_weights, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = policy.load_state_dict(state, strict=False)
        print(
            f"loaded full weights: {args.policy_full_weights}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if unexpected:
            print(f"unexpected keys sample: {unexpected[:8]}")
    patch_arxx5_state_obs_processor(policy)
    policy.to(args.device)
    policy.eval()
    return policy


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

    rollout_actions = None
    if args.rollout_source == "dataset":
        actions = load_episode_actions(env, args.episode_index)
        rollout_actions = actions[1:]
        if args.max_steps is not None:
            rollout_actions = rollout_actions[: args.max_steps]
        max_steps = int(len(rollout_actions))
    else:
        max_steps = int(args.max_steps or env_cfg.get("max_episode_steps", 256))
        if args.num_action_chunks % env.chunk != 0:
            raise ValueError(
                f"--num-action-chunks ({args.num_action_chunks}) must be a "
                f"multiple of WM chunk ({env.chunk})"
            )

    with open_dict(env_cfg):
        env_cfg.max_episode_steps = max_steps

    obs, info = env.reset(episode_indices=np.asarray([args.episode_index]))
    obs = unwrap_obs(obs)
    del info

    policy = None
    if args.rollout_source == "vla":
        if not args.policy_model_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_model_path}")
        if args.policy_full_weights is not None and not args.policy_full_weights.exists():
            raise FileNotFoundError(f"Policy full weights not found: {args.policy_full_weights}")
        policy = make_policy(full_cfg, args)

    generated_frames = [
        tensor_frames_to_uint8(env.current_obs[0, :, 0, -1:, :, :])[0]
    ]
    rewards = [0.0]

    print(
        f"Rolling out episode={args.episode_index}, source={args.rollout_source}, "
        f"steps={max_steps}, wm_chunk={env.chunk}"
    )
    print(f"dataset={dataset_dir}")
    print(f"reward_ckpt={args.reward_ckpt}")
    print(f"reward_threshold={env.reward_model.threshold}")
    if policy is not None:
        print(f"policy_model_path={args.policy_model_path}")
        if args.policy_full_weights is not None:
            print(f"policy_full_weights={args.policy_full_weights}")

    steps_done = 0
    policy_chunk_idx = 0
    while steps_done < max_steps:
        if args.rollout_source == "dataset":
            assert rollout_actions is not None
            chunk_np = rollout_actions[steps_done : steps_done + env.chunk]
            if len(chunk_np) == 0:
                break
            action_chunk = torch.from_numpy(chunk_np[None]).to(
                env.device, dtype=torch.float32
            )
            policy_chunk_idx += 1
        else:
            obs = unwrap_obs(obs)
            action_chunk, _ = policy.predict_action_batch(
                obs, mode="eval", compute_values=False
            )
            action_chunk = action_chunk.detach().to(env.device, dtype=torch.float32)
            if action_chunk.shape[1] % env.chunk != 0:
                raise ValueError(
                    f"PI05 returned action chunk length {action_chunk.shape[1]}, "
                    f"which is not a multiple of WM chunk {env.chunk}"
                )
            policy_chunk_idx += 1

        action_len = int(action_chunk.shape[1])
        for local_start in range(0, action_len, env.chunk):
            if steps_done >= max_steps:
                break

            sub_actions = action_chunk[:, local_start : local_start + env.chunk]
            valid_len = min(sub_actions.shape[1], max_steps - steps_done)
            if valid_len == 0:
                break
            if valid_len < env.chunk:
                pad = sub_actions[:, -1:, :].repeat(1, env.chunk - valid_len, 1)
                sub_actions = torch.cat([sub_actions, pad], dim=1)

            obs, chunk_rewards, terminations, truncations, _ = env.chunk_step(sub_actions)
            chunk_reward_np = chunk_rewards[0, :valid_len].detach().cpu().numpy()
            chunk_frames = tensor_frames_to_uint8(
                env.current_obs[0, :, 0, -env.chunk :, :, :]
            )[:valid_len]

            generated_frames.extend(list(chunk_frames))
            rewards.extend([float(x) for x in chunk_reward_np])
            steps_done += valid_len
            print(
                f"policy_chunk={policy_chunk_idx} wm_subchunk={local_start // env.chunk + 1}: "
                f"frames={valid_len}, steps_done={steps_done}, "
                f"rewards={chunk_reward_np.tolist()}"
            )

            done = bool(terminations[0, :valid_len].any() or truncations[0, :valid_len].any())
            if done:
                steps_done = max_steps
                break

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
                "rollout_source": args.rollout_source,
                "policy_model_path": str(args.policy_model_path) if policy is not None else None,
                "policy_full_weights": str(args.policy_full_weights)
                if args.policy_full_weights is not None
                else None,
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
