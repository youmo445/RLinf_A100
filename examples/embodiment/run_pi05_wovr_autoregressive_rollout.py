#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _activate_wovr_diffsynth_backend() -> Path:
    """Select the WoVR DiffSynth implementation before importing RLinf."""
    backend_root = Path(
        os.environ.get(
            "WOVR_DIFFSYNTH_PATH", "/data/zsq/diffsynth-studio-wovr-ppu"
        )
    ).expanduser().resolve()
    if not backend_root.is_dir():
        raise FileNotFoundError(
            "WoVR DiffSynth backend not found: "
            f"{backend_root}. Set WOVR_DIFFSYNTH_PATH to the correct checkout."
        )

    backend_str = str(backend_root)
    sys.path[:] = [entry for entry in sys.path if entry != backend_str]
    sys.path.insert(0, backend_str)
    return backend_root


_activate_wovr_diffsynth_backend()

import cv2
import imageio
import matplotlib
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlinf.envs import get_env_cls
from rlinf.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autoregressively roll out PI05 inside single- or three-view WoVR."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/data/zsq/RLinf_A100/examples/embodiment/config"),
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="wovr_robotwin_dual_task_grpo_pi05_head_only_no_state.yaml",
        help="Hydra config name, with or without .yaml.",
    )
    parser.add_argument("--episode-index", type=int, default=160)
    parser.add_argument("--num-chunks", type=int, default=12)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/zsq/RLinf_A100/examples/embodiment/pi05_wovr_rollouts"),
    )
    parser.add_argument(
        "--policy-model-path",
        type=str,
        default=None,
        help="Optional OpenPI/RLinf policy checkpoint root. Defaults to actor.model.model_path in yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Alias for --policy-model-path. Overrides actor.model.model_path from yaml.",
    )
    parser.add_argument(
        "--policy-full-weights",
        type=str,
        default=None,
        help="Optional RLinf full_weights.pt to load after constructing the policy.",
    )
    parser.add_argument(
        "--wm-model-path",
        type=str,
        default=None,
        help="Override env.train.model_path for the WoVR checkpoint.",
    )
    parser.add_argument(
        "--initial-action14",
        type=str,
        default=None,
        help="Optional comma-separated 14-dim action used as current/condition action after reset.",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=("eval", "train"),
        default="eval",
        help="Use eval for deterministic ODE-style rollout, train for flow-SDE exploration.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override env.train.num_inference_steps for the WoVR sampler.",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop after the reward model terminates the single rollout.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_cfg(args):
    config_name = args.config_name.removesuffix(".yaml")
    os.environ.setdefault("EMBODIED_PATH", str(args.config_dir.parent))
    with initialize_config_dir(config_dir=str(args.config_dir), version_base=None):
        cfg = compose(config_name=config_name)

    with open_dict(cfg):
        cfg.env.train.total_num_envs = 1
        cfg.env.train.group_size = 1
        # The standalone script should respect --episode-index exactly.
        cfg.env.train.use_fixed_reset_state_ids = False
        cfg.env.train.auto_reset = False
        cfg.env.train.video_cfg.save_video = False
        cfg.env.train.seed = args.seed
        cfg.env.train.wm_seed = args.seed
        if args.num_inference_steps is not None:
            cfg.env.train.num_inference_steps = int(args.num_inference_steps)
        if args.wm_model_path is not None:
            cfg.env.train.model_path = args.wm_model_path
        override_model_path = args.ckpt_path or args.policy_model_path
        if override_model_path is not None:
            cfg.actor.model.model_path = override_model_path
            cfg.rollout.model.model_path = override_model_path
    return cfg


def build_world_model_env(env_cfg):
    env_type = env_cfg.get("env_type", "wan_multiview_wm")
    if env_type != "wovr_wm":
        raise ValueError(
            f"This script only accepts env_type='wovr_wm', got {env_type!r}"
        )
    env_cls = get_env_cls(env_type, env_cfg)
    return env_cls(env_cfg, num_envs=1, seed_offset=0, total_num_processes=1)


def tensor_frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """Convert [C, T, H, W] or [T, C, H, W] frames in [-1, 1] to [T, H, W, C]."""
    x = frames.detach().float().cpu()
    if x.ndim != 4:
        raise ValueError(f"Expected 4D frames, got {tuple(x.shape)}")
    if x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 3, 0)
    else:
        x = x.permute(0, 2, 3, 1)
    x = (x + 1.0) * 127.5
    return x.clamp(0, 255).to(torch.uint8).numpy()


def split_obs_to_full_image(obs: dict) -> np.ndarray:
    main = obs["main_images"][0].detach().cpu().numpy()
    wrists = obs["wrist_images"][0].detach().cpu().numpy()
    return np.concatenate([main, wrists[0], wrists[1]], axis=0).astype(np.uint8)


def unwrap_obs(obs):
    if isinstance(obs, (list, tuple)):
        if len(obs) != 1:
            raise ValueError(f"Expected a single wrapped observation, got {len(obs)}")
        obs = obs[0]
    if not isinstance(obs, dict):
        raise TypeError(f"Expected observation dict, got {type(obs).__name__}")
    return obs


def parse_action14(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 14:
        raise ValueError(f"--initial-action14 expects 14 values, got {len(vals)}")
    return np.asarray(vals, dtype=np.float32)


def set_initial_action(env, action14: np.ndarray):
    if hasattr(env, "set_initial_action"):
        env.set_initial_action(action14[None])
        return
    action = torch.from_numpy(action14[None]).to(env.device, dtype=torch.float32)
    env.current_action14 = action
    if env.condition_action is not None and env.condition_action.shape[1] > 0:
        env.condition_action[:, -1, :] = action.to(
            env.condition_action.device, dtype=env.condition_action.dtype
        )


def make_policy(cfg, device: str, full_weights: str | None):
    policy = get_model(cfg.actor.model)
    if full_weights:
        state = torch.load(full_weights, map_location="cpu")
        missing, unexpected = policy.load_state_dict(state, strict=False)
        print(
            f"loaded full weights: {full_weights}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if unexpected:
            print(f"unexpected keys sample: {unexpected[:8]}")
    policy.to(device)
    policy.eval()
    return policy


def draw_arm(canvas, root, joints, gripper, color, mirror=False):
    lengths = [58, 50, 44, 38, 32, 26]
    theta = -np.pi / 2
    pts = [np.asarray(root, dtype=np.float32)]
    sign = -1.0 if mirror else 1.0
    for joint, length in zip(joints, lengths):
        theta += sign * float(joint)
        nxt = pts[-1] + np.asarray([np.cos(theta), np.sin(theta)]) * length
        pts.append(nxt)

    for i in range(len(pts) - 1):
        p0 = tuple(np.round(pts[i]).astype(int))
        p1 = tuple(np.round(pts[i + 1]).astype(int))
        cv2.line(canvas, p0, p1, color, 4, cv2.LINE_AA)
        cv2.circle(canvas, p0, 6, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(np.round(pts[-1]).astype(int)), 6, (40, 40, 40), -1)

    grip = float(np.clip(gripper, 0.0, 1.0))
    sep = int(8 + 18 * grip)
    tip = pts[-1]
    cv2.line(
        canvas,
        tuple(np.round(tip + [-sep, 0]).astype(int)),
        tuple(np.round(tip + [-sep, 24]).astype(int)),
        color,
        3,
    )
    cv2.line(
        canvas,
        tuple(np.round(tip + [sep, 0]).astype(int)),
        tuple(np.round(tip + [sep, 24]).astype(int)),
        color,
        3,
    )


def draw_action_frame(action: np.ndarray, frame_idx: int) -> np.ndarray:
    canvas = np.full((540, 900, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"frame {frame_idx:04d}  action14",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    draw_arm(canvas, (245, 265), action[:6], action[6], (30, 110, 220), mirror=False)
    draw_arm(canvas, (655, 265), action[7:13], action[13], (220, 90, 40), mirror=True)
    cv2.putText(canvas, "left", (210, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 110, 220), 2)
    cv2.putText(canvas, "right", (615, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 90, 40), 2)

    x0, y0, bar_w, gap = 40, 430, 45, 12
    vals = np.clip(action.astype(np.float32), -3.2, 3.2)
    for i, val in enumerate(vals):
        x = x0 + i * (bar_w + gap)
        h = int(abs(val) / 3.2 * 70)
        y_mid = y0
        color = (70, 130, 210) if i < 7 else (210, 110, 70)
        cv2.line(canvas, (x, y_mid - 75), (x, y_mid + 75), (210, 210, 210), 1)
        if val >= 0:
            cv2.rectangle(canvas, (x - 12, y_mid - h), (x + 12, y_mid), color, -1)
        else:
            cv2.rectangle(canvas, (x - 12, y_mid), (x + 12, y_mid + h), color, -1)
        cv2.putText(canvas, str(i), (x - 10, y_mid + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    return canvas


def save_video(path: Path, frames: list[np.ndarray], fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def save_action_axis_plots(path_prefix: Path, actions: np.ndarray):
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected actions with shape [T, 14], got {actions.shape}")

    x = np.arange(actions.shape[0])
    arms = [
        ("left", actions[:, :7], ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]),
        ("right", actions[:, 7:], ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]),
    ]
    for arm_name, arm_actions, labels in arms:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
        for axis_idx, label in enumerate(labels):
            ax.plot(x, arm_actions[:, axis_idx], linewidth=1.8, label=label)
        ax.set_title(f"{arm_name.capitalize()} arm action axes over time")
        ax.set_xlabel("frame")
        ax.set_ylabel("action value")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=4, fontsize=9)
        fig.tight_layout()
        fig.savefig(path_prefix.parent / f"{path_prefix.name}_{arm_name}_axes.png")
        plt.close(fig)


def annotate_wovr_frame(
    frame: np.ndarray, frame_idx: int, reward: float, view_name: str
) -> np.ndarray:
    frame = np.ascontiguousarray(frame.copy())
    color = (40, 220, 40) if reward >= 0.5 else (255, 110, 70)
    text = f"{view_name} frame={frame_idx:04d} reward={reward:.3f}"
    cv2.rectangle(frame, (4, 4), (min(frame.shape[1] - 4, 380), 36), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def split_wovr_views(frame: np.ndarray, view_mode: str) -> dict[str, np.ndarray]:
    if view_mode == "single":
        return {"head": frame}
    if frame.shape[0] % 3 != 0:
        raise ValueError(f"Three-view frame height is not divisible by 3: {frame.shape}")
    view_height = frame.shape[0] // 3
    names = ("head", "left_wrist", "right_wrist")
    return {
        name: frame[idx * view_height : (idx + 1) * view_height]
        for idx, name in enumerate(names)
    }


@torch.no_grad()
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = load_cfg(args)
    config_stem = args.config_name.removesuffix(".yaml")
    out_dir = args.output_dir / config_stem / f"episode_{args.episode_index:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")

    env_cfg = cfg.env.train
    env = build_world_model_env(env_cfg)
    obs, _ = env.reset(
        episode_indices=torch.tensor([args.episode_index], dtype=torch.long)
    )
    obs = unwrap_obs(obs)
    initial_frame_idx = 0
    next_output_frame_idx = initial_frame_idx + 1
    print(
        f"reset episode={args.episode_index} view_mode={env.wovr_view_mode} "
        f"initial_frame={initial_frame_idx} next_frame={next_output_frame_idx}"
    )
    if args.initial_action14 is not None:
        set_initial_action(env, parse_action14(args.initial_action14))

    policy = make_policy(cfg, args.device, args.policy_full_weights)

    wm_frames = [tensor_frames_to_uint8(env.current_obs[0, :, 0, -1:, :, :])[0]]
    reward_history = [0.0]
    frame_indices = [initial_frame_idx]
    action_frames = []
    action_history = []
    stopped_on_success = False

    for chunk_idx in range(args.num_chunks):
        obs = unwrap_obs(obs)
        actions, _ = policy.predict_action_batch(
            obs, mode=args.policy_mode, compute_values=False
        )
        actions = actions.detach().to(env.device, dtype=torch.float32)
        action_chunk_len = int(actions.shape[1])
        if action_chunk_len < env.chunk:
            pad = actions[:, -1:, :].repeat(1, env.chunk - actions.shape[1], 1)
            actions = torch.cat([actions, pad], dim=1)
            action_chunk_len = int(actions.shape[1])
        if action_chunk_len % env.chunk != 0:
            raise ValueError(
                f"Policy action chunk length {action_chunk_len} must be a multiple "
                f"of world-model chunk {env.chunk}"
            )

        sub_reward_means = []
        for start in range(0, action_chunk_len, env.chunk):
            sub_actions = actions[:, start : start + env.chunk, :]
            obs, rewards, terminations, truncations, infos = env.chunk_step(sub_actions)
            obs = unwrap_obs(obs)
            action_np = sub_actions[0].detach().cpu().numpy().astype(np.float32)
            action_history.append(action_np)

            chunk_frames = tensor_frames_to_uint8(env.current_obs[0, :, 0, -env.chunk :])
            wm_frames.extend(list(chunk_frames[: len(action_np)]))
            for local_idx, action in enumerate(action_np):
                frame_idx = next_output_frame_idx + len(action_frames)
                action_frames.append(draw_action_frame(action, frame_idx))

            reward_np = rewards.detach().cpu().numpy() if torch.is_tensor(rewards) else rewards
            reward_values = np.asarray(reward_np, dtype=np.float32).reshape(-1)
            reward_history.extend(float(value) for value in reward_values[: len(action_np)])
            first_sub_frame_idx = next_output_frame_idx + len(frame_indices) - 1
            frame_indices.extend(
                range(
                    first_sub_frame_idx,
                    first_sub_frame_idx + len(action_np),
                )
            )
            sub_reward_means.append(float(np.mean(reward_np)))
            if args.stop_on_success and bool(terminations.any()):
                stopped_on_success = True
                break
        print(
            f"chunk {chunk_idx + 1}/{args.num_chunks}: "
            f"actions={tuple(actions.shape)} "
            f"wm_rollouts={action_chunk_len // env.chunk} "
            f"reward_mean={float(np.mean(sub_reward_means)):.4f}"
        )
        if stopped_on_success:
            print("stopping after reward-model success")
            break

    if not (len(wm_frames) == len(frame_indices) == len(reward_history)):
        raise RuntimeError(
            "PI05-WoVR output lengths diverged: "
            f"frames={len(wm_frames)} indices={len(frame_indices)} "
            f"rewards={len(reward_history)}"
        )

    rollout_name = f"wovr_rollout_{env.wovr_view_mode}.mp4"
    annotated_frames = [
        annotate_wovr_frame(frame, frame_idx, reward, env.wovr_view_mode)
        for frame, frame_idx, reward in zip(wm_frames, frame_indices, reward_history)
    ]
    save_video(out_dir / rollout_name, annotated_frames, args.fps)
    for view_name in split_wovr_views(wm_frames[0], env.wovr_view_mode):
        view_frames = []
        for frame, frame_idx, reward in zip(wm_frames, frame_indices, reward_history):
            view = split_wovr_views(frame, env.wovr_view_mode)[view_name]
            view_frames.append(annotate_wovr_frame(view, frame_idx, reward, view_name))
        save_video(out_dir / f"{view_name}.mp4", view_frames, args.fps)
    save_video(out_dir / "action_visualization.mp4", action_frames, args.fps)
    total_generated_frames = len(wm_frames) - 1
    metadata = {
        "config_name": args.config_name,
        "view_mode": env.wovr_view_mode,
        "policy_mode": args.policy_mode,
        "num_inference_steps": int(env.num_inference_steps),
        "episode_index": int(args.episode_index),
        "initial_frame_idx": int(initial_frame_idx),
        "first_generated_frame_idx": int(next_output_frame_idx),
        "last_generated_frame_idx": int(next_output_frame_idx + total_generated_frames - 1)
        if total_generated_frames > 0
        else None,
        "num_policy_chunks": int(args.num_chunks),
        "num_generated_frames": int(total_generated_frames),
        "wm_chunk": int(env.chunk),
        "stopped_on_success": bool(stopped_on_success),
        "frame_indices": frame_indices,
        "rewards": reward_history,
    }
    with (out_dir / "rollout_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    if action_history:
        all_actions = np.concatenate(action_history, axis=0)
        save_action_axis_plots(out_dir / "action", all_actions)
    print(f"saved: {out_dir / rollout_name}")
    print(f"saved: {out_dir / 'action_visualization.mp4'}")
    print(f"saved: {out_dir / 'action_left_axes.png'}")
    print(f"saved: {out_dir / 'action_right_axes.png'}")
    print(f"saved: {out_dir / 'rollout_metadata.json'}")


if __name__ == "__main__":
    main()
