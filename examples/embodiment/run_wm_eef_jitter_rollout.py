#!/usr/bin/env python3
"""Generate a WM rollout from dataset actions with per-frame EE xyz jitter.

Frame step convention:
  - step 0 is the reset/initial frame.
  - generated steps start from 1.
  - --jitter-start-step and --jitter-end-step are inclusive generated frame indices.

This script does not modify any existing project files. It reuses the direct-EE
world-model driver from run_wm_eef_gradio_interactive.py. The baseline trajectory
is the episode action trajectory from the npy dataset; jitter is added only after
converting each frame's action to EE pose.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import imageio
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_wm_eef_gradio_interactive import EEFInteractiveWM, save_video


def parse_axes(text: str) -> Tuple[int, ...]:
    axes = []
    for ch in str(text).lower():
        if ch == "x":
            axes.append(0)
        elif ch == "y":
            axes.append(1)
        elif ch == "z":
            axes.append(2)
        else:
            raise ValueError(f"Unsupported jitter axis {ch!r}; use a combination of x/y/z.")
    if not axes:
        raise ValueError("At least one jitter axis is required.")
    return tuple(dict.fromkeys(axes))


def sample_jitter_offset(
    rng: np.random.Generator,
    global_step: int,
    start_step: int,
    end_step: int,
    axes: Tuple[int, ...],
    amplitude: float,
    noise_type: str,
) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    if global_step < start_step or global_step > end_step:
        return out
    if noise_type == "uniform":
        noise = rng.uniform(-float(amplitude), float(amplitude), size=len(axes))
    elif noise_type == "normal":
        noise = rng.normal(0.0, float(amplitude), size=len(axes))
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    for value, axis in zip(noise, axes):
        out[axis] = float(value)
    return out


def save_frames(path: Path, frames: List[np.ndarray], fps: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)
    return str(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/data/zsq/RLinf/examples/embodiment/config")
    parser.add_argument("--config-name", default="wanmultiview_robotwin_grpo_pi05")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wm-seed", type=int, default=42)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        default="/data/zsq/RLinf/examples/embodiment/eef_jitter_rollouts",
    )
    parser.add_argument("--jitter-start-step", type=int, default=1)
    parser.add_argument("--jitter-end-step", type=int, default=32)
    parser.add_argument(
        "--jitter-arm",
        choices=["left", "right", "both"],
        default="left",
    )
    parser.add_argument(
        "--jitter-axes",
        default="xyz",
        help="Combination of x/y/z axes to jitter, e.g. x, yz, xyz.",
    )
    parser.add_argument(
        "--jitter-amplitude",
        type=float,
        default=0.18,
        help="Uniform max absolute noise in meters, or normal std if --jitter-noise-type=normal.",
    )
    parser.add_argument(
        "--jitter-period",
        type=float,
        default=4.0,
        help="Ignored. Kept only for backward CLI compatibility.",
    )
    parser.add_argument(
        "--jitter-noise-type",
        choices=["uniform", "normal"],
        default="uniform",
    )
    parser.add_argument("--jitter-seed", type=int, default=0)
    parser.add_argument("--initial-left-gripper", type=float, default=0.0)
    parser.add_argument("--initial-right-gripper", type=float, default=0.0)
    parser.add_argument("--initial-left-pose", default=None)
    parser.add_argument("--initial-right-pose", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.jitter_end_step < args.jitter_start_step:
        raise ValueError("--jitter-end-step must be >= --jitter-start-step")

    # EEFInteractiveWM expects these fields even though this script does not use UI.
    args.server_name = "0.0.0.0"
    args.server_port = 7860
    args.share = False
    args.translation_step = 0.01
    args.pixel_step = 16
    args.rotation_step_deg = 5.0
    args.gripper_step = 0.05

    app = EEFInteractiveWM(args)
    env = app.env
    axes = parse_axes(args.jitter_axes)
    rng = np.random.default_rng(int(args.jitter_seed))

    trajectory = np.load(env.dataset.npy_files[int(args.episode_index)], allow_pickle=True)
    actions = np.asarray(
        [frame.get("action", frame.get("state")) for frame in trajectory],
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected episode actions [T, 14], got {actions.shape}")
    rollout_actions = actions[1:]
    if args.max_steps is not None:
        rollout_actions = rollout_actions[: int(args.max_steps)]
    if args.num_chunks is None or int(args.num_chunks) <= 0:
        num_chunks = int(np.ceil(len(rollout_actions) / env.chunk))
    else:
        num_chunks = min(int(args.num_chunks), int(np.ceil(len(rollout_actions) / env.chunk)))
    rollout_actions = rollout_actions[: num_chunks * env.chunk]

    out_dir = Path(app.output_dir)
    wm_frames: List[np.ndarray] = [app.latest_frame]
    control_frames: List[np.ndarray] = [app.latest_control_frame]
    csv_rows = []

    base_left = app.left_pose.copy()
    base_right = app.right_pose.copy()
    anchor_left = app.condition_left_poses[0].copy()
    anchor_right = app.condition_right_poses[0].copy()
    condition_actions14 = app.env.condition_action[0].detach().float().cpu().numpy().copy()

    print(
        f"[jitter] config={args.config_name} episode={args.episode_index} "
        f"context={env.condition_frame_length} chunk={env.chunk} "
        f"dataset_steps={len(actions)} rollout_steps={len(rollout_actions)} num_chunks={num_chunks}"
    )
    print(
        f"[jitter] arm={args.jitter_arm} axes={args.jitter_axes} "
        f"amp={args.jitter_amplitude} noise={args.jitter_noise_type} seed={args.jitter_seed} "
        f"steps=[{args.jitter_start_step}, {args.jitter_end_step}]"
    )
    print(f"[jitter] output_dir={out_dir}")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * env.chunk
        end = min(start + env.chunk, len(rollout_actions))
        chunk_actions = rollout_actions[start:end].astype(np.float32)
        valid_len = int(len(chunk_actions))
        if valid_len <= 0:
            break
        if valid_len < env.chunk:
            pad = np.repeat(chunk_actions[-1:], env.chunk - valid_len, axis=0)
            chunk_actions = np.concatenate([chunk_actions, pad], axis=0)

        left_context = [p.copy() for p in app.condition_left_poses[-env.condition_frame_length :]]
        right_context = [p.copy() for p in app.condition_right_poses[-env.condition_frame_length :]]
        while len(left_context) < env.condition_frame_length:
            left_context.insert(0, left_context[0].copy())
            right_context.insert(0, right_context[0].copy())

        base_ee_pose, _ = env._action14_to_ee_pose_and_state(chunk_actions)
        base_left_future = base_ee_pose[:, 0:7].astype(np.float32)
        base_right_future = base_ee_pose[:, 7:14].astype(np.float32)
        left_future = []
        right_future = []
        chunk_noise_abs_max = 0.0
        for i in range(env.chunk):
            global_step = start + i + 1
            offset = sample_jitter_offset(
                rng,
                global_step,
                args.jitter_start_step,
                args.jitter_end_step,
                axes,
                args.jitter_amplitude,
                args.jitter_noise_type,
            )
            chunk_noise_abs_max = max(chunk_noise_abs_max, float(np.max(np.abs(offset))))
            left_pose = base_left_future[i].copy()
            right_pose = base_right_future[i].copy()
            if args.jitter_arm in ("left", "both"):
                left_pose[:3] = base_left_future[i, :3] + offset
            if args.jitter_arm in ("right", "both"):
                right_pose[:3] = base_right_future[i, :3] + offset
            left_future.append(left_pose)
            right_future.append(right_pose)
            csv_rows.append(
                {
                    "step": global_step,
                    "chunk": chunk_idx,
                    "base_left_x": float(base_left_future[i, 0]),
                    "base_left_y": float(base_left_future[i, 1]),
                    "base_left_z": float(base_left_future[i, 2]),
                    "left_x": float(left_pose[0]),
                    "left_y": float(left_pose[1]),
                    "left_z": float(left_pose[2]),
                    "base_right_x": float(base_right_future[i, 0]),
                    "base_right_y": float(base_right_future[i, 1]),
                    "base_right_z": float(base_right_future[i, 2]),
                    "right_x": float(right_pose[0]),
                    "right_y": float(right_pose[1]),
                    "right_z": float(right_pose[2]),
                    "offset_x": float(offset[0]),
                    "offset_y": float(offset[1]),
                    "offset_z": float(offset[2]),
                }
            )

        left_seq = np.stack(left_context + left_future, axis=0).astype(np.float32)
        right_seq = np.stack(right_context + right_future, axis=0).astype(np.float32)
        action_seq14 = np.concatenate([condition_actions14, chunk_actions], axis=0).astype(np.float32)

        app.left_pose = left_future[-1].copy()
        app.right_pose = right_future[-1].copy()
        control_video, ray_map_o, ray_map_d = app._build_controls(
            left_seq, right_seq, action_seq14
        )
        input_video = app._build_input_video()
        output = app._run_world_model(input_video, control_video, ray_map_o, ray_map_d)
        chunk_wm_frames, chunk_control_frames = app._commit_generation(output, control_video)

        recent_count = max(0, int(env.condition_frame_length) - 1)
        app.condition_left_poses = ([anchor_left.copy()] + [p.copy() for p in left_future[-recent_count:]])[
            -env.condition_frame_length :
        ]
        app.condition_right_poses = ([anchor_right.copy()] + [p.copy() for p in right_future[-recent_count:]])[
            -env.condition_frame_length :
        ]
        recent_action_part = (
            chunk_actions[-recent_count:]
            if recent_count > 0
            else np.zeros((0, 14), dtype=np.float32)
        )
        condition_actions14 = np.concatenate(
            [env.reference_action14[0].detach().float().cpu().numpy()[None], recent_action_part],
            axis=0,
        ).astype(np.float32)
        env.current_action14[0] = env.current_action14.new_tensor(chunk_actions[-1])
        env.condition_action[0] = env.condition_action.new_tensor(condition_actions14)

        wm_frames.extend(chunk_wm_frames[:valid_len])
        control_frames.extend(chunk_control_frames[:valid_len])
        print(
            f"[jitter] chunk {chunk_idx + 1}/{num_chunks}: "
            f"frames={valid_len} raw_steps=[{start + 1},{start + valid_len}] "
            f"max_abs_noise={chunk_noise_abs_max:.4f} "
            f"last_base_left_xyz={base_left_future[valid_len - 1, :3].tolist()} "
            f"last_left_xyz={app.left_pose[:3].tolist()} "
            f"last_right_xyz={app.right_pose[:3].tolist()}"
        )

    wm_path = save_frames(out_dir / "wm_jitter_rollout.mp4", wm_frames, int(args.fps))
    control_path = save_frames(
        out_dir / "control_map_jitter_rollout.mp4", control_frames, int(args.fps)
    )
    csv_path = out_dir / "ee_xyz_jitter.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[jitter] saved wm_video={wm_path}")
    print(f"[jitter] saved control_video={control_path}")
    print(f"[jitter] saved ee_csv={csv_path}")


if __name__ == "__main__":
    main()
