#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch autoregressive PI05+multi-view WAN world-model rollout over all "
            "initial test-data episodes, with reward annotations and success summary."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/data/zsq/RLinf/examples/embodiment/config"),
    )
    parser.add_argument("--config-name", type=str, default="wanmultiview_arxx5_grpo_pi05")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/zsq/RLinf/examples/embodiment/pi05_wm_batch_rollouts_0605_step30"),
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=46, help="Exclusive episode index.")
    parser.add_argument("--episode-indices", type=str, default=None, help="Comma-separated explicit indices.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--num-policy-chunks",
        type=int,
        default=None,
        help="Defaults to ceil(env.max_episode_steps / actor.model.num_action_chunks).",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--policy-model-path",
        type=str,
        default=None,
        help="Optional OpenPI/RLinf policy checkpoint root. Defaults to actor.model.model_path in yaml.",
    )
    parser.add_argument("--ckpt-path", type=str, default=None, help="Alias for --policy-model-path.")
    parser.add_argument(
        "--policy-full-weights",
        type=str,
        default=None,
        help="Optional RLinf full_weights.pt to load after constructing the policy.",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=("eval", "train"),
        default="eval",
        help="eval uses deterministic ODE-style rollout; train uses flow-SDE exploration.",
    )
    parser.add_argument("--success-threshold", type=float, default=None)
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop the current episode once any generated frame reaches the success threshold.",
    )
    parser.add_argument(
        "--disable-kir",
        action="store_true",
        help="Disable KIR even if the yaml enables it.",
    )
    parser.add_argument(
        "--kir-initial-frame",
        type=int,
        default=None,
        help="Override yaml KIR start frame. Enables KIR for each evaluated episode.",
    )
    parser.add_argument("--save-raw-video", action="store_true")
    parser.add_argument("--save-action-video", action="store_true")
    parser.add_argument("--save-action-plots", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def load_cfg(args):
    config_name = args.config_name.removesuffix(".yaml")
    os.environ.setdefault("EMBODIED_PATH", str(args.config_dir.parent))
    with initialize_config_dir(config_dir=str(args.config_dir), version_base=None):
        cfg = compose(config_name=config_name)

    with open_dict(cfg):
        cfg.env.train.total_num_envs = 1
        cfg.env.train.group_size = 1
        cfg.env.train.use_fixed_reset_state_ids = False
        cfg.env.train.auto_reset = False
        cfg.env.train.video_cfg.save_video = False
        cfg.env.train.seed = args.seed
        override_model_path = args.ckpt_path or args.policy_model_path
        if override_model_path is not None:
            cfg.actor.model.model_path = override_model_path
            cfg.rollout.model.model_path = override_model_path
        if args.success_threshold is not None:
            cfg.env.train.success_reward_threshold = float(args.success_threshold)
        if args.disable_kir:
            cfg.env.train.enable_kir = False
            cfg.env.train.kir_initial_frame = None
        elif args.kir_initial_frame is not None:
            cfg.env.train.enable_kir = True
            cfg.env.train.kir_initial_frame = int(args.kir_initial_frame)
    return cfg


def unwrap_obs(obs):
    if isinstance(obs, (list, tuple)):
        if len(obs) != 1:
            raise ValueError(f"Expected a single wrapped observation, got {len(obs)}")
        obs = obs[0]
    if not isinstance(obs, dict):
        raise TypeError(f"Expected observation dict, got {type(obs).__name__}")
    return obs


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


def make_policy(cfg, device: str, full_weights: str | None):
    from rlinf.models import get_model

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
    return canvas


def annotate_wm_frame(
    frame: np.ndarray,
    *,
    episode_idx: int,
    generated_frame_idx: int,
    reward: float | None,
    max_reward: float,
    success_threshold: float,
    success: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = 44
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h] = (28, 28, 28)
    canvas[bar_h:] = frame
    reward_text = "init" if reward is None else f"reward={reward:.4f}"
    status = "SUCCESS" if success else "running"
    status_color = (80, 220, 80) if success else (230, 230, 230)
    text = (
        f"episode={episode_idx:06d} frame={generated_frame_idx:04d}  "
        f"{reward_text} max={max_reward:.4f} threshold={success_threshold:.3f}"
    )
    cv2.putText(canvas, text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, status, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)
    return canvas


def save_video(path: Path, frames: list[np.ndarray], fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
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


def parse_episode_indices(args, dataset_len: int) -> list[int]:
    if args.episode_indices:
        indices = [int(x.strip()) for x in args.episode_indices.split(",") if x.strip()]
    else:
        end = dataset_len if args.episode_end is None else min(args.episode_end, dataset_len)
        indices = list(range(args.episode_start, end))
    indices = [idx for idx in indices if 0 <= idx < dataset_len]
    if args.max_episodes is not None:
        indices = indices[: args.max_episodes]
    if not indices:
        raise ValueError("No episode indices selected.")
    return indices


def default_num_policy_chunks(cfg) -> int:
    max_steps = int(cfg.env.train.max_episode_steps)
    action_chunk = int(cfg.actor.model.num_action_chunks)
    return int(math.ceil(max_steps / action_chunk))


def write_rewards_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "generated_frame_idx",
        "policy_chunk_idx",
        "wm_subchunk_idx",
        "local_idx",
        "reward",
        "success",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def rollout_episode(
    *,
    env: WanmultiviewEnv,
    policy,
    episode_idx: int,
    num_policy_chunks: int,
    args,
    success_threshold: float,
) -> dict:
    out_dir = args.output_dir / f"episode_{episode_idx:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(episode_indices=torch.tensor([episode_idx], dtype=torch.long))
    obs = unwrap_obs(obs)

    kir_active = bool(env.enable_kir and env.kir_initial_frame is not None)
    initial_frame_idx = int(env.kir_initial_frame) if kir_active else 0
    next_generated_frame_idx = initial_frame_idx + 1
    task = obs.get("task_descriptions", [""])[0] if isinstance(obs.get("task_descriptions"), list) else ""

    init_frame = tensor_frames_to_uint8(env.current_obs[0, :, 0, -1:, :, :])[0]
    raw_wm_frames = [init_frame]
    annotated_frames = [
        annotate_wm_frame(
            init_frame,
            episode_idx=episode_idx,
            generated_frame_idx=initial_frame_idx,
            reward=None,
            max_reward=0.0,
            success_threshold=success_threshold,
            success=False,
        )
    ]
    action_frames = []
    action_history = []
    reward_rows: list[dict] = []
    all_rewards: list[float] = []
    success = False
    total_generated = 0

    for policy_chunk_idx in range(num_policy_chunks):
        obs = unwrap_obs(obs)
        actions, _ = policy.predict_action_batch(obs, mode=args.policy_mode, compute_values=False)
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

        for wm_subchunk_idx, start in enumerate(range(0, action_chunk_len, env.chunk)):
            sub_actions = actions[:, start : start + env.chunk, :]
            obs, rewards, terminations, truncations, infos = env.chunk_step(sub_actions)
            obs = unwrap_obs(obs)

            action_np = sub_actions[0].detach().cpu().numpy().astype(np.float32)
            reward_np = rewards[0].detach().float().cpu().numpy().astype(np.float32)
            term_np = terminations[0].detach().cpu().numpy().astype(bool)
            all_rewards.extend(float(x) for x in reward_np)
            action_history.append(action_np)

            chunk_frames = tensor_frames_to_uint8(env.current_obs[0, :, 0, -env.chunk :])
            valid_len = min(len(action_np), len(chunk_frames), len(reward_np))
            for local_idx in range(valid_len):
                generated_frame_idx = next_generated_frame_idx + total_generated
                reward = float(reward_np[local_idx])
                max_reward = max(all_rewards) if all_rewards else 0.0
                frame_success = bool(max_reward >= success_threshold or term_np[local_idx])
                success = bool(success or frame_success)

                raw_wm_frames.append(chunk_frames[local_idx])
                annotated_frames.append(
                    annotate_wm_frame(
                        chunk_frames[local_idx],
                        episode_idx=episode_idx,
                        generated_frame_idx=generated_frame_idx,
                        reward=reward,
                        max_reward=max_reward,
                        success_threshold=success_threshold,
                        success=success,
                    )
                )
                if args.save_action_video:
                    action_frames.append(draw_action_frame(action_np[local_idx], generated_frame_idx))
                reward_rows.append(
                    {
                        "generated_frame_idx": generated_frame_idx,
                        "policy_chunk_idx": policy_chunk_idx,
                        "wm_subchunk_idx": wm_subchunk_idx,
                        "local_idx": local_idx,
                        "reward": reward,
                        "success": int(success),
                    }
                )
                total_generated += 1

            print(
                f"episode={episode_idx:06d} policy_chunk={policy_chunk_idx + 1}/{num_policy_chunks} "
                f"wm_subchunk={wm_subchunk_idx + 1}/{action_chunk_len // env.chunk} "
                f"reward_max={max(all_rewards) if all_rewards else 0.0:.4f} success={success}"
            )
            if success and args.stop_on_success:
                break
        if success and args.stop_on_success:
            break

    annotated_video = out_dir / "wm_rollout_reward_annotated.mp4"
    save_video(annotated_video, annotated_frames, args.fps)
    if args.save_raw_video:
        save_video(out_dir / "wm_rollout_multiview.mp4", raw_wm_frames, args.fps)
    if args.save_action_video and action_frames:
        save_video(out_dir / "action_visualization.mp4", action_frames, args.fps)
    if action_history:
        all_actions = np.concatenate(action_history, axis=0)
        np.save(out_dir / "actions.npy", all_actions)
        if args.save_action_plots:
            save_action_axis_plots(out_dir / "action", all_actions)
    write_rewards_csv(out_dir / "rewards.csv", reward_rows)

    max_reward = max(all_rewards) if all_rewards else 0.0
    metadata = {
        "episode_index": int(episode_idx),
        "task": str(task),
        "kir_active": bool(kir_active),
        "kir_initial_frame": int(env.kir_initial_frame) if kir_active else None,
        "initial_frame_idx": int(initial_frame_idx),
        "num_policy_chunks_requested": int(num_policy_chunks),
        "num_generated_frames": int(total_generated),
        "wm_chunk": int(env.chunk),
        "policy_mode": args.policy_mode,
        "success_threshold": float(success_threshold),
        "max_reward": float(max_reward),
        "success": bool(success),
        "annotated_video": str(annotated_video),
    }
    with (out_dir / "rollout_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def write_summary(output_dir: Path, rows: list[dict], cfg, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "episode_index",
        "success",
        "max_reward",
        "num_generated_frames",
        "annotated_video",
        "task",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    success_rate = float(np.mean([bool(row["success"]) for row in rows])) if rows else 0.0
    summary = {
        "config_name": args.config_name,
        "num_episodes": len(rows),
        "num_success": int(sum(bool(row["success"]) for row in rows)),
        "success_rate": success_rate,
        "success_threshold": float(rows[0]["success_threshold"]) if rows else None,
        "num_policy_chunks": int(args.num_policy_chunks or default_num_policy_chunks(cfg)),
        "policy_mode": args.policy_mode,
        "episodes": rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"summary csv: {csv_path}")
    print(f"summary json: {output_dir / 'summary.json'}")
    print(
        f"success: {summary['num_success']}/{summary['num_episodes']} "
        f"= {summary['success_rate']:.4f}"
    )


@torch.no_grad()
def main():
    args = parse_args()
    from rlinf.envs.world_model.world_model_wan_multi_view_env import WanmultiviewEnv

    cfg = load_cfg(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output_dir / "resolved_config.yaml")

    env_cfg = cfg.env.train
    env = WanmultiviewEnv(env_cfg, num_envs=1, seed_offset=0, total_num_processes=1)
    dataset_len = len(env.dataset)
    episode_indices = parse_episode_indices(args, dataset_len)
    num_policy_chunks = args.num_policy_chunks or default_num_policy_chunks(cfg)
    success_threshold = float(getattr(env.cfg, "success_reward_threshold", 0.9))

    print(f"dataset_len={dataset_len}")
    print(f"episode_indices={episode_indices}")
    print(f"num_policy_chunks={num_policy_chunks}")
    print(f"success_threshold={success_threshold}")

    policy = make_policy(cfg, args.device, args.policy_full_weights)

    summaries = []
    for idx, episode_idx in enumerate(episode_indices):
        print(f"=== episode {idx + 1}/{len(episode_indices)} index={episode_idx} ===")
        try:
            metadata = rollout_episode(
                env=env,
                policy=policy,
                episode_idx=episode_idx,
                num_policy_chunks=num_policy_chunks,
                args=args,
                success_threshold=success_threshold,
            )
            summaries.append(metadata)
        except Exception as exc:
            error_dir = args.output_dir / f"episode_{episode_idx:06d}"
            error_dir.mkdir(parents=True, exist_ok=True)
            error_meta = {
                "episode_index": int(episode_idx),
                "success": False,
                "max_reward": 0.0,
                "num_generated_frames": 0,
                "annotated_video": "",
                "task": "",
                "success_threshold": success_threshold,
                "error": repr(exc),
            }
            with (error_dir / "error.json").open("w", encoding="utf-8") as f:
                json.dump(error_meta, f, indent=2, ensure_ascii=False)
            summaries.append(error_meta)
            print(f"episode={episode_idx:06d} failed: {exc!r}")
            if args.stop_on_error:
                raise

    write_summary(args.output_dir, summaries, cfg, args)


if __name__ == "__main__":
    main()
