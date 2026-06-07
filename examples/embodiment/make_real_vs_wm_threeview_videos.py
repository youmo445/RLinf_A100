#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


REAL_CAMERAS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)


def ffprobe_video(path: Path) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    info = json.loads(out)
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    rate = stream.get("avg_frame_rate", "0/1")
    num, den = rate.split("/")
    fps = float(num) / max(float(den), 1.0)
    return width, height, fps


def read_video_rgb_ffmpeg(path: Path) -> list[np.ndarray]:
    width, height, _ = ffprobe_video(path)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    frame_size = width * height * 3
    if len(raw) % frame_size != 0:
        raise RuntimeError(
            f"Decoded byte count is not divisible by frame size for {path}: "
            f"bytes={len(raw)} frame_size={frame_size}"
        )
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((-1, height, width, 3))
    return [frame.copy() for frame in arr]


def resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def add_header(frame: np.ndarray, text: str, header_h: int = 32) -> np.ndarray:
    out = np.zeros((frame.shape[0] + header_h, frame.shape[1], 3), dtype=np.uint8)
    out[header_h:] = frame
    cv2.putText(
        out,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def pad_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    if frame.shape[0] > height:
        return frame[:height]
    pad = np.zeros((height - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    return np.concatenate([frame, pad], axis=0)


def frame_at(frames: list[np.ndarray], index: int) -> np.ndarray:
    if not frames:
        raise RuntimeError("empty video")
    return frames[min(index, len(frames) - 1)]


def make_real_column(
    camera_frames: list[list[np.ndarray]],
    frame_idx: int,
    view_size: tuple[int, int],
) -> np.ndarray:
    views = [resize_rgb(frame_at(frames, frame_idx), view_size) for frames in camera_frames]
    return np.concatenate(views, axis=0)


def make_episode_video(
    episode_idx: int,
    real_root: Path,
    gen_root: Path,
    out_dir: Path,
    fps: int,
    view_size: tuple[int, int],
) -> dict[str, int | str]:
    ep_name = f"episode_{episode_idx:06d}.mp4"
    real_paths = [
        real_root / "videos" / "chunk-000" / camera / ep_name for camera in REAL_CAMERAS
    ]
    gen_path = gen_root / f"episode_{episode_idx:06d}" / "wm_rollout_reward_annotated.mp4"

    for path in real_paths + [gen_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    real_frames_by_cam = [read_video_rgb_ffmpeg(path) for path in real_paths]
    gen_frames = read_video_rgb_ffmpeg(gen_path)

    real_len = min(len(frames) for frames in real_frames_by_cam)
    gen_len = len(gen_frames)
    out_len = max(real_len, gen_len)

    gen_w = gen_frames[0].shape[1]
    gen_h = gen_frames[0].shape[0]
    real_base_h = view_size[1] * len(REAL_CAMERAS)
    column_h = max(real_base_h, gen_h)
    column_w = max(view_size[0], gen_w)
    sep_w = 8

    out_path = out_dir / f"episode_{episode_idx:06d}_real_vs_wm.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for i in range(out_len):
            real_col = make_real_column(real_frames_by_cam, i, view_size)
            real_col = pad_to_height(real_col, column_h)
            if real_col.shape[1] != column_w:
                real_col = cv2.copyMakeBorder(
                    real_col,
                    0,
                    0,
                    0,
                    column_w - real_col.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

            gen_col = frame_at(gen_frames, i)
            if gen_col.shape[1] != column_w:
                gen_col = resize_rgb(gen_col, (column_w, gen_col.shape[0]))
            gen_col = pad_to_height(gen_col, column_h)

            left = add_header(real_col, f"REAL ep={episode_idx:06d} frame={min(i, real_len - 1):04d}")
            right = add_header(gen_col, f"WM+VLA ep={episode_idx:06d} frame={min(i, gen_len - 1):04d}")
            sep = np.zeros((left.shape[0], sep_w, 3), dtype=np.uint8)
            writer.append_data(np.concatenate([left, sep, right], axis=1))
    finally:
        writer.close()

    return {
        "episode": episode_idx,
        "real_frames": real_len,
        "gen_frames": gen_len,
        "out_frames": out_len,
        "out_path": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, default=Path("/data/zsq/rollout_openpi_0601_6000_step3_2"))
    parser.add_argument(
        "--gen-root",
        type=Path,
        default=Path("/data/zsq/RLinf/examples/embodiment/pi05_wm_batch_rollouts"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/zsq/RLinf/examples/embodiment/real_vs_wm_threeview_0601_step3_2"),
    )
    parser.add_argument("--episodes", type=int, default=46)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--view-width", type=int, default=224)
    parser.add_argument("--view-height", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for episode_idx in range(args.episodes):
        result = make_episode_video(
            episode_idx=episode_idx,
            real_root=args.real_root,
            gen_root=args.gen_root,
            out_dir=args.out_dir,
            fps=args.fps,
            view_size=(args.view_width, args.view_height),
        )
        results.append(result)
        print(
            f"[{episode_idx + 1:02d}/{args.episodes}] "
            f"ep={episode_idx:06d} real={result['real_frames']} gen={result['gen_frames']} "
            f"out={result['out_frames']} -> {result['out_path']}",
            flush=True,
        )
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] {args.out_dir}")


if __name__ == "__main__":
    main()
