#!/usr/bin/env python3
"""Build a grid video from matched rollout videos.

Each matched pair is shown as two adjacent tiles:
left = rollout_openpi_0601_6000_step3_2, right = rollout_openpi_0604_rlinf_step3_1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")

import cv2
import numpy as np


CSV_PATH = Path(
    "/data/zsq/rollout_openpi_0604_rlinf_step3_1/"
    "base_0_rgb_match_to_rollout_openpi_0601_6000_step3_2/matched_pairs.csv"
)
LEFT_DIR = Path(
    "/data/zsq/rollout_openpi_0601_6000_step3_2/videos/chunk-000/"
    "observation.images.base_0_rgb"
)
RIGHT_DIR = Path(
    "/data/zsq/rollout_openpi_0604_rlinf_step3_1/videos/chunk-000/"
    "observation.images.base_0_rgb"
)
OUT_PATH = Path(
    "/data/zsq/rollout_openpi_0604_rlinf_step3_1/"
    "base_0_rgb_match_to_rollout_openpi_0601_6000_step3_2/"
    "matched_base_0_rgb_before_after_grid.mp4"
)


def read_pairs(csv_path: Path, left_dir: Path, right_dir: Path) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            left_path = left_dir / row["left_file"]
            right_path = right_dir / row["right_file"]
            if not left_path.exists():
                raise FileNotFoundError(left_path)
            if not right_path.exists():
                raise FileNotFoundError(right_path)
            label = f"L{int(row['left_idx']):02d}/R{int(row['right_idx']):02d}"
            pairs.append((left_path, right_path, label))
    if not pairs:
        raise ValueError(f"No matched pairs found in {csv_path}")
    return pairs


def probe_video(path: Path) -> tuple[int, int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    info = json.loads(subprocess.check_output(cmd, text=True))
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    frame_count = int(stream.get("nb_frames") or 0)
    num, den = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / max(float(den), 1.0)
    return width, height, frame_count, fps


class VideoSlot:
    def __init__(self, path: Path, tile_size: int):
        self.path = path
        self.tile_size = tile_size
        self.width, self.height, self.frame_count, self.fps = probe_video(path)
        self.frame_bytes = self.width * self.height * 3
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.last = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    def read(self) -> np.ndarray:
        if self.proc.stdout is None:
            return self.last
        raw = self.proc.stdout.read(self.frame_bytes)
        if len(raw) == self.frame_bytes:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
            self.last = cv2.resize(frame, (self.tile_size, self.tile_size), interpolation=cv2.INTER_AREA)
        return self.last

    def release(self) -> None:
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.38,
    color: tuple[int, int, int] = (235, 235, 235),
) -> None:
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def build_frame(
    frame_idx: int,
    pairs: list[tuple[VideoSlot, VideoSlot, str]],
    *,
    pairs_per_row: int,
    tile_size: int,
    label_h: int,
    cell_gap: int,
    pair_gap: int,
    row_gap: int,
    margin: int,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    draw_text(canvas, f"frame {frame_idx:04d}", margin, 18, scale=0.5, color=(255, 255, 255))

    pair_w = tile_size * 2 + cell_gap
    row_h = label_h + tile_size
    content_y0 = margin + 18

    for idx, (left_slot, right_slot, pair_label) in enumerate(pairs):
        row = idx // pairs_per_row
        col = idx % pairs_per_row
        x0 = margin + col * (pair_w + pair_gap)
        y0 = content_y0 + row * (row_h + row_gap)

        left_frame = left_slot.read()
        right_frame = right_slot.read()

        draw_text(canvas, f"{pair_label}  0601", x0, y0 + 14, scale=0.34, color=(180, 220, 255))
        draw_text(canvas, left_slot.path.name.replace(".mp4", ""), x0, y0 + 30, scale=0.32)
        rx = x0 + tile_size + cell_gap
        draw_text(canvas, "0604 RLinf", rx, y0 + 14, scale=0.34, color=(255, 210, 170))
        draw_text(canvas, right_slot.path.name.replace(".mp4", ""), rx, y0 + 30, scale=0.32)

        iy = y0 + label_h
        canvas[iy : iy + tile_size, x0 : x0 + tile_size] = left_frame
        canvas[iy : iy + tile_size, rx : rx + tile_size] = right_frame
        cv2.rectangle(canvas, (x0, iy), (x0 + tile_size - 1, iy + tile_size - 1), (80, 120, 150), 1)
        cv2.rectangle(canvas, (rx, iy), (rx + tile_size - 1, iy + tile_size - 1), (150, 110, 80), 1)

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--left-dir", type=Path, default=LEFT_DIR)
    parser.add_argument("--right-dir", type=Path, default=RIGHT_DIR)
    parser.add_argument("--out-path", type=Path, default=OUT_PATH)
    parser.add_argument("--pairs-per-row", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=144)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_pairs = read_pairs(args.csv_path, args.left_dir, args.right_dir)
    pairs = [(VideoSlot(left, args.tile_size), VideoSlot(right, args.tile_size), label) for left, right, label in raw_pairs]

    try:
        max_frames = max(max(left.frame_count, right.frame_count) for left, right, _ in pairs)
        if args.max_frames is not None:
            max_frames = min(max_frames, args.max_frames)

        label_h = 36
        cell_gap = 4
        pair_gap = 12
        row_gap = 12
        margin = 16
        rows = math.ceil(len(pairs) / args.pairs_per_row)
        pair_w = args.tile_size * 2 + cell_gap
        row_h = label_h + args.tile_size
        width = margin * 2 + args.pairs_per_row * pair_w + (args.pairs_per_row - 1) * pair_gap
        height = margin * 2 + 18 + rows * row_h + (rows - 1) * row_gap
        width += width % 2
        height += height % 2

        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.out_path), fourcc, args.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {args.out_path}")

        print(f"pairs={len(pairs)} rows={rows} size={width}x{height} frames={max_frames} fps={args.fps}")
        for frame_idx in range(max_frames):
            frame = build_frame(
                frame_idx,
                pairs,
                pairs_per_row=args.pairs_per_row,
                tile_size=args.tile_size,
                label_h=label_h,
                cell_gap=cell_gap,
                pair_gap=pair_gap,
                row_gap=row_gap,
                margin=margin,
                width=width,
                height=height,
            )
            writer.write(frame)
            if frame_idx % 50 == 0 or frame_idx == max_frames - 1:
                print(f"wrote {frame_idx + 1}/{max_frames}", flush=True)
        writer.release()
    finally:
        for left, right, _ in pairs:
            left.release()
            right.release()

    print(f"output: {args.out_path}")


if __name__ == "__main__":
    main()
