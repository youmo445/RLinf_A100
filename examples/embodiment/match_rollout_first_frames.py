#!/usr/bin/env python3
"""Interactively match rollout mp4 files by comparing their first frames.

Left side advances only when the answer is "y".
Right side advances after every "y" or "n".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LEFT_DIR = Path(
    "/data/zsq/rollout_openpi_0601_6000_step3_2/videos/chunk-000/"
    "observation.images.base_0_rgb"
)
RIGHT_DIR = Path(
    "/data/zsq/rollout_openpi_0604_rlinf_step3_1/videos/chunk-000/"
    "observation.images.base_0_rgb"
)
OUT_DIR = Path(
    "/data/zsq/rollout_openpi_0604_rlinf_step3_1/"
    "base_0_rgb_match_to_rollout_openpi_0601_6000_step3_2"
)


def list_mp4s(path: Path) -> list[Path]:
    files = sorted(path.glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No mp4 files found under {path}")
    return files


def read_first_frame(path: Path) -> np.ndarray:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        try:
            return np.asarray(reader.get_data(0))[..., :3]
        finally:
            reader.close()
    except Exception as exc:
        raise RuntimeError(f"Cannot decode first frame from {path}") from exc


def resize_keep_aspect(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    width = max(1, int(round(image.width * target_height / image.height)))
    return image.resize((width, target_height), Image.BILINEAR)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str]) -> None:
    font = ImageFont.load_default()
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += 14


def make_panel(
    left_path: Path,
    right_path: Path,
    left_idx: int,
    right_idx: int,
    left_count: int,
    right_count: int,
    display_height: int,
) -> Image.Image:
    left = Image.fromarray(read_first_frame(left_path))
    right = Image.fromarray(read_first_frame(right_path))
    left = resize_keep_aspect(left, display_height)
    right = resize_keep_aspect(right, display_height)

    pad = 12
    title_h = 52
    width = left.width + right.width + pad * 3
    height = display_height + title_h + pad
    panel = Image.new("RGB", (width, height), (18, 18, 18))

    left_x = pad
    right_x = pad * 2 + left.width
    image_y = title_h
    panel.paste(left, (left_x, image_y))
    panel.paste(right, (right_x, image_y))

    draw = ImageDraw.Draw(panel)
    draw_label(
        draw,
        (left_x, 8),
        [
            f"LEFT 0601 [{left_idx + 1}/{left_count}]",
            left_path.name,
        ],
    )
    draw_label(
        draw,
        (right_x, 8),
        [
            f"RIGHT 0604 [{right_idx + 1}/{right_count}]",
            right_path.name,
        ],
    )
    draw.line((right_x - pad // 2, 0, right_x - pad // 2, height), fill=(90, 90, 90), width=2)
    return panel


def load_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    decisions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))
    return decisions


def write_outputs(out_dir: Path, decisions: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = out_dir / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as f:
        for item in decisions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    matched_path = out_dir / "matched_pairs.csv"
    with matched_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "left_idx",
                "left_file",
                "right_idx",
                "right_file",
            ],
        )
        writer.writeheader()
        for item in decisions:
            if item["matched"]:
                writer.writerow(
                    {
                        "left_idx": item["left_idx"],
                        "left_file": item["left_file"],
                        "right_idx": item["right_idx"],
                        "right_file": item["right_file"],
                    }
                )

    unmatched_path = out_dir / "unmatched_right.csv"
    with unmatched_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["right_idx", "right_file", "tested_left_idx", "tested_left_file"])
        writer.writeheader()
        for item in decisions:
            if not item["matched"]:
                writer.writerow(
                    {
                        "right_idx": item["right_idx"],
                        "right_file": item["right_file"],
                        "tested_left_idx": item["left_idx"],
                        "tested_left_file": item["left_file"],
                    }
                )


def state_from_decisions(decisions: list[dict[str, Any]]) -> tuple[int, int]:
    right_idx = len(decisions)
    left_idx = sum(1 for item in decisions if item["matched"])
    return left_idx, right_idx


def show_image(path: Path, mode: str, viewer: str | None) -> None:
    print(f"panel: {path}", flush=True)
    if viewer:
        subprocess.Popen([viewer, str(path)])
        return
    if mode == "save":
        return
    if mode == "cv2":
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read panel image {path}")
        cv2.imshow("rollout first-frame matcher", image)
        cv2.waitKey(1)
        return
    if mode == "matplotlib":
        import matplotlib.pyplot as plt

        image = Image.open(path)
        plt.figure("rollout first-frame matcher")
        plt.clf()
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)
        return
    raise ValueError(f"Unknown display mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", type=Path, default=LEFT_DIR)
    parser.add_argument("--right-dir", type=Path, default=RIGHT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--left-start-index", type=int, default=0)
    parser.add_argument("--right-start-index", type=int, default=0)
    parser.add_argument("--display-height", type=int, default=420)
    parser.add_argument(
        "--display",
        choices=("save", "cv2", "matplotlib"),
        default="save",
        help="save only writes current_pair.png; cv2/matplotlib also opens a window when GUI is available.",
    )
    parser.add_argument("--viewer", type=str, default=None, help="Optional external image viewer command.")
    parser.add_argument("--restart", action="store_true", help="Ignore existing decisions.jsonl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_left_videos = list_mp4s(args.left_dir)
    all_right_videos = list_mp4s(args.right_dir)
    left_videos = all_left_videos[args.left_start_index :]
    right_videos = all_right_videos[args.right_start_index :]
    if not left_videos:
        raise ValueError(f"No left videos from --left-start-index={args.left_start_index}")
    if not right_videos:
        raise ValueError(f"No right videos from --right-start-index={args.right_start_index}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = args.out_dir / "decisions.jsonl"
    decisions = [] if args.restart else load_decisions(decisions_path)
    left_idx, right_idx = state_from_decisions(decisions)

    print(f"left videos:  {len(left_videos)} / {len(all_left_videos)}  {args.left_dir}")
    print(f"right videos: {len(right_videos)} / {len(all_right_videos)}  {args.right_dir}")
    print(f"start index:  left={args.left_start_index}, right={args.right_start_index}")
    print(f"out dir:      {args.out_dir}")
    print(f"resume:       left_idx={left_idx}, right_idx={right_idx}")
    print("input: y = matched, n = not matched, u = undo last, q = quit")

    while right_idx < len(right_videos):
        if left_idx >= len(left_videos):
            print("No left videos remaining; stopping.")
            break

        left_path = left_videos[left_idx]
        right_path = right_videos[right_idx]
        panel = make_panel(
            left_path,
            right_path,
            left_idx,
            right_idx,
            len(left_videos),
            len(right_videos),
            args.display_height,
        )
        panel_path = args.out_dir / "current_pair.png"
        panel.save(panel_path)
        show_image(panel_path, args.display, args.viewer)

        answer = input(
            f"[L {left_idx + 1}/{len(left_videos)} {left_path.name}] "
            f"[R {right_idx + 1}/{len(right_videos)} {right_path.name}] y/n/u/q > "
        ).strip().lower()

        if answer in ("q", "quit", "exit"):
            break
        if answer in ("u", "undo"):
            if decisions:
                removed = decisions.pop()
                write_outputs(args.out_dir, decisions)
                left_idx, right_idx = state_from_decisions(decisions)
                print(f"undone: {removed}")
            else:
                print("nothing to undo")
            continue
        if answer not in ("y", "n"):
            print("Please input y, n, u, or q.")
            continue

        matched = answer == "y"
        decisions.append(
            {
                "matched": matched,
                "left_idx": args.left_start_index + left_idx,
                "left_file": left_path.name,
                "left_path": str(left_path),
                "right_idx": args.right_start_index + right_idx,
                "right_file": right_path.name,
                "right_path": str(right_path),
            }
        )
        write_outputs(args.out_dir, decisions)

        right_idx += 1
        if matched:
            left_idx += 1

    write_outputs(args.out_dir, decisions)
    matched_count = sum(1 for item in decisions if item["matched"])
    print(f"done. decisions={len(decisions)}, matched={matched_count}")
    print(f"matched csv: {args.out_dir / 'matched_pairs.csv'}")
    print(f"all decisions: {args.out_dir / 'decisions.jsonl'}")


if __name__ == "__main__":
    main()
