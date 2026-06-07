#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


DEFAULT_SOURCE_DIR = Path(
    "/data/zsq/RoboTwin_dataset/dataset/pi05_eval_rollout/data_ckpt_10000_step5"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/zsq/RLinf/examples/embodiment/config/test_data/"
    "wanmultiview_robotwin_0602"
)
CAMERAS = ("head_camera", "left_camera", "right_camera")
DEFAULT_TASK = "Pick up mug, rotate it, place it on the table, and hang it on rack."


def episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode[_-]?(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def decode_rgb(raw) -> np.ndarray:
    if isinstance(raw, np.bytes_):
        raw = raw.tobytes()
    elif isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    elif not isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def read_resized_first_view(f: h5py.File, camera: str, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(decode_rgb(f[f"observation/{camera}/rgb"][0]))
    if image.size != (image_size[1], image_size[0]):
        image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def build_episode(path: Path, image_size: tuple[int, int], task_override: str | None) -> tuple[list[dict], dict]:
    with h5py.File(path, "r") as f:
        actions = np.asarray(f["joint_action/vector"], dtype=np.float32)
        task = task_override or f.attrs.get("instruction", DEFAULT_TASK)
        if isinstance(task, bytes):
            task = task.decode("utf-8")
        views = [read_resized_first_view(f, camera, image_size) for camera in CAMERAS]
        first_image = np.concatenate(views, axis=0)

        frames = []
        for idx, action in enumerate(actions):
            frame = {
                "action": action.astype(np.float32),
                "task": str(task),
            }
            if idx == 0:
                frame["image"] = first_image
            frames.append(frame)

        meta = {
            "source_path": str(path),
            "source_episode_id": int(f.attrs.get("episode_id", episode_sort_key(path)[0])),
            "source_episode_len": int(f.attrs.get("episode_len", len(actions))),
            "success": bool(f.attrs.get("success", False)),
            "task_name": str(f.attrs.get("task_name", "")),
            "task_config": str(f.attrs.get("task_config", "")),
            "instruction": str(task),
            "frames": len(frames),
        }
    return frames, meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare RoboTwin rollout HDF5 initial-image npy dataset for WanmultiviewEnv."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--view-height", type=int, default=224)
    parser.add_argument("--view-width", type=int, default=224)
    parser.add_argument("--task", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(args.source_dir.glob("*.hdf5"), key=episode_sort_key)
    if args.limit is not None and args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No HDF5 files found under {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "total_episodes": 0,
        "limit": args.limit,
        "image_key": "image",
        "action_key": "action",
        "task_key": "task",
        "action_source": "joint_action/vector",
        "image_policy": "first_frame_only",
        "camera_order": list(CAMERAS),
        "view_height": args.view_height,
        "view_width": args.view_width,
        "stacked_image_shape": [args.view_height * len(CAMERAS), args.view_width, 3],
        "episode_mapping": [],
    }

    for out_idx, path in enumerate(files):
        frames, meta = build_episode(
            path=path,
            image_size=(args.view_height, args.view_width),
            task_override=args.task,
        )
        out_path = args.output_dir / f"episode_{out_idx:06d}.npy"
        np.save(out_path, np.asarray(frames, dtype=object), allow_pickle=True)
        meta["output_episode_index"] = out_idx
        meta["output_path"] = str(out_path)
        manifest["episode_mapping"].append(meta)
        print(f"wrote {out_path} source={path.name} frames={len(frames)}")

    manifest["total_episodes"] = len(files)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
