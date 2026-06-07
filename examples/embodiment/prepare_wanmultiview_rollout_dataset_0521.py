#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare_wanmultiview_rollout_dataset import (
    episode_paths,
    load_json,
    parse_episodes,
    read_video_frame_rgb,
    stack_views,
)


DEFAULT_TASK = (
    "use the left arm to put the orange square into the green box and use the "
    "right arm to put the blue square into the green box"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a minimal NpyTrajectoryDatasetWrapper-compatible ARXX5 "
            "rollout dataset with only image/action/task keys."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        default=[Path("/data/zsq/arx_policy_rollout_0520")],
        help="One or more LeRobot dataset roots.",
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=None,
        help="Alias for --dataset-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/zsq/RLinf/examples/embodiment/config/test_data/"
            "wanmultiview_arxx5_0521_rollout"
        ),
    )
    parser.add_argument("--episodes", type=str, default="all")
    parser.add_argument("--view-height", type=int, default=160)
    parser.add_argument("--view-width", type=int, default=224)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    return parser.parse_args()


def build_episode(
    dataset_root: Path,
    episode_index: int,
    chunk_size: int,
    view_height: int,
    view_width: int,
    task: str,
):
    parquet_path, video_paths = episode_paths(dataset_root, episode_index, chunk_size)
    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)

    first_views = {
        camera: read_video_frame_rgb(path, 0) for camera, path in video_paths.items()
    }
    first_image = stack_views(first_views, view_height, view_width)

    frames = []
    for local_idx, action in enumerate(actions):
        frame = {
            "action": action,
            "task": task,
        }
        if local_idx == 0:
            frame["image"] = first_image
        frames.append(frame)
    return frames


def main():
    args = parse_args()
    dataset_roots = args.dataset_roots or args.dataset_root

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_roots": [str(path) for path in dataset_roots],
        "episodes": args.episodes,
        "total_episodes": 0,
        "view_height": args.view_height,
        "view_width": args.view_width,
        "image_key": "image",
        "action_key": "action",
        "task_key": "task",
        "task": args.task,
        "image_policy": "first_frame_only",
        "frame_keys": {
            "first_frame": ["image", "action", "task"],
            "other_frames": ["action", "task"],
        },
        "episode_mapping": [],
    }

    output_episode_index = 0
    for source_id, dataset_root in enumerate(dataset_roots):
        info = load_json(dataset_root / "meta" / "info.json")
        total_episodes = int(info["total_episodes"])
        chunk_size = int(info.get("chunks_size", 1000))
        episodes = parse_episodes(args.episodes, total_episodes)

        for episode_index in episodes:
            frames = build_episode(
                dataset_root=dataset_root,
                episode_index=episode_index,
                chunk_size=chunk_size,
                view_height=args.view_height,
                view_width=args.view_width,
                task=args.task,
            )
            out_path = args.output_dir / f"episode_{output_episode_index:06d}.npy"
            np.save(out_path, np.asarray(frames, dtype=object), allow_pickle=True)
            manifest["episode_mapping"].append(
                {
                    "output_episode_index": output_episode_index,
                    "source_id": source_id,
                    "dataset_root": str(dataset_root),
                    "source_episode_index": episode_index,
                    "frames": len(frames),
                }
            )
            print(
                f"wrote {out_path} source={dataset_root} "
                f"episode={episode_index:06d} frames={len(frames)}"
            )
            output_episode_index += 1

    manifest["total_episodes"] = output_episode_index

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
