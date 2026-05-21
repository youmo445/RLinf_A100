#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare_wanmultiview_rollout_dataset import (
    episode_paths,
    load_json,
    read_task_map,
    read_video_frame_rgb,
    stack_views,
)


DATASET_ROOTS = [
    Path("/data/zsq/arx_policy_rollout_new"),
    Path("/data/zsq/arx_policy_rollout_new_continue"),
]
OUTPUT_DIR = Path(
    "/data/zsq/RLinf/examples/embodiment/config/test_data/"
    "wanmultiview_arxx5_stack_blocks_0516"
)
VIEW_HEIGHT = 160
VIEW_WIDTH = 224


def iter_episode_indices(dataset_root: Path):
    info = load_json(dataset_root / "meta" / "info.json")
    total_episodes = int(info["total_episodes"])
    chunk_size = int(info.get("chunks_size", 1000))
    return range(total_episodes), chunk_size


def build_one_episode(
    dataset_root: Path,
    source_episode_index: int,
    output_episode_index: int,
    chunk_size: int,
    task_map: dict[int, str],
):
    parquet_path, video_paths = episode_paths(
        dataset_root, source_episode_index, chunk_size
    )
    columns = [
        "action",
        "observation.state",
        "frame_index",
        "task_index",
    ]
    table = pq.read_table(
        parquet_path,
        columns=columns,
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
    task = task_map.get(int(task_indices[0]), "")

    first_views = {
        camera: read_video_frame_rgb(path, 0) for camera, path in video_paths.items()
    }
    first_image = stack_views(first_views, VIEW_HEIGHT, VIEW_WIDTH)

    frames = []
    for local_idx in range(len(actions)):
        frame = {
            "action": actions[local_idx],
            "state": states[local_idx],
            "frame_index": np.int64(frame_indices[local_idx]),
            "episode_index": np.int64(output_episode_index),
            "source_episode_index": np.int64(source_episode_index),
            "source_dataset_root": str(dataset_root),
            "task": task,
            "instruction": task,
        }
        if local_idx == 0:
            frame["image"] = first_image
        frames.append(frame)
    return frames


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_episode_index = 0
    sources = []

    for dataset_root in DATASET_ROOTS:
        episode_indices, chunk_size = iter_episode_indices(dataset_root)
        task_map = read_task_map(dataset_root)
        start_output_index = output_episode_index

        for source_episode_index in episode_indices:
            frames = build_one_episode(
                dataset_root=dataset_root,
                source_episode_index=source_episode_index,
                output_episode_index=output_episode_index,
                chunk_size=chunk_size,
                task_map=task_map,
            )
            out_path = OUTPUT_DIR / f"episode_{output_episode_index:06d}.npy"
            np.save(out_path, np.asarray(frames, dtype=object), allow_pickle=True)
            print(
                f"wrote {out_path} source={dataset_root} "
                f"source_episode={source_episode_index:06d} frames={len(frames)}"
            )
            output_episode_index += 1

        sources.append(
            {
                "dataset_root": str(dataset_root),
                "source_episode_start": 0,
                "source_episode_count": output_episode_index - start_output_index,
                "output_episode_start": start_output_index,
                "output_episode_end": output_episode_index,
            }
        )

    manifest = {
        "dataset_roots": [str(path) for path in DATASET_ROOTS],
        "sources": sources,
        "total_episodes": output_episode_index,
        "store_all_images": False,
        "view_height": VIEW_HEIGHT,
        "view_width": VIEW_WIDTH,
        "image_key": "image",
        "action_key": "action",
        "state_key": "state",
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
