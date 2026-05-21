#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


CAMERA_KEYS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build NpyTrajectoryDatasetWrapper-compatible ARXX5 rollout data."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/zsq/stack_blocks_bimanual_0513"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/zsq/RLinf/examples/embodiment/config/test_data/"
            "wanmultiview_arxx5_stack_blocks"
        ),
    )
    parser.add_argument("--episodes", type=str, default="all")
    parser.add_argument("--view-height", type=int, default=160)
    parser.add_argument("--view-width", type=int, default=224)
    parser.add_argument(
        "--store-all-images",
        action="store_true",
        help=(
            "Store stacked images for every frame. The default stores only frame 0 "
            "to keep the rollout reset dataset small; full episode actions/states "
            "are always stored."
        ),
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_episodes(value: str, total_episodes: int):
    if value == "all":
        return list(range(total_episodes))
    episodes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, end = item.split(":", 1)
            episodes.extend(range(int(start), int(end)))
        else:
            episodes.append(int(item))
    return sorted(set(episodes))


def read_task_map(dataset_root: Path):
    path = dataset_root / "meta" / "tasks.jsonl"
    task_map = {}
    if not path.exists():
        return task_map
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_map[int(row["task_index"])] = row.get("task", "")
    return task_map


def episode_paths(dataset_root: Path, episode_index: int, chunk_size: int):
    chunk = episode_index // chunk_size
    parquet = (
        dataset_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    videos = {
        name: (
            dataset_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )
        for name, video_key in CAMERA_KEYS.items()
    }
    return parquet, videos


def read_video_rgb(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    ok, frame_bgr = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ok, frame_bgr = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return frames


def read_video_frame_rgb(path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return read_video_frame_rgb_fallback(path, frame_index)
    frame_bgr = None
    ok = False
    # Some mp4 files do not seek reliably with OpenCV on this machine. Sequential
    # read is slower but robust for the reset frame and small episode builds.
    for _ in range(int(frame_index) + 1):
        ok, frame_bgr = cap.read()
        if not ok:
            break
    cap.release()
    if not ok or frame_bgr is None:
        return read_video_frame_rgb_fallback(path, frame_index)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def read_video_frame_rgb_fallback(path: Path, frame_index: int):
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(path), ctx=cpu(0))
        if frame_index >= len(vr):
            raise IndexError(f"frame {frame_index} out of range for {path}")
        return vr[int(frame_index)].asnumpy()
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        for idx, frame in enumerate(iio.imiter(path)):
            if idx == int(frame_index):
                return np.asarray(frame, dtype=np.uint8)
    except Exception:
        pass

    raise RuntimeError(f"Cannot decode frame {frame_index} from video: {path}")


def stack_views(frames_by_camera, view_height: int, view_width: int):
    resized = []
    for camera in ("head", "left_wrist", "right_wrist"):
        frame = frames_by_camera[camera]
        resized.append(
            cv2.resize(frame, (view_width, view_height), interpolation=cv2.INTER_AREA)
        )
    return np.concatenate(resized, axis=0).astype(np.uint8)


def main():
    args = parse_args()
    info = load_json(args.dataset_root / "meta" / "info.json")
    total_episodes = int(info["total_episodes"])
    chunk_size = int(info.get("chunks_size", 1000))
    episodes = parse_episodes(args.episodes, total_episodes)
    task_map = read_task_map(args.dataset_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_root": str(args.dataset_root),
        "episodes": episodes,
        "store_all_images": bool(args.store_all_images),
        "view_height": args.view_height,
        "view_width": args.view_width,
        "image_key": "image",
        "action_key": "action",
        "state_key": "state",
    }

    for episode_index in episodes:
        parquet_path, video_paths = episode_paths(
            args.dataset_root, episode_index, chunk_size
        )
        table = pq.read_table(
            parquet_path,
            columns=[
                "action",
                "observation.state",
                "observation.ee_pose",
                "frame_index",
                "task_index",
            ],
        )
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        ee_poses = np.asarray(
            table["observation.ee_pose"].to_pylist(), dtype=np.float32
        )
        frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        task = task_map.get(int(task_indices[0]), "")

        if args.store_all_images:
            videos = {camera: read_video_rgb(path) for camera, path in video_paths.items()}
            image_count = min(len(videos[camera]) for camera in videos)
        else:
            first_views = {
                camera: read_video_frame_rgb(path, 0)
                for camera, path in video_paths.items()
            }
            image_count = 1

        frames = []
        for local_idx in range(len(actions)):
            frame = {
                "action": actions[local_idx],
                "state": states[local_idx],
                "observation.ee_pose": ee_poses[local_idx],
                "frame_index": np.int64(frame_indices[local_idx]),
                "episode_index": np.int64(episode_index),
                "task": task,
                "instruction": task,
            }
            if args.store_all_images and local_idx < image_count:
                views = {camera: videos[camera][local_idx] for camera in video_paths}
                frame["image"] = stack_views(views, args.view_height, args.view_width)
            elif local_idx == 0:
                frame["image"] = stack_views(
                    first_views, args.view_height, args.view_width
                )
            frames.append(frame)

        out_path = args.output_dir / f"episode_{episode_index:06d}.npy"
        np.save(out_path, np.asarray(frames, dtype=object), allow_pickle=True)
        print(f"wrote {out_path} frames={len(frames)} image_frames={image_count}")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
