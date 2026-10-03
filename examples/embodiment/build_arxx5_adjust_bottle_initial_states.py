from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


TASK = "Straighten the bottle that has fallen over on the table."
CAMERAS = (
    "observation.images.base_0",
    "observation.images.left_wrist_0",
    "observation.images.right_wrist_0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the adjust-bottle episodes from the six-task RLinf NPY "
            "set with the 0809 LeRobot dataset."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/data/zsq/RLinf_A100/examples/embodiment/config/test_data/"
            "wanmultiview_arxx5_newcam_acwm_6tasks_0717_0720_0728_all729"
        ),
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=Path("/data/zsq/0809pace2_eef_lerobot_v21_10hz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/zsq/RLinf_A100/examples/embodiment/config/test_data/"
            "wanmultiview_arxx5_newcam_adjust_bottle_6tasks_plus_0809"
        ),
    )
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_rgb_first_frame(path: Path, size: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to decode the first frame: {path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame_rgb.shape[:2] != (size, size):
        frame_rgb = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame_rgb, dtype=np.uint8)


def validate_vector(value, name: str, source: Path) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (14,):
        raise ValueError(f"Expected {name} shape (14,), got {vector.shape}: {source}")
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite {name}: {source}")
    return vector


def save_episode(path: Path, frames: list[dict]) -> None:
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as file:
        np.save(file, np.asarray(frames, dtype=object), allow_pickle=True)
    temporary.replace(path)


def convert_old_episode(
    source_path: Path,
    output_path: Path,
    episode_index: int,
) -> int:
    source = np.load(source_path, allow_pickle=True)
    frames = []
    for frame_index, raw_frame in enumerate(source):
        frame = raw_frame if isinstance(raw_frame, dict) else raw_frame.item()
        converted = dict(frame)
        converted["action"] = validate_vector(frame["action"], "action", source_path)
        converted["state"] = validate_vector(frame["state"], "state", source_path)
        converted["reward"] = np.float32(frame.get("reward", 0.0))
        converted["episode_index"] = np.int64(episode_index)
        converted["frame_index"] = np.int64(frame_index)
        converted["task_index"] = np.int64(0)
        converted["task"] = TASK
        converted["instruction"] = TASK
        if frame_index == 0:
            image = np.asarray(frame["image"], dtype=np.uint8)
            if image.shape != (672, 224, 3):
                raise ValueError(f"Unexpected old image shape {image.shape}: {source_path}")
            converted["image"] = np.ascontiguousarray(image)
        else:
            converted.pop("image", None)
        frames.append(converted)
    save_episode(output_path, frames)
    return len(frames)


def convert_new_episode(
    dataset_root: Path,
    source_episode_index: int,
    output_path: Path,
    episode_index: int,
    view_size: int,
) -> tuple[int, Path]:
    chunk = source_episode_index // 1000
    parquet_path = dataset_root / (
        f"data/chunk-{chunk:03d}/episode_{source_episode_index:06d}.parquet"
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    rows = pq.read_table(
        parquet_path,
        columns=["action", "observation.state", "frame_index", "task_index"],
    ).to_pylist()
    if not rows:
        raise ValueError(f"Empty episode: {parquet_path}")

    views = []
    for camera in CAMERAS:
        video_path = dataset_root / (
            f"videos/chunk-{chunk:03d}/{camera}/episode_{source_episode_index:06d}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        views.append(read_rgb_first_frame(video_path, view_size))
    stacked_image = np.concatenate(views, axis=0)

    frames = []
    last_index = len(rows) - 1
    for frame_index, row in enumerate(rows):
        frame = {
            "action": validate_vector(row["action"], "action", parquet_path),
            "state": validate_vector(
                row["observation.state"], "observation.state", parquet_path
            ),
            "reward": np.float32(1.0 if frame_index == last_index else 0.0),
            "episode_index": np.int64(episode_index),
            "frame_index": np.int64(frame_index),
            "task_index": np.int64(0),
            "task": TASK,
            "instruction": TASK,
        }
        if frame_index == 0:
            frame["image"] = stacked_image
        frames.append(frame)
    save_episode(output_path, frames)
    return len(frames), parquet_path


def rounded_p90_plus_chunk(lengths: list[int], chunk: int = 16) -> int:
    target = float(np.percentile(np.asarray(lengths), 90)) + chunk
    return int(math.ceil(target / chunk) * chunk)


def main() -> None:
    args = parse_args()
    source_manifest = json.loads(
        (args.source_root / "manifest.json").read_text(encoding="utf-8")
    )
    old_selected = [
        row for row in source_manifest.get("episodes", []) if row.get("task") == TASK
    ]
    if not old_selected:
        raise RuntimeError(f"No {TASK!r} episodes found in {args.source_root}")

    new_info = json.loads((args.new_root / "meta/info.json").read_text(encoding="utf-8"))
    new_tasks = read_jsonl(args.new_root / "meta/tasks.jsonl")
    if [row["task"] for row in new_tasks] != [TASK]:
        raise ValueError(f"Expected only {TASK!r}, got {new_tasks}")
    new_episodes = read_jsonl(args.new_root / "meta/episodes.jsonl")
    if len(new_episodes) != int(new_info["total_episodes"]):
        raise ValueError("meta/info.json total_episodes disagrees with episodes.jsonl")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("episode_*.npy"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output contains {len(existing)} NPY files; pass --overwrite: {args.output_dir}"
        )

    mapping = []
    lengths = []
    merged_index = 0
    for row in old_selected:
        source_path = args.source_root / row["output_file"]
        output_file = f"episode_{merged_index:06d}.npy"
        length = convert_old_episode(
            source_path, args.output_dir / output_file, merged_index
        )
        lengths.append(length)
        mapping.append(
            {
                "episode_index": merged_index,
                "task_index": 0,
                "group_id": 1,
                "reward_group_id": 1,
                "task": TASK,
                "length": length,
                "source_kind": "old_6tasks_adjust_bottle",
                "source_episode_index": int(row["episode_index"]),
                "source_path": str(source_path),
                "reward_source": row.get("reward_source"),
                "output_file": output_file,
            }
        )
        merged_index += 1

    for ordinal, row in enumerate(new_episodes, start=1):
        source_episode_index = int(row["episode_index"])
        if row.get("tasks") != [TASK]:
            raise ValueError(f"Unexpected task for episode {source_episode_index}: {row}")
        output_file = f"episode_{merged_index:06d}.npy"
        length, parquet_path = convert_new_episode(
            args.new_root,
            source_episode_index,
            args.output_dir / output_file,
            merged_index,
            args.view_size,
        )
        if length != int(row["length"]):
            raise ValueError(
                f"Episode length mismatch for {source_episode_index}: "
                f"{length} != {row['length']}"
            )
        lengths.append(length)
        mapping.append(
            {
                "episode_index": merged_index,
                "task_index": 0,
                "group_id": 1,
                "reward_group_id": 1,
                "task": TASK,
                "length": length,
                "source_kind": "new_0809_adjust_bottle",
                "source_episode_index": source_episode_index,
                "source_path": str(parquet_path),
                "reward_source": "successful_demo_last_frame",
                "output_file": output_file,
            }
        )
        merged_index += 1
        print(f"[new] {ordinal}/{len(new_episodes)} -> {output_file}")

    stats = {
        "count": len(lengths),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p90_plus_16_rounded_to_16": rounded_p90_plus_chunk(lengths),
    }
    manifest = {
        "source_datasets": [str(args.source_root), str(args.new_root)],
        "output_dir": str(args.output_dir),
        "format": "RLinf NPY trajectory object array",
        "total_episodes": len(mapping),
        "total_frames": int(sum(lengths)),
        "fps": int(new_info["fps"]),
        "image_policy": "only frame zero stores image",
        "image_dtype": "uint8 RGB",
        "view_order_top_to_bottom": ["head", "left_wrist", "right_wrist"],
        "source_camera_keys": {
            "head": CAMERAS[0],
            "left_wrist": CAMERAS[1],
            "right_wrist": CAMERAS[2],
        },
        "per_view_shape_hwc": [args.view_size, args.view_size, 3],
        "stacked_image_shape_hwc": [3 * args.view_size, args.view_size, 3],
        "action_shape": [14],
        "state_shape": [14],
        "dataset_action_semantics": "EEF pose retained; WM config intentionally reads state",
        "dataset_state_semantics": "14-D joint/gripper state used by WM action map",
        "reward_policy": "old rewards preserved; new 0809 demonstrations use final-frame success",
        "length_statistics": stats,
        "tasks": [
            {
                "task_index": 0,
                "group_id": 1,
                "task": TASK,
                "episodes": len(mapping),
                "frames": int(sum(lengths)),
                "old_6tasks_episodes": len(old_selected),
                "new_0809_episodes": len(new_episodes),
                "episode_indices": list(range(len(mapping))),
            }
        ],
        "episodes": mapping,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[done] episodes={len(mapping)} output={args.output_dir}")


if __name__ == "__main__":
    main()
