from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


TASK = "Pack the phone."
CAMERAS = (
    "observation.images.base_0",
    "observation.images.left_wrist_0",
    "observation.images.right_wrist_0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the four-stage pack-phone LeRobot dataset to RLinf NPY "
            "initial-state trajectories and duplicate one stage-3 episode so "
            "that every reward group contains exactly 100 episodes."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/data/zsq/packphone_0804_0810_0811_stage_combined_stage1fix324_"
            "eef_lerobot_v21_10hz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/zsq/RLinf_A100/examples/embodiment/config/test_data/"
            "wanmultiview_arxx5_packphone_0804_0810_0811_"
            "stage_combined_stage1fix324_initial_states"
        ),
    )
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--target-per-stage", type=int, default=100)
    parser.add_argument("--duplicate-stage", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_vector(value, name: str, source: Path) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (14,):
        raise ValueError(f"Expected {name} shape (14,), got {vector.shape}: {source}")
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite {name}: {source}")
    return vector


def read_rgb_first_frame(path: Path, size: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to decode first frame: {path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame_rgb.shape[:2] != (size, size):
        frame_rgb = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame_rgb, dtype=np.uint8)


def save_episode(path: Path, frames: list[dict]) -> None:
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as file:
        np.save(file, np.asarray(frames, dtype=object), allow_pickle=True)
    temporary.replace(path)


def aligned_p90_plus_chunk(lengths: list[int], chunk: int = 16) -> int:
    target = float(np.percentile(np.asarray(lengths), 90)) + chunk
    return int(math.ceil(target / chunk) * chunk)


def convert_episode(
    dataset_root: Path,
    source_episode_index: int,
    output_path: Path,
    output_episode_index: int,
    stage: int,
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
        columns=["action", "observation.state"],
    ).to_pylist()
    if not rows:
        raise ValueError(f"Empty episode: {parquet_path}")

    views = []
    for camera in CAMERAS:
        video_path = dataset_root / (
            f"videos/chunk-{chunk:03d}/{camera}/"
            f"episode_{source_episode_index:06d}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        views.append(read_rgb_first_frame(video_path, view_size))
    stacked_image = np.concatenate(views, axis=0)

    frames = []
    group_key = f"stage_{stage}"
    source_stage_task = f"Pack the phone, stage {stage}."
    for frame_index, row in enumerate(rows):
        state = validate_vector(
            row["observation.state"], "observation.state", parquet_path
        )
        eef_action = validate_vector(row["action"], "action", parquet_path)
        frame = {
            # The WM/FK path consumes absolute joint targets as actions.
            "action": state.copy(),
            "state": state,
            "eef_action": eef_action,
            "reward": np.float32(0.0),
            "frame_index": np.int64(frame_index),
            "episode_index": np.int64(output_episode_index),
            "task_index": np.int64(stage - 1),
            "task": TASK,
            "instruction": TASK,
            "reward_group_id": np.int64(stage),
            "reward_group_key": group_key,
            "source_stage_task": source_stage_task,
        }
        if frame_index == 0:
            frame["image"] = stacked_image
        frames.append(frame)

    save_episode(output_path, frames)
    return len(frames), parquet_path


def main() -> None:
    args = parse_args()
    report_path = args.dataset_root / "conversion_report.json"
    info_path = args.dataset_root / "meta/info.json"
    episodes_path = args.dataset_root / "meta/episodes.jsonl"
    for path in (report_path, info_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_rows = read_jsonl(episodes_path)
    if int(info["total_episodes"]) != len(episode_rows):
        raise ValueError("meta/info.json and meta/episodes.jsonl disagree")
    if report.get("instruction") != TASK:
        raise ValueError(f"Unexpected instruction: {report.get('instruction')!r}")

    converted_by_index = {
        int(item["output_episode_index"]): item
        for item in report["converted_episodes"]
    }
    expected_indices = set(range(len(episode_rows)))
    if set(converted_by_index) != expected_indices:
        raise ValueError("conversion_report.json does not cover every LeRobot episode")

    source_by_stage: dict[int, list[int]] = {}
    for source_index in sorted(converted_by_index):
        stage = int(converted_by_index[source_index]["stage"])
        source_by_stage.setdefault(stage, []).append(source_index)
    source_counts = {stage: len(indices) for stage, indices in source_by_stage.items()}
    print(f"[source] stage counts: {source_counts}")

    expected_stages = [1, 2, 3, 4]
    if sorted(source_by_stage) != expected_stages:
        raise ValueError(f"Expected stages {expected_stages}, got {sorted(source_by_stage)}")

    plan: list[dict] = []
    for stage in expected_stages:
        indices = list(source_by_stage[stage])
        if len(indices) > args.target_per_stage:
            raise ValueError(
                f"Stage {stage} has {len(indices)} episodes, exceeding target "
                f"{args.target_per_stage}"
            )
        if len(indices) < args.target_per_stage:
            if stage != args.duplicate_stage:
                raise ValueError(
                    f"Only stage {args.duplicate_stage} may be duplicated; "
                    f"stage {stage} has {len(indices)} episodes"
                )
            missing = args.target_per_stage - len(indices)
            if missing != 1:
                raise ValueError(
                    f"Expected exactly one missing stage-{stage} episode, got {missing}"
                )
            indices.append(indices[0])

        seen = Counter()
        for source_index in indices:
            seen[source_index] += 1
            plan.append(
                {
                    "stage": stage,
                    "source_episode_index": source_index,
                    "duplicated": seen[source_index] > 1,
                }
            )

    planned_counts = Counter(item["stage"] for item in plan)
    if planned_counts != Counter({stage: args.target_per_stage for stage in expected_stages}):
        raise AssertionError(planned_counts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output_dir.glob("episode_*.npy"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already contains {len(existing)} NPY files: {args.output_dir}. "
            "Use --overwrite only when intentionally regenerating it."
        )

    mapping = []
    lengths_by_stage: dict[int, list[int]] = {stage: [] for stage in expected_stages}
    for output_index, item in enumerate(plan):
        stage = int(item["stage"])
        source_index = int(item["source_episode_index"])
        output_file = f"episode_{output_index:06d}.npy"
        length, parquet_path = convert_episode(
            args.dataset_root,
            source_index,
            args.output_dir / output_file,
            output_index,
            stage,
            args.view_size,
        )
        source_length = int(episode_rows[source_index]["length"])
        if length != source_length:
            raise ValueError(
                f"Length mismatch for source episode {source_index}: "
                f"{length} != {source_length}"
            )
        lengths_by_stage[stage].append(length)
        report_item = converted_by_index[source_index]
        mapping.append(
            {
                "episode_index": output_index,
                "output_file": output_file,
                "episode_length": length,
                "task": TASK,
                "task_index": stage - 1,
                "group_id": stage,
                "reward_group_id": stage,
                "group_key": f"stage_{stage}",
                "source_stage_task": f"Pack the phone, stage {stage}.",
                "source_episode_index": source_index,
                "source_episode": report_item.get("source_episode"),
                "source_path": report_item.get("source_path", str(parquet_path)),
                "duplicated": bool(item["duplicated"]),
                "duplicate_of_source_episode_index": (
                    source_index if item["duplicated"] else None
                ),
            }
        )
        print(
            f"[convert] {output_index + 1:03d}/{len(plan)} "
            f"stage={stage} source={source_index:06d} -> {output_file} "
            f"frames={length}{' DUPLICATE' if item['duplicated'] else ''}"
        )

    stages = []
    for stage in expected_stages:
        lengths = lengths_by_stage[stage]
        stages.append(
            {
                "group_id": stage,
                "group_key": f"stage_{stage}",
                "task": TASK,
                "source_stage_task": f"Pack the phone, stage {stage}.",
                "episode_count": len(lengths),
                "frame_count": int(sum(lengths)),
                "length_min": int(np.min(lengths)),
                "length_median": float(np.median(lengths)),
                "length_p90": float(np.percentile(lengths, 90)),
                "length_max": int(np.max(lengths)),
                "recommended_horizon_p90_plus_16_aligned_16": (
                    aligned_p90_plus_chunk(lengths)
                ),
            }
        )

    manifest = {
        "format": "RLinf NpyTrajectoryDatasetWrapper initial_image",
        "source_dataset_root": str(args.dataset_root),
        "source_conversion_report": str(report_path),
        "source_codebase_version": "LeRobot v2.1",
        "source_fps": int(info["fps"]),
        "instructions": [TASK],
        "vla_instruction": TASK,
        "source_stage_tasks": [f"Pack the phone, stage {stage}." for stage in expected_stages],
        "image_layout": {
            "camera_order": list(CAMERAS),
            "per_view_shape": [args.view_size, args.view_size, 3],
            "stacked_shape": [args.view_size * len(CAMERAS), args.view_size, 3],
            "stack_axis": "height",
            "image_saved_on_frame": 0,
            "color_order": "RGB",
        },
        "action_semantics": (
            "action is copied from observation.state (14D absolute joints) "
            "for the packphone WM/FK condition"
        ),
        "state_semantics": "14D absolute joints from observation.state",
        "eef_action_semantics": (
            "14D left/right EEF xyz+quaternion_wxyz from source action"
        ),
        "group_semantics": (
            "reward_group_id is the source stage; every stage exposes the same "
            "Pack the phone. instruction to the VLA"
        ),
        "balancing": {
            "target_episodes_per_stage": args.target_per_stage,
            "source_counts": {str(k): v for k, v in source_counts.items()},
            "output_counts": {str(k): v for k, v in sorted(planned_counts.items())},
            "duplicated_stage": args.duplicate_stage,
            "duplicated_source_episode_indices": [
                item["source_episode_index"] for item in mapping if item["duplicated"]
            ],
        },
        "episodes": mapping,
        "episode_mapping": mapping,
        "total_episodes": len(mapping),
        "total_frames": int(sum(sum(v) for v in lengths_by_stage.values())),
        "stages": stages,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] output={args.output_dir}")
    print(f"[done] manifest={manifest_path}")
    print(f"[done] stage counts={dict(sorted(planned_counts.items()))}")


if __name__ == "__main__":
    main()
