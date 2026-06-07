#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REAL_CAMERAS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)


def ffprobe_video(path: Path) -> tuple[int, int, float, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    stream = json.loads(out)["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    rate = stream.get("avg_frame_rate", "0/1")
    num, den = rate.split("/")
    fps = float(num) / max(float(den), 1.0)
    nb_frames = int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else 0
    return width, height, fps, nb_frames


def read_video_rgb_ffmpeg(path: Path) -> list[np.ndarray]:
    width, height, _, _ = ffprobe_video(path)
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
        raise RuntimeError(f"Bad decode size for {path}: bytes={len(raw)} frame_size={frame_size}")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((-1, height, width, 3))
    return [frame.copy() for frame in arr]


def resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


def split_generated_three_views(
    frame: np.ndarray,
    view_width: int,
    view_height: int,
    crop_top: int,
) -> list[np.ndarray]:
    # The batch rollout video has a top reward/caption band. Remove it before
    # estimating depth, otherwise VGGT sees text as scene geometry.
    frame = frame[crop_top:]
    if frame.shape[0] < view_height * 3:
        raise RuntimeError(
            f"Generated frame too short after crop: {frame.shape}, "
            f"need at least {view_height * 3} high"
        )
    frame = frame[: view_height * 3]
    frame = resize_rgb(frame, view_width, view_height * 3)
    return [
        frame[0:view_height],
        frame[view_height : 2 * view_height],
        frame[2 * view_height : 3 * view_height],
    ]


def sample_indices(length: int, num_samples: int) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if num_samples <= 1:
        return [0]
    return np.linspace(0, length - 1, num_samples).round().astype(int).tolist()


def write_temp_images(frames: list[np.ndarray], temp_dir: Path, prefix: str) -> list[str]:
    paths = []
    for i, frame in enumerate(frames):
        path = temp_dir / f"{prefix}_{i:02d}.png"
        Image.fromarray(frame).save(path)
        paths.append(str(path))
    return paths


def depth_to_color(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(vmax - vmin, 1e-6)
    x = np.clip((depth - vmin) / denom, 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        colored = cm.get_cmap("turbo")(x)[..., :3]
    except Exception:
        colored = np.repeat(x[..., None], 3, axis=-1)
    return (colored * 255.0).astype(np.uint8)


def run_vggt_depth(model, load_and_preprocess_images, image_paths: list[str], device: str, dtype) -> np.ndarray:
    images = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    depth = predictions["depth"].detach().float().cpu().numpy()
    # Expected [S, H, W, 1] or [1, S, H, W, 1], depending on repo version.
    depth = np.squeeze(depth)
    if depth.ndim == 2:
        depth = depth[None]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise RuntimeError(f"Unexpected depth shape: {depth.shape}")
    return depth


def build_episode_sheet(
    episode_idx: int,
    real_root: Path,
    gen_root: Path,
    out_dir: Path,
    model,
    load_and_preprocess_images,
    device: str,
    dtype,
    samples: int,
    view_width: int,
    view_height: int,
    gen_crop_top: int,
) -> dict:
    ep_mp4 = f"episode_{episode_idx:06d}.mp4"
    real_paths = [real_root / "videos" / "chunk-000" / cam / ep_mp4 for cam in REAL_CAMERAS]
    gen_path = gen_root / f"episode_{episode_idx:06d}" / "wm_rollout_reward_annotated.mp4"
    for path in real_paths + [gen_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    real_frames_by_cam = [read_video_rgb_ffmpeg(path) for path in real_paths]
    gen_frames = read_video_rgb_ffmpeg(gen_path)
    real_len = min(len(frames) for frames in real_frames_by_cam)
    gen_len = len(gen_frames)
    common_len = min(real_len, gen_len)
    indices = sample_indices(common_len, samples)

    real_depth_groups: list[np.ndarray] = []
    gen_depth_groups: list[np.ndarray] = []
    all_depth_values = []

    temp_dir = Path(tempfile.mkdtemp(prefix=f"vggt_ep{episode_idx:06d}_"))
    try:
        for sample_i, idx in enumerate(indices):
            real_views = [
                resize_rgb(frames[idx], view_width, view_height)
                for frames in real_frames_by_cam
            ]
            gen_views = split_generated_three_views(
                gen_frames[idx],
                view_width=view_width,
                view_height=view_height,
                crop_top=gen_crop_top,
            )

            real_image_paths = write_temp_images(real_views, temp_dir, f"real_{sample_i:02d}")
            gen_image_paths = write_temp_images(gen_views, temp_dir, f"gen_{sample_i:02d}")

            real_depth = run_vggt_depth(
                model, load_and_preprocess_images, real_image_paths, device, dtype
            )
            gen_depth = run_vggt_depth(
                model, load_and_preprocess_images, gen_image_paths, device, dtype
            )

            real_depth_groups.append(real_depth)
            gen_depth_groups.append(gen_depth)
            all_depth_values.append(real_depth.reshape(-1))
            all_depth_values.append(gen_depth.reshape(-1))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    all_depth = np.concatenate(all_depth_values)
    finite = all_depth[np.isfinite(all_depth)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(finite, [2.0, 98.0]).tolist()
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))

    time_tiles = []
    for real_depth, gen_depth in zip(real_depth_groups, gen_depth_groups):
        real_col = np.concatenate(
            [resize_rgb(depth_to_color(d, vmin, vmax), view_width, view_height) for d in real_depth],
            axis=0,
        )
        gen_col = np.concatenate(
            [resize_rgb(depth_to_color(d, vmin, vmax), view_width, view_height) for d in gen_depth],
            axis=0,
        )
        time_tiles.append(np.concatenate([real_col, gen_col], axis=1))

    sheet = np.concatenate(time_tiles, axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"episode_{episode_idx:06d}_vggt_depth_real_vs_wm.png"
    Image.fromarray(sheet).save(out_path)
    return {
        "episode": episode_idx,
        "real_frames": real_len,
        "gen_frames": gen_len,
        "sample_indices": indices,
        "depth_vmin": vmin,
        "depth_vmax": vmax,
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
        default=Path("/data/zsq/RLinf/examples/embodiment/vggt_depth_real_vs_wm_0601_step3_2"),
    )
    parser.add_argument("--episodes", type=int, default=46)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--view-width", type=int, default=224)
    parser.add_argument("--view-height", type=int, default=224)
    parser.add_argument(
        "--gen-crop-top",
        type=int,
        default=44,
        help="Pixels to crop from top of wm_rollout_reward_annotated.mp4 before splitting views.",
    )
    parser.add_argument("--ckpt", type=Path, default=Path("/data/zsq/vggt/ckpt/model.pt"))
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import after parse so --help does not require PYTHONPATH.
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = VGGT()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    results = []
    for episode_idx in range(args.episodes):
        result = build_episode_sheet(
            episode_idx=episode_idx,
            real_root=args.real_root,
            gen_root=args.gen_root,
            out_dir=args.out_dir,
            model=model,
            load_and_preprocess_images=load_and_preprocess_images,
            device=device,
            dtype=dtype,
            samples=args.samples,
            view_width=args.view_width,
            view_height=args.view_height,
            gen_crop_top=args.gen_crop_top,
        )
        results.append(result)
        print(
            f"[{episode_idx + 1:02d}/{args.episodes}] ep={episode_idx:06d} "
            f"samples={result['sample_indices']} -> {result['out_path']}",
            flush=True,
        )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] {args.out_dir}")


if __name__ == "__main__":
    main()
