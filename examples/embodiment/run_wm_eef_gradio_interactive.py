#!/usr/bin/env python3
"""Interactive end-effector control for Wan multi-view world model.

This script keeps an absolute EE pose state for left/right arms, applies
button-driven EE deltas, repeats the resulting action for one WM chunk, and feeds
the direct EE trajectory into the existing WM pipeline. Initial EE pose is read
from the npy when available; otherwise the first-frame action is converted once
through the env's FK helper.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DIFFSYNTH_PATH = Path(
    os.environ.get("DIFFSYNTH_PATH", "/data/zsq/diffsynth-studio-wujie-ppu")
)
if DEFAULT_DIFFSYNTH_PATH.exists() and str(DEFAULT_DIFFSYNTH_PATH) not in sys.path:
    sys.path.insert(0, str(DEFAULT_DIFFSYNTH_PATH))


from rlinf.envs.world_model.world_model_wan_multi_view_env import WanmultiviewEnv


def _import_control_helpers():
    from diffsynth.trainers.arxx5_dataset4_wancontrolmultiview import (
        generate_raymap,
        generate_traj_map,
        make_center_ray_w2c_sequences,
        make_per_camera_sequences,
        scale_intrinsic,
        to_uint8_img,
    )

    from rlinf.envs.world_model.robotwin_camera_fk import (
        make_abs_actions_from_fk_wxyz,
        scale_intrinsic as robotwin_scale_intrinsic,
    )

    return {
        "generate_raymap": generate_raymap,
        "generate_traj_map": generate_traj_map,
        "make_center_ray_w2c_sequences": make_center_ray_w2c_sequences,
        "make_per_camera_sequences": make_per_camera_sequences,
        "scale_intrinsic": scale_intrinsic,
        "to_uint8_img": to_uint8_img,
        "make_abs_actions_from_fk_wxyz": make_abs_actions_from_fk_wxyz,
        "robotwin_scale_intrinsic": robotwin_scale_intrinsic,
    }


HELPERS = _import_control_helpers()


def parse_pose(text: Optional[str], name: str) -> Optional[np.ndarray]:
    if text is None or str(text).strip() == "":
        return None
    values = [float(x) for x in str(text).replace(" ", "").split(",") if x != ""]
    if len(values) != 7:
        raise ValueError(f"{name} must contain 7 comma-separated values: x,y,z,w,qx,qy,qz")
    pose = np.asarray(values, dtype=np.float32)
    pose[3:7] = normalize_quat_wxyz(pose[3:7])
    return pose


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def axis_angle_quat_wxyz(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float32)
    axis_arr = axis_arr / max(float(np.linalg.norm(axis_arr)), 1e-8)
    half = 0.5 * float(angle_rad)
    return np.asarray(
        [np.cos(half), *(np.sin(half) * axis_arr)], dtype=np.float32
    )


def tensor_frame_to_uint8(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim == 4:
        frame = frame[:, 0]
    frame = frame.detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((frame.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255)
    return arr.astype(np.uint8)


def pil_to_norm_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device)


def save_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> Optional[str]:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR))
            writer.append_data(frame)
    return str(path)


def load_cfg(args: argparse.Namespace) -> DictConfig:
    os.environ.setdefault("EMBODIED_PATH", str(Path(args.config_dir).resolve().parent))
    with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base=None):
        cfg = compose(config_name=args.config_name)
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.env.train.total_num_envs = 1
        cfg.env.train.group_size = 1
        cfg.env.train.use_fixed_reset_state_ids = True
        cfg.env.train.auto_reset = False
        cfg.env.train.seed = int(args.seed)
        cfg.env.train.reset_state_ids = [int(args.episode_index)]
        if "video_cfg" in cfg.env.train:
            cfg.env.train.video_cfg.save_video = False
    return cfg


class EEFInteractiveWM:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = load_cfg(args)
        self.env = WanmultiviewEnv(
            self.cfg.env.train,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
        )
        self.output_dir = (
            Path(args.output_dir)
            / f"{args.config_name}_episode_{int(args.episode_index):06d}_{int(time.time())}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_idx = 0
        self.left_pose = parse_pose(args.initial_left_pose, "initial-left-pose")
        self.right_pose = parse_pose(args.initial_right_pose, "initial-right-pose")
        self.left_gripper = float(args.initial_left_gripper)
        self.right_gripper = float(args.initial_right_gripper)
        self.condition_left_poses: List[np.ndarray] = []
        self.condition_right_poses: List[np.ndarray] = []
        self.rollout_wm_video: Optional[str] = None
        self.rollout_control_video: Optional[str] = None
        self.rollout_wm_frames: List[np.ndarray] = []
        self.rollout_control_frames: List[np.ndarray] = []
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_control_frame: Optional[np.ndarray] = None
        self.reset()

    @property
    def device(self) -> torch.device:
        return self.env.device

    def reset(self):
        self.step_idx = 0
        if hasattr(self.env, "reset_state_ids"):
            self.env.reset_state_ids = torch.tensor(
                [int(self.args.episode_index)], device=self.env.device, dtype=torch.long
            )
        self.env.reset()
        if self.left_pose is None or self.right_pose is None:
            left_pose, right_pose = self._read_initial_ee_pose()
            if self.left_pose is None:
                self.left_pose = left_pose
            if self.right_pose is None:
                self.right_pose = right_pose
        self.left_pose = self.left_pose.astype(np.float32)
        self.right_pose = self.right_pose.astype(np.float32)
        self.condition_left_poses = [
            self.left_pose.copy() for _ in range(self.env.condition_frame_length)
        ]
        self.condition_right_poses = [
            self.right_pose.copy() for _ in range(self.env.condition_frame_length)
        ]
        self.latest_frame = self._current_latest_frame()
        self.latest_control_frame = self._build_initial_control_frame()
        self.rollout_wm_frames = [self.latest_frame]
        self.rollout_control_frames = [self.latest_control_frame]
        self.rollout_wm_video = save_video(
            self.output_dir / "wm_rollout_full_reset.mp4",
            self.rollout_wm_frames,
            int(self.args.fps),
        )
        self.rollout_control_video = save_video(
            self.output_dir / "control_map_full_reset.mp4",
            self.rollout_control_frames,
            int(self.args.fps),
        )
        return self._outputs()

    def _read_initial_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        init_ee = getattr(self.env, "init_ee_poses", None)
        if init_ee is not None:
            init_ee0 = init_ee[0]
            if init_ee0 is None:
                init_ee0 = None
            else:
                if isinstance(init_ee0, torch.Tensor):
                    init_ee0 = init_ee0.detach().cpu().numpy()
                init_ee0 = np.asarray(init_ee0, dtype=np.float32)
                if init_ee0.shape[-1] >= 14:
                    return init_ee0[:7].copy(), init_ee0[7:14].copy()

        first_action = self._read_first_frame_action14()
        if first_action is not None and hasattr(self.env, "_action14_to_ee_pose_and_state"):
            ee_pose, _ = self.env._action14_to_ee_pose_and_state(first_action[None])
            ee_pose = np.asarray(ee_pose[0], dtype=np.float32)
            if ee_pose.shape[-1] >= 14:
                return ee_pose[:7].copy(), ee_pose[7:14].copy()

        raise RuntimeError(
            "No initial EE pose or first-frame action14 was found. Add observation.ee_pose "
            "or action to the initial npy, or pass --initial-left-pose and "
            "--initial-right-pose as x,y,z,w,qx,qy,qz."
        )

    def _read_first_frame_action14(self) -> Optional[np.ndarray]:
        for name in ("reference_action14", "condition_action", "current_action14"):
            value = getattr(self.env, name, None)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 3 and value.shape[0] > 0 and value.shape[-1] == 14:
                return value[0, 0].copy()
            if value.ndim == 2 and value.shape[0] > 0 and value.shape[-1] == 14:
                return value[0].copy()
            if value.ndim == 1 and value.shape[0] == 14:
                return value.copy()
        return None

    def _current_latest_frame(self) -> np.ndarray:
        current_obs = getattr(self.env, "current_obs", None)
        if isinstance(current_obs, torch.Tensor) and current_obs.numel() > 0:
            frame = current_obs[0, :, 0, -1]
            return tensor_frame_to_uint8(frame)
        queue = getattr(self.env, "image_queue", None)
        if queue and len(queue[0]) > 0:
            return tensor_frame_to_uint8(queue[0][-1])
        return np.zeros((self.env.view_height * len(self.env.camera_names), self.env.view_width, 3), dtype=np.uint8)

    def _build_initial_control_frame(self) -> np.ndarray:
        try:
            left_seq, right_seq, dummy_action14 = self._build_eef_sequence()
            control_video, _, _ = self._build_controls(left_seq, right_seq, dummy_action14)
            if control_video:
                return np.asarray(control_video[0].convert("RGB"))
        except Exception:
            pass
        return np.zeros_like(self.latest_frame)

    def _pose_status(self, tag: str) -> str:
        left = ", ".join(f"{x:.4f}" for x in self.left_pose)
        right = ", ".join(f"{x:.4f}" for x in self.right_pose)
        return (
            f"{tag}\n"
            f"step={self.step_idx}, chunk={self.env.chunk}, context={self.env.condition_frame_length}\n"
            f"left_eef[x,y,z,w,qx,qy,qz]=[{left}], gripper={self.left_gripper:.3f}\n"
            f"right_eef[x,y,z,w,qx,qy,qz]=[{right}], gripper={self.right_gripper:.3f}"
        )

    def _outputs(self):
        return self.rollout_wm_video, self.rollout_control_video

    def _normalize_arm(self, arm: str) -> str:
        mapping = {
            "left": "left",
            "right": "right",
            "both": "both",
            "左臂": "left",
            "右臂": "right",
            "双臂": "both",
        }
        arm_value = mapping.get(str(arm), str(arm))
        if arm_value not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported arm value from UI: {arm!r}")
        return arm_value

    def apply_delta(
        self,
        command: str,
        arm: str,
        translation_step: float,
        rotation_step_deg: float,
        gripper_step: float,
    ) -> None:
        arm = self._normalize_arm(arm)
        targets = []
        if arm in ("left", "both"):
            targets.append("left")
        if arm in ("right", "both"):
            targets.append("right")

        trans = {
            "x+": np.asarray([translation_step, 0.0, 0.0], dtype=np.float32),
            "x-": np.asarray([-translation_step, 0.0, 0.0], dtype=np.float32),
            "y+": np.asarray([0.0, translation_step, 0.0], dtype=np.float32),
            "y-": np.asarray([0.0, -translation_step, 0.0], dtype=np.float32),
            "z+": np.asarray([0.0, 0.0, translation_step], dtype=np.float32),
            "z-": np.asarray([0.0, 0.0, -translation_step], dtype=np.float32),
        }
        rot_axis = {
            "roll+": ([1.0, 0.0, 0.0], np.deg2rad(rotation_step_deg)),
            "roll-": ([1.0, 0.0, 0.0], -np.deg2rad(rotation_step_deg)),
            "pitch+": ([0.0, 1.0, 0.0], np.deg2rad(rotation_step_deg)),
            "pitch-": ([0.0, 1.0, 0.0], -np.deg2rad(rotation_step_deg)),
            "yaw+": ([0.0, 0.0, 1.0], np.deg2rad(rotation_step_deg)),
            "yaw-": ([0.0, 0.0, 1.0], -np.deg2rad(rotation_step_deg)),
        }
        for target in targets:
            pose = self.left_pose if target == "left" else self.right_pose
            if command in trans:
                pose[:3] += trans[command]
            elif command in rot_axis:
                axis, angle = rot_axis[command]
                delta_q = axis_angle_quat_wxyz(axis, angle)
                pose[3:7] = normalize_quat_wxyz(quat_mul_wxyz(delta_q, pose[3:7]))
            elif command == "gripper_open":
                if target == "left":
                    self.left_gripper = min(1.0, self.left_gripper + gripper_step)
                else:
                    self.right_gripper = min(1.0, self.right_gripper + gripper_step)
            elif command == "gripper_close":
                if target == "left":
                    self.left_gripper = max(0.0, self.left_gripper - gripper_step)
                else:
                    self.right_gripper = max(0.0, self.right_gripper - gripper_step)

    def apply_camera_delta(
        self,
        command: str,
        arm: str,
        camera_name: str,
        pixel_step: float,
        depth_step: float,
    ) -> None:
        arm = self._normalize_arm(arm)
        pixel_delta = {
            "cam_left": (-pixel_step, 0.0, 0.0),
            "cam_right": (pixel_step, 0.0, 0.0),
            "cam_up": (0.0, -pixel_step, 0.0),
            "cam_down": (0.0, pixel_step, 0.0),
            "cam_near": (0.0, 0.0, -depth_step),
            "cam_far": (0.0, 0.0, depth_step),
        }
        if command not in pixel_delta:
            raise ValueError(f"Unsupported camera-plane command: {command}")

        targets = []
        if arm in ("left", "both"):
            targets.append("left")
        if arm in ("right", "both"):
            targets.append("right")

        du, dv, dz = pixel_delta[command]
        w2c, intrinsic = self._current_camera_w2c_and_intrinsic(camera_name)
        c2w = np.linalg.inv(w2c).astype(np.float32)

        for target in targets:
            pose = self.left_pose if target == "left" else self.right_pose
            point_world = np.ones(4, dtype=np.float32)
            point_world[:3] = pose[:3]
            point_cam = (w2c @ point_world).astype(np.float32)
            depth = max(abs(float(point_cam[2])), 1e-4)
            dx_cam = float(du) * depth / max(float(intrinsic[0, 0]), 1e-6)
            dy_cam = float(dv) * depth / max(float(intrinsic[1, 1]), 1e-6)
            delta_cam = np.asarray([dx_cam, dy_cam, float(dz)], dtype=np.float32)
            delta_world = c2w[:3, :3] @ delta_cam
            pose[:3] += delta_world.astype(np.float32)

    def _current_camera_w2c_and_intrinsic(self, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        camera_name = str(camera_name)
        if camera_name not in self.env.camera_names:
            raise ValueError(f"Unknown camera {camera_name}; available={self.env.camera_names}")

        if getattr(self.env, "is_robotwin_env", False):
            anchor = self.env.robotwin_camera_anchors[0]
            if anchor is None:
                raise RuntimeError("RobotWin camera anchor is missing after env.reset().")
            camera_idx = self.env.camera_names.index(camera_name)
            robotwin_cam = self.env.robotwin_camera_names[camera_idx]
            w2c_by_robotwin_cam = self.env.robotwin_projector.build_w2c_sequences(
                anchor,
                self.left_pose[None].astype(np.float32),
                self.right_pose[None].astype(np.float32),
            )
            original_h, original_w = anchor.original_hw_by_cam[robotwin_cam]
            intrinsic = HELPERS["robotwin_scale_intrinsic"](
                anchor.intrinsic_by_cam[robotwin_cam],
                original_h,
                original_w,
                self.env.view_height,
                self.env.view_width,
            )
            return (
                w2c_by_robotwin_cam[robotwin_cam][0].astype(np.float32),
                intrinsic.astype(np.float32),
            )

        ee_pose = np.concatenate([self.left_pose, self.right_pose], axis=-1)[None].astype(np.float32)
        dummy_action14 = np.zeros((1, 14), dtype=np.float32)
        dummy_action14[:, 6] = self.left_gripper
        dummy_action14[:, 13] = self.right_gripper
        _, w2c_by_cam = HELPERS["make_per_camera_sequences"](
            ee_pose,
            dummy_action14,
            self.env.extrinsics,
            getattr(self.env, "camera_axis_mode", "opencv"),
        )
        intrinsic = HELPERS["scale_intrinsic"](
            self.env.intrinsics[camera_name],
            self.env.camera_original_height,
            self.env.camera_original_width,
            self.env.view_height,
            self.env.view_width,
        )
        return w2c_by_cam[camera_name][0].astype(np.float32), intrinsic.astype(np.float32)

    def _generate_after_delta(self, tag: str, fps: int):
        left_seq, right_seq, dummy_action14 = self._build_eef_sequence()
        control_video, ray_map_o, ray_map_d = self._build_controls(
            left_seq, right_seq, dummy_action14
        )
        input_video = self._build_input_video()
        output = self._run_world_model(input_video, control_video, ray_map_o, ray_map_d)
        wm_frames, control_frames = self._commit_generation(output, control_video)
        if wm_frames:
            self.latest_frame = wm_frames[-1]
            self.rollout_wm_frames.extend(wm_frames)
        if control_frames:
            self.latest_control_frame = control_frames[-1]
            self.rollout_control_frames.extend(control_frames)
        self.rollout_wm_video = save_video(
            self.output_dir / f"wm_rollout_full_{self.step_idx:04d}.mp4",
            self.rollout_wm_frames,
            int(fps),
        )
        self.rollout_control_video = save_video(
            self.output_dir / f"control_map_full_{self.step_idx:04d}.mp4",
            self.rollout_control_frames,
            int(fps),
        )
        print(
            "[eef_gradio] "
            f"step={self.step_idx} tag={tag} "
            f"left_xyz={self.left_pose[:3].tolist()} "
            f"right_xyz={self.right_pose[:3].tolist()} "
            f"wm_video={self.rollout_wm_video} "
            f"control_video={self.rollout_control_video}",
            flush=True,
        )
        self.step_idx += 1
        return self._outputs()

    def step(
        self,
        command: str,
        arm: str,
        translation_step: float,
        rotation_step_deg: float,
        gripper_step: float,
        fps: int,
    ):
        self.apply_delta(command, arm, translation_step, rotation_step_deg, gripper_step)
        return self._generate_after_delta(command, fps)

    def step_camera(
        self,
        command: str,
        arm: str,
        camera_name: str,
        pixel_step: float,
        depth_step: float,
        fps: int,
    ):
        self.apply_camera_delta(command, arm, camera_name, pixel_step, depth_step)
        return self._generate_after_delta(f"{camera_name}:{command}", fps)

    def _build_eef_sequence(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_context = [x.copy() for x in self.condition_left_poses[-self.env.condition_frame_length :]]
        right_context = [x.copy() for x in self.condition_right_poses[-self.env.condition_frame_length :]]
        while len(left_context) < self.env.condition_frame_length:
            left_context.insert(0, left_context[0].copy())
            right_context.insert(0, right_context[0].copy())
        left_future = [self.left_pose.copy() for _ in range(self.env.chunk)]
        right_future = [self.right_pose.copy() for _ in range(self.env.chunk)]
        left_seq = np.stack(left_context + left_future, axis=0).astype(np.float32)
        right_seq = np.stack(right_context + right_future, axis=0).astype(np.float32)
        dummy_action14 = np.zeros((left_seq.shape[0], 14), dtype=np.float32)
        dummy_action14[:, 6] = self.left_gripper
        dummy_action14[:, 13] = self.right_gripper
        return left_seq, right_seq, dummy_action14

    def _build_input_video(self) -> List[Image.Image]:
        target_size = (int(self.env.view_width), int(self.env.view_height) * len(self.env.camera_names))
        queue = self.env.image_queue[0]
        frames = list(queue[-self.env.condition_frame_length :])
        while len(frames) < self.env.condition_frame_length:
            frames.insert(0, frames[0])
        input_video = []
        for frame in frames:
            image = self.env._tensor_frame_to_pil(frame)
            if image.size != target_size:
                image = image.resize(target_size, Image.BILINEAR)
            input_video.append(image)
        return input_video

    def _build_controls(
        self,
        left_seq: np.ndarray,
        right_seq: np.ndarray,
        dummy_action14: np.ndarray,
    ) -> Tuple[List[Image.Image], List[np.ndarray], List[np.ndarray]]:
        if getattr(self.env, "is_robotwin_env", False):
            return self._build_robotwin_controls_direct(left_seq, right_seq, dummy_action14)
        return self._build_arxx5_controls_direct(left_seq, right_seq, dummy_action14)

    def _build_robotwin_controls_direct(
        self,
        left_seq: np.ndarray,
        right_seq: np.ndarray,
        dummy_action14: np.ndarray,
    ) -> Tuple[List[Image.Image], List[np.ndarray], List[np.ndarray]]:
        anchor = self.env.robotwin_camera_anchors[0]
        if anchor is None:
            raise RuntimeError(
                "RobotWin camera anchor is missing after env.reset(); cannot build direct EE camera controls."
            )
        abs_actions = HELPERS["make_abs_actions_from_fk_wxyz"](
            dummy_action14, left_seq, right_seq
        ).astype(np.float32)
        w2c_by_robotwin_cam = self.env.robotwin_projector.build_w2c_sequences(
            anchor, left_seq, right_seq
        )
        control_by_cam: Dict[str, np.ndarray] = {}
        ray_o_by_cam: Dict[str, np.ndarray] = {}
        ray_d_by_cam: Dict[str, np.ndarray] = {}
        for camera_name, robotwin_cam in zip(
            self.env.camera_names, self.env.robotwin_camera_names
        ):
            original_h, original_w = anchor.original_hw_by_cam[robotwin_cam]
            intrinsic = HELPERS["robotwin_scale_intrinsic"](
                anchor.intrinsic_by_cam[robotwin_cam],
                original_h,
                original_w,
                self.env.view_height,
                self.env.view_width,
            )
            w2c_seq = w2c_by_robotwin_cam[robotwin_cam]
            control_by_cam[camera_name] = HELPERS["generate_traj_map"](
                abs_actions,
                w2c_seq,
                intrinsic,
                self.env.view_height,
                self.env.view_width,
                radius=self.env.traj_radius,
                radius_mode=self.env.traj_radius_mode,
                min_radius=self.env.traj_min_radius,
                max_radius=self.env.traj_max_radius,
                ref_depth=self.env.traj_ref_depth,
                near_depth=self.env.traj_near_depth,
                far_depth=self.env.traj_far_depth,
            )
            ray_o_seq = []
            ray_d_seq = []
            for frame_idx in range(w2c_seq.shape[0]):
                c2w = np.linalg.inv(w2c_seq[frame_idx]).astype(np.float32)
                ray_o, ray_d = HELPERS["generate_raymap"](
                    intrinsic,
                    c2w,
                    self.env.view_height,
                    self.env.view_width,
                )
                ray_o_seq.append(ray_o)
                ray_d_seq.append(ray_d)
            ray_o_by_cam[camera_name] = ray_o_seq
            ray_d_by_cam[camera_name] = ray_d_seq
        return self._stack_camera_controls(control_by_cam, ray_o_by_cam, ray_d_by_cam)

    def _build_arxx5_controls_direct(
        self,
        left_seq: np.ndarray,
        right_seq: np.ndarray,
        dummy_action14: np.ndarray,
    ) -> Tuple[List[Image.Image], List[np.ndarray], List[np.ndarray]]:
        ee_pose = np.concatenate([left_seq, right_seq], axis=-1).astype(np.float32)
        actions_by_cam, w2c_by_cam = HELPERS["make_per_camera_sequences"](
            ee_pose,
            dummy_action14,
            self.env.extrinsics,
            getattr(self.env, "camera_axis_mode", "opencv"),
        )
        ray_w2c_by_cam = HELPERS["make_center_ray_w2c_sequences"](
            ee_pose,
            self.env.extrinsics,
            getattr(self.env, "camera_axis_mode", "opencv"),
        )
        control_by_cam: Dict[str, np.ndarray] = {}
        ray_o_by_cam: Dict[str, np.ndarray] = {}
        ray_d_by_cam: Dict[str, np.ndarray] = {}
        for camera_name in self.env.camera_names:
            intrinsic = HELPERS["scale_intrinsic"](
                self.env.intrinsics[camera_name],
                self.env.camera_original_height,
                self.env.camera_original_width,
                self.env.view_height,
                self.env.view_width,
            )
            w2c_seq = w2c_by_cam[camera_name]
            control_by_cam[camera_name] = HELPERS["generate_traj_map"](
                actions_by_cam[camera_name],
                w2c_seq,
                intrinsic,
                self.env.view_height,
                self.env.view_width,
                radius=self.env.traj_radius,
                radius_mode=self.env.traj_radius_mode,
                min_radius=self.env.traj_min_radius,
                max_radius=self.env.traj_max_radius,
                ref_depth=self.env.traj_ref_depth,
                near_depth=self.env.traj_near_depth,
                far_depth=self.env.traj_far_depth,
            )
            ray_o_seq = []
            ray_d_seq = []
            for frame_idx in range(ee_pose.shape[0]):
                c2w = np.linalg.inv(ray_w2c_by_cam[camera_name][frame_idx]).astype(np.float32)
                ray_o, ray_d = HELPERS["generate_raymap"](
                    intrinsic,
                    c2w,
                    self.env.view_height,
                    self.env.view_width,
                )
                ray_o_seq.append(ray_o)
                ray_d_seq.append(ray_d)
            ray_o_by_cam[camera_name] = ray_o_seq
            ray_d_by_cam[camera_name] = ray_d_seq
        return self._stack_camera_controls(control_by_cam, ray_o_by_cam, ray_d_by_cam)

    def _stack_camera_controls(
        self,
        control_by_cam: Dict[str, np.ndarray],
        ray_o_by_cam: Dict[str, np.ndarray],
        ray_d_by_cam: Dict[str, np.ndarray],
    ) -> Tuple[List[Image.Image], List[np.ndarray], List[np.ndarray]]:
        control_video: List[Image.Image] = []
        ray_map_o: List[np.ndarray] = []
        ray_map_d: List[np.ndarray] = []
        for frame_idx in range(self.env.num_frames):
            control_frame = np.concatenate(
                [
                    control_by_cam[camera_name][frame_idx]
                    for camera_name in self.env.camera_names
                ],
                axis=0,
            )
            ray_o_frame = np.concatenate(
                [
                    HELPERS["to_uint8_img"](
                        ray_o_by_cam[camera_name][frame_idx],
                        self.env.ray_o_vmin,
                        self.env.ray_o_vmax,
                    )
                    for camera_name in self.env.camera_names
                ],
                axis=0,
            )
            ray_d_frame = np.concatenate(
                [
                    HELPERS["to_uint8_img"](
                        ray_d_by_cam[camera_name][frame_idx],
                        self.env.ray_d_vmin,
                        self.env.ray_d_vmax,
                    )
                    for camera_name in self.env.camera_names
                ],
                axis=0,
            )
            control_video.append(Image.fromarray(control_frame))
            ray_map_o.append(Image.fromarray(ray_o_frame))
            ray_map_d.append(Image.fromarray(ray_d_frame))
        return control_video, ray_map_o, ray_map_d

    def _run_world_model(
        self,
        input_video: List[Image.Image],
        control_video: List[Image.Image],
        ray_map_o: List[np.ndarray],
        ray_map_d: List[np.ndarray],
    ):
        prompt = self.env.task_descriptions[0] if hasattr(self.env, "task_descriptions") else self.env.prompt
        kwargs = {
            "prompt": prompt,
            "input_video": input_video,
            "vace_video": control_video,
            "ray_map_o": ray_map_o,
            "ray_map_d": ray_map_d,
            "height": int(self.env.view_height) * len(self.env.camera_names),
            "width": int(self.env.view_width),
            "num_frames": int(self.env.num_frames),
            "num_inference_steps": int(self.env.num_inference_steps),
            "cfg_scale": float(self.env.cfg_scale),
            "seed": int(self.args.wm_seed),
            "tiled": bool(self.env.tiled),
            "progress_bar_cmd": lambda x: x,
        }
        return self.env.pipe(**kwargs)

    def _commit_generation(
        self,
        output,
        control_video: List[Image.Image],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        frames = output
        if hasattr(output, "frames"):
            frames = output.frames
        if isinstance(frames, tuple):
            frames = frames[0]
        if not isinstance(frames, list):
            raise RuntimeError(f"Unsupported WM output type: {type(output)}")

        start = int(self.env.condition_frame_length)
        end = start + int(self.env.chunk)
        pred_pils = frames[start:end]
        pred_controls = control_video[start:end]
        pred_tensors = torch.stack(
            [pil_to_norm_tensor(image, self.device) for image in pred_pils], dim=0
        )  # [T, 3, H, W]
        pred_tensors = pred_tensors.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(2)

        if self.env.current_obs is None:
            self.env.current_obs = pred_tensors
        else:
            self.env.current_obs = torch.cat([self.env.current_obs, pred_tensors], dim=3)
            max_frames = int(getattr(self.env, "max_episode_frames", 1000000))
            if self.env.current_obs.shape[3] > max_frames:
                self.env.current_obs = self.env.current_obs[:, :, :, -max_frames:]

        recent_count = max(0, int(self.env.condition_frame_length) - 1)
        reference_frame = self.env.image_queue[0][0]
        recent_frames = [
            pred_tensors[0, :, 0, i : i + 1].detach().clone()
            for i in range(max(0, pred_tensors.shape[3] - recent_count), pred_tensors.shape[3])
        ]
        new_queue = [reference_frame] + recent_frames
        while len(new_queue) < self.env.condition_frame_length:
            new_queue.insert(1, new_queue[-1])
        self.env.image_queue[0] = new_queue[-self.env.condition_frame_length :]

        self.condition_left_poses = (
            [self.condition_left_poses[0].copy()]
            + [self.left_pose.copy() for _ in range(recent_count)]
        )[-self.env.condition_frame_length :]
        self.condition_right_poses = (
            [self.condition_right_poses[0].copy()]
            + [self.right_pose.copy() for _ in range(recent_count)]
        )[-self.env.condition_frame_length :]

        wm_frames = [np.asarray(image.convert("RGB")) for image in pred_pils]
        control_frames = [np.asarray(image.convert("RGB")) for image in pred_controls]
        return wm_frames, control_frames


def build_ui(app: EEFInteractiveWM) -> gr.Blocks:
    with gr.Blocks(title="Wan WM EEF Interactive Control") as demo:
        gr.Markdown("## Wan 多视角世界模型 EE 实时交互")
        with gr.Row():
            arm = gr.Dropdown(
                choices=[("左臂", "left"), ("右臂", "right"), ("双臂", "both")],
                value="left",
                label="控制对象",
            )
            translation_step = gr.Slider(
                minimum=0.001,
                maximum=0.08,
                value=float(app.args.translation_step),
                step=0.001,
                label="平移步长",
            )
            rotation_step = gr.Slider(
                minimum=1.0,
                maximum=30.0,
                value=float(app.args.rotation_step_deg),
                step=1.0,
                label="旋转步长(度)",
            )
            gripper_step = gr.Slider(
                minimum=0.01,
                maximum=0.5,
                value=float(app.args.gripper_step),
                step=0.01,
                label="夹爪步长",
            )
            fps = gr.Slider(minimum=1, maximum=30, value=int(app.args.fps), step=1, label="保存 FPS")
            camera = gr.Dropdown(
                choices=[(name, name) for name in app.env.camera_names],
                value=app.env.camera_names[0],
                label="投影视角",
            )
            pixel_step = gr.Slider(
                minimum=1,
                maximum=80,
                value=int(app.args.pixel_step),
                step=1,
                label="画面步长(px)",
            )

        with gr.Row():
            wm_video = gr.Video(label="从初始到当前的 WM rollout", autoplay=True, loop=True)
            control_video = gr.Video(label="从初始到当前的 control_map", autoplay=True, loop=True)

        outputs = [wm_video, control_video]

        def run_command(command, arm_value, trans_value, rot_value, grip_value, fps_value):
            return app.step(
                command,
                arm_value,
                float(trans_value),
                float(rot_value),
                float(grip_value),
                int(fps_value),
            )

        def run_camera_command(command, arm_value, camera_value, pixel_value, depth_value, fps_value):
            return app.step_camera(
                command,
                arm_value,
                camera_value,
                float(pixel_value),
                float(depth_value),
                int(fps_value),
            )

        gr.Markdown("### 按相机画面移动")
        with gr.Row():
            gr.Button("画面左移").click(
                lambda a, c, p, d, f: run_camera_command("cam_left", a, c, p, d, f),
                inputs=[arm, camera, pixel_step, translation_step, fps],
                outputs=outputs,
            )
            gr.Button("画面右移").click(
                lambda a, c, p, d, f: run_camera_command("cam_right", a, c, p, d, f),
                inputs=[arm, camera, pixel_step, translation_step, fps],
                outputs=outputs,
            )
            gr.Button("画面上移").click(
                lambda a, c, p, d, f: run_camera_command("cam_up", a, c, p, d, f),
                inputs=[arm, camera, pixel_step, translation_step, fps],
                outputs=outputs,
            )
            gr.Button("画面下移").click(
                lambda a, c, p, d, f: run_camera_command("cam_down", a, c, p, d, f),
                inputs=[arm, camera, pixel_step, translation_step, fps],
                outputs=outputs,
            )
            gr.Button("靠近相机").click(
                lambda a, c, p, d, f: run_camera_command("cam_near", a, c, p, d, f),
                inputs=[arm, camera, pixel_step, translation_step, fps],
                outputs=outputs,
            )
            gr.Button("远离相机").click(
                lambda a, c, p, d, f: run_camera_command("cam_far", a, c, p, d, f),
                inputs=[arm, camera, pixel_step, translation_step, fps],
                outputs=outputs,
            )

        gr.Markdown("### 世界坐标移动/旋转")
        with gr.Row():
            gr.Button("X-").click(
                lambda a, t, r, g, f: run_command("x-", a, t, r, g, f),
                inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                outputs=outputs,
            )
            gr.Button("X+").click(
                lambda a, t, r, g, f: run_command("x+", a, t, r, g, f),
                inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                outputs=outputs,
            )
            gr.Button("Y-").click(
                lambda a, t, r, g, f: run_command("y-", a, t, r, g, f),
                inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                outputs=outputs,
            )
            gr.Button("Y+").click(
                lambda a, t, r, g, f: run_command("y+", a, t, r, g, f),
                inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                outputs=outputs,
            )
            gr.Button("Z-").click(
                lambda a, t, r, g, f: run_command("z-", a, t, r, g, f),
                inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                outputs=outputs,
            )
            gr.Button("Z+").click(
                lambda a, t, r, g, f: run_command("z+", a, t, r, g, f),
                inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                outputs=outputs,
            )

        with gr.Row():
            for label, command in [
                ("Roll-", "roll-"),
                ("Roll+", "roll+"),
                ("Pitch-", "pitch-"),
                ("Pitch+", "pitch+"),
                ("Yaw-", "yaw-"),
                ("Yaw+", "yaw+"),
            ]:
                gr.Button(label).click(
                    lambda a, t, r, g, f, c=command: run_command(c, a, t, r, g, f),
                    inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                    outputs=outputs,
                )

        with gr.Row():
            gr.Button("夹爪打开").click(
                lambda a, t, r, g, f: run_command("gripper_open", a, t, r, g, f),
                inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                outputs=outputs,
            )
            gr.Button("夹爪闭合").click(
                lambda a, t, r, g, f: run_command("gripper_close", a, t, r, g, f),
                inputs=[arm, translation_step, rotation_step, gripper_step, fps],
                outputs=outputs,
            )
            gr.Button("Reset").click(lambda: app.reset(), outputs=outputs)

        demo.load(lambda: app._outputs(), outputs=outputs)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/data/zsq/RLinf/examples/embodiment/config")
    parser.add_argument("--config-name", default="wanmultiview_robotwin_grpo_pi05")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="/data/zsq/RLinf/examples/embodiment/eef_gradio_rollouts",
    )
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--wm-seed", type=int, default=42)
    parser.add_argument("--translation-step", type=float, default=0.01)
    parser.add_argument("--pixel-step", type=int, default=16)
    parser.add_argument("--rotation-step-deg", type=float, default=5.0)
    parser.add_argument("--gripper-step", type=float, default=0.05)
    parser.add_argument("--initial-left-gripper", type=float, default=0.0)
    parser.add_argument("--initial-right-gripper", type=float, default=0.0)
    parser.add_argument(
        "--initial-left-pose",
        default=None,
        help="Optional x,y,z,w,qx,qy,qz initial left EE pose.",
    )
    parser.add_argument(
        "--initial-right-pose",
        default=None,
        help="Optional x,y,z,w,qx,qy,qz initial right EE pose.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = EEFInteractiveWM(args)
    demo = build_ui(app)
    demo.launch(
        server_name=args.server_name,
        server_port=int(args.server_port),
        share=bool(args.share),
    )


if __name__ == "__main__":
    main()
