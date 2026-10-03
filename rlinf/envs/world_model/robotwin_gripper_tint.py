"""RoboTwin gripper-tint preprocessing shared by world-model environments."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch

from rlinf.envs.world_model.robotwin_camera_fk import (
    RobotWinCameraFKProjector,
    scale_intrinsic,
    to_robotwin_camera_name,
)


class RobotWinGripperTintSupport:
    """Build training-compatible head-view tint inputs from absolute actions."""

    def __init__(
        self,
        cfg,
        *,
        num_envs: int,
        chunk: int,
        camera_names,
        view_height: int,
        view_width: int,
        device,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.chunk = int(chunk)
        self.camera_names = list(camera_names)
        self.view_height = int(view_height)
        self.view_width = int(view_width)
        self.device = torch.device(device)

        if "head" not in self.camera_names:
            raise ValueError(
                "Gripper-tint reward inference requires a head camera, got "
                f"{self.camera_names}"
            )
        self.robotwin_camera_names = [
            to_robotwin_camera_name(name) for name in self.camera_names
        ]

        self.fk_fn = self._resolve_fk_fn(cfg.get("fk_fn", None))
        self.left_fk_fn = (
            self._resolve_fk_fn(cfg.get("left_fk_fn", None)) or self.fk_fn
        )
        self.right_fk_fn = (
            self._resolve_fk_fn(cfg.get("right_fk_fn", None)) or self.fk_fn
        )
        self.fk_pose_quat_order = cfg.get("fk_pose_quat_order", "qwxyz")
        self.fk_urdf_path = cfg.get("fk_urdf_path", None)
        self.fk_offset_path = cfg.get("fk_offset_path", None)
        self.fk_base_link = cfg.get("fk_base_link", "base_link")
        self.fk_tip_link = cfg.get("fk_tip_link", "link6")
        self.fk_offset_mode = cfg.get("fk_offset_mode", None)
        self.fk_state_unit = cfg.get("fk_state_unit", "rad")
        self._configure_fk_modules()

        self.projector = RobotWinCameraFKProjector(self.robotwin_camera_names)
        self.source_paths = self._load_episode_source_paths(cfg.initial_image_path)
        if not self.source_paths:
            raise ValueError(
                "Gripper-tint reward inference requires "
                "initial_image_path/manifest.json with episode source paths"
            )
        self.camera_anchors = [None] * self.num_envs

    @staticmethod
    def _resolve_fk_fn(fn_or_path):
        if fn_or_path is None:
            return None
        if callable(fn_or_path):
            return fn_or_path
        if not isinstance(fn_or_path, str):
            raise TypeError(
                f"FK function must be callable or import path, got {type(fn_or_path)}"
            )
        module_name, sep, attr_name = fn_or_path.replace(":", ".").rpartition(".")
        if not sep:
            raise ValueError(f"Invalid FK import path: {fn_or_path}")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    def _configure_fk_modules(self):
        seen_modules = set()
        for fk_fn in (self.fk_fn, self.left_fk_fn, self.right_fk_fn):
            if fk_fn is None:
                continue
            module_name = getattr(fk_fn, "__module__", None)
            if module_name is None or module_name in seen_modules:
                continue
            seen_modules.add(module_name)
            module = importlib.import_module(module_name)
            configure = getattr(module, "configure", None)
            if callable(configure):
                configure(
                    urdf_path=self.fk_urdf_path,
                    offset_path=self.fk_offset_path,
                    base_link=self.fk_base_link,
                    tip_link=self.fk_tip_link,
                    offset_mode=self.fk_offset_mode,
                    state_unit=self.fk_state_unit,
                )

    def _call_fk(self, joints, arm):
        fk_fn = self.left_fk_fn if arm == "left" else self.right_fk_fn
        if fk_fn is None:
            raise ValueError(
                "Gripper-tint reward inference requires cfg.fk_fn or "
                "cfg.left_fk_fn/cfg.right_fk_fn"
            )
        pose = fk_fn(joints)
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        pose = np.asarray(pose, dtype=np.float32)
        if pose.ndim == 1:
            pose = pose[None, :]
        if pose.shape[-1] != 7:
            raise ValueError(f"FK for {arm} arm must return [..., 7], got {pose.shape}")
        if self.fk_pose_quat_order == "xyzw":
            pose = pose.copy()
            pose[:, 3:7] = pose[:, [6, 3, 4, 5]]
        elif self.fk_pose_quat_order != "qwxyz":
            raise ValueError(
                f"Unsupported fk_pose_quat_order: {self.fk_pose_quat_order}"
            )
        return pose.astype(np.float32)

    def _action14_to_ee_pose(self, action14_seq):
        action14_seq = np.asarray(action14_seq, dtype=np.float32)
        if action14_seq.ndim != 2 or action14_seq.shape[-1] != 14:
            raise ValueError(
                f"Expected action sequence [T, 14], got {action14_seq.shape}"
            )
        left_pose = self._call_fk(action14_seq[:, 0:6], "left")
        right_pose = self._call_fk(action14_seq[:, 7:13], "right")
        if (
            left_pose.shape[0] != action14_seq.shape[0]
            or right_pose.shape[0] != action14_seq.shape[0]
        ):
            raise ValueError(
                f"FK output length mismatch: actions={action14_seq.shape[0]}, "
                f"left={left_pose.shape[0]}, right={right_pose.shape[0]}"
            )
        return np.concatenate([left_pose, right_pose], axis=-1).astype(np.float32)

    @staticmethod
    def _load_episode_source_paths(initial_image_path):
        manifest_path = Path(initial_image_path) / "manifest.json"
        if not manifest_path.exists():
            return []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_paths = []
        for item in manifest.get("episode_mapping", []):
            source_path = item.get("source_path") or item.get("path")
            source_paths.append(str(source_path) if source_path else None)
        return source_paths

    def _source_path_for_episode(self, episode_idx):
        episode_idx = int(episode_idx)
        if episode_idx < len(self.source_paths) and self.source_paths[episode_idx]:
            source_path = Path(self.source_paths[episode_idx])
            if source_path.exists():
                return str(source_path)
            raise FileNotFoundError(
                "RoboTwin manifest source path does not exist for episode "
                f"{episode_idx}: {source_path}"
            )
        raise ValueError(
            f"Cannot resolve RoboTwin source path for episode {episode_idx}"
        )

    def reset(self, episode_indices, reference_actions14):
        reference_actions14 = torch.as_tensor(reference_actions14).detach().cpu().numpy()
        if reference_actions14.shape != (self.num_envs, 14):
            raise ValueError(
                "Expected reference actions "
                f"{(self.num_envs, 14)}, got {reference_actions14.shape}"
            )
        for env_idx, episode_idx in enumerate(episode_indices):
            ee_pose = self._action14_to_ee_pose(
                reference_actions14[env_idx].reshape(1, 14)
            )
            self.camera_anchors[env_idx] = self.projector.build_anchor(
                source_path=self._source_path_for_episode(episode_idx),
                left_pose0_wxyz=ee_pose[0, 0:7],
                right_pose0_wxyz=ee_pose[0, 7:14],
            )

    @staticmethod
    def _project_point(point_xyz, extrinsic, intrinsic, height, width):
        point = np.array(
            [point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64
        )
        best = None
        for mode, ext in (("as_is", extrinsic), ("inv", np.linalg.inv(extrinsic))):
            point_camera = ext @ point
            depth = float(point_camera[2])
            if abs(depth) <= 1e-6:
                continue
            pixel = intrinsic @ point_camera[:3]
            u = float(pixel[0] / pixel[2])
            v = float(pixel[1] / pixel[2])
            visible = depth > 1e-6 and 0 <= u < width and 0 <= v < height
            clamped_u = float(np.clip(u, 0, width - 1))
            clamped_v = float(np.clip(v, 0, height - 1))
            candidate = {
                "mode": mode,
                "visible": visible,
                "offscreen_dist": (u - clamped_u) ** 2 + (v - clamped_v) ** 2,
                "center_dist": (u - width / 2) ** 2 + (v - height / 2) ** 2,
            }
            if visible:
                return candidate
            if best is None or (
                candidate["offscreen_dist"], candidate["center_dist"]
            ) < (best["offscreen_dist"], best["center_dist"]):
                best = candidate
        if best is not None:
            return best
        return {
            "mode": "none",
            "visible": False,
            "offscreen_dist": float("inf"),
            "center_dist": float("inf"),
        }

    def select_gripper_norms(self, actions):
        if actions is None:
            raise ValueError(
                "Gripper-tint reward inference requires actions aligned with "
                "the generated chunk"
            )
        actions_tensor = torch.as_tensor(
            actions, device=self.device, dtype=torch.float32
        )
        expected_shape = (self.num_envs, self.chunk, 14)
        if tuple(actions_tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected gripper-tint actions {expected_shape}, got "
                f"{tuple(actions_tensor.shape)}"
            )
        actions_np = actions_tensor.detach().cpu().numpy()

        head_view_idx = self.camera_names.index("head")
        head_robotwin_cam = self.robotwin_camera_names[head_view_idx]
        value_min = float(self.cfg.reward_model.get("gripper_value_min", 0.0))
        value_max = float(self.cfg.reward_model.get("gripper_value_max", 1.0))
        if value_max <= value_min:
            raise ValueError(
                f"gripper_value_max must exceed gripper_value_min, got "
                f"{value_min}, {value_max}"
            )

        selected_norms = np.empty(
            (self.num_envs, self.chunk), dtype=np.float32
        )
        for env_idx in range(self.num_envs):
            anchor = self.camera_anchors[env_idx]
            if anchor is None:
                raise RuntimeError(f"Missing RoboTwin camera anchor for env {env_idx}")
            action_seq = actions_np[env_idx]
            ee_pose = self._action14_to_ee_pose(action_seq)
            left_pose = ee_pose[:, 0:7]
            right_pose = ee_pose[:, 7:14]
            w2c_seq = self.projector.build_w2c_sequences(
                anchor, left_pose, right_pose
            )[head_robotwin_cam]
            original_h, original_w = anchor.original_hw_by_cam[head_robotwin_cam]
            intrinsic = scale_intrinsic(
                anchor.intrinsic_by_cam[head_robotwin_cam],
                original_h,
                original_w,
                self.view_height,
                self.view_width,
            )

            for frame_idx in range(self.chunk):
                choices = []
                for side, pose, action_col in (
                    ("left", left_pose, 6),
                    ("right", right_pose, 13),
                ):
                    projection = self._project_point(
                        pose[frame_idx, :3],
                        w2c_seq[frame_idx],
                        intrinsic,
                        self.view_height,
                        self.view_width,
                    )
                    value = float(action_seq[frame_idx, action_col])
                    norm = float(
                        np.clip(
                            (value - value_min) / (value_max - value_min),
                            0.0,
                            1.0,
                        )
                    )
                    choices.append({**projection, "side": side, "norm": norm})

                visible = [choice for choice in choices if choice["visible"]]
                if visible:
                    selected = min(
                        visible, key=lambda choice: choice["center_dist"]
                    )
                else:
                    selected = min(
                        choices,
                        key=lambda choice: (
                            choice["offscreen_dist"], choice["center_dist"]
                        ),
                    )
                selected_norms[env_idx, frame_idx] = selected["norm"]

        return torch.from_numpy(selected_norms).to(
            device=self.device, dtype=torch.float32
        )

    def prepare_reward_obs(self, stacked_obs, actions):
        if self.cfg.reward_model.get("input_view", "head") != "head":
            raise ValueError(
                "Gripper-tint reward model currently supports input_view=head"
            )
        expected_frames = self.num_envs * self.chunk
        if stacked_obs.shape[0] != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} reward frames, got {stacked_obs.shape[0]}"
            )

        head_view_idx = self.camera_names.index("head")
        head_start = head_view_idx * self.view_height
        head_end = head_start + self.view_height
        if stacked_obs.shape[-2] < head_end:
            raise ValueError(
                f"Stacked reward image height {stacked_obs.shape[-2]} is too small "
                f"for head crop [{head_start}:{head_end}]"
            )
        head_obs = stacked_obs[:, :, head_start:head_end, :].reshape(
            self.num_envs,
            self.chunk,
            3,
            self.view_height,
            self.view_width,
        )

        gripper_norms = self.select_gripper_norms(actions)
        alpha = float(np.clip(self.cfg.reward_model.get("tint_alpha", 0.42), 0, 1))
        green = float(self.cfg.reward_model.get("tint_green", 35.0))
        colors = torch.stack(
            [
                255.0 * gripper_norms,
                torch.full_like(gripper_norms, green),
                255.0 * (1.0 - gripper_norms),
            ],
            dim=-1,
        )[:, :, :, None, None]

        head_uint8 = torch.clamp((head_obs + 1.0) * 127.5, 0.0, 255.0).to(
            torch.uint8
        )
        tinted_uint8 = (
            head_uint8.to(torch.float32) * (1.0 - alpha) + colors * alpha
        ).clamp(0.0, 255.0).to(torch.uint8)
        return (tinted_uint8.to(torch.float32) / 127.5 - 1.0).reshape(
            expected_frames, 3, self.view_height, self.view_width
        )
