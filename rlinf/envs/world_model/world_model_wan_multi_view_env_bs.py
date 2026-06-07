# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

_BS_DIFFSYNTH_ROOT = Path(
    os.environ.get("DIFFSYNTH_BS_PATH", "/data/zsq/diffsynth-studio-wujie-ppu_bs")
)
if _BS_DIFFSYNTH_ROOT.exists() and str(_BS_DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BS_DIFFSYNTH_ROOT))

from diffsynth.models.reward_model import ResnetRewModel

from diffsynth.models.wan_video_vace_wan22_codex import VaceWan22CodexModel
from diffsynth.pipelines.wan_video_new_wan22_codex import ModelConfig, WanVideoPipeline
from diffsynth.models.utils import load_state_dict
from diffsynth.utils import PipelineUnit
from diffsynth.trainers.arxx5_dataset4_wancontrolmultiview import (
    generate_raymap,
    generate_traj_map,
    get_color_intrinsics,
    load_calib_extrinsics,
    make_center_ray_w2c_sequences,
    make_per_camera_sequences,
    scale_intrinsic,
    to_uint8_img,
)

from PIL import Image

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.envs.world_model.robotwin_camera_fk import (
    RobotWinCameraFKProjector,
    make_abs_actions_from_fk_wxyz,
    scale_intrinsic as robotwin_scale_intrinsic,
    to_robotwin_camera_name,
)

__all__ = ["WanmultiviewEnv"]


class WanVideoUnit_TI2V2ContextEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "latents",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_video, noise, latents, tiled, tile_size, tile_stride):
        if input_video is None:
            return {}
        context_frame_count = (
            len(input_video[0])
            if len(input_video) > 0 and isinstance(input_video[0], (list, tuple))
            else len(input_video)
        )
        if context_frame_count < 5:
            raise ValueError(
                f"TI2V2 context inference expects 5 context frames, got {context_frame_count}."
            )
        pipe.load_models_to_device(["vae"])
        context = pipe.preprocess_video(input_video)
        context_latents = pipe.vae.encode(
            context,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if context_latents.shape[2] < 2:
            raise ValueError(
                f"5 context frames should encode to at least 2 latent slots, got {context_latents.shape}."
            )
        if latents is None:
            latents = noise.clone()
        latents[:, :, 0:2] = context_latents[:, :, 0:2]
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "clean_latent_slots": 2,
            "first_frame_latents": context_latents[:, :, 0:2],
        }


class WanmultiviewEnv(BaseWorldEnv):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        record_metrics=True,
        worker_info=None,
    ):
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )
        # Reset state management
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids # TODO
        self.group_size = cfg.group_size # 一个组里有多少个环境共享同一个reset state id
        self.num_group = self.num_envs // self.group_size # 组的数量

        # Initialize reset state generator
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)

        # Update reset state ids
        self.update_reset_state_ids()

        # Model hyperparameters
        self.num_inference_steps = cfg.num_inference_steps
        self.chunk = int(cfg.get("wm_chunk", cfg.get("horizon_frames", 8)))  # Ta = 8
        self.condition_frame_length = int(
            cfg.get("context_frames", cfg.get("dataset_context_frames", 5))
        )  # To = 5
        self.num_frames = int(
            cfg.get("wm_num_frames", self.condition_frame_length + self.chunk)
        )  # Total number of frames to encode = 13
        assert self.num_frames == self.condition_frame_length + self.chunk, (
            "num_frames must be equal to condition_frame_length + action_chunk_length"
        )

        self.image_size = tuple(cfg.image_size)

        #
        self.retain_action = cfg.get("retain_action", True)  # Default True TODO
        self.enable_kir = cfg.get("enable_kir", True) # TODO
        self.is_robotwin_env = cfg.get("wm_env_type", "robotwin") == "robotwin"

        self.camera_names = list(
            cfg.get("camera_names", ["head", "left_wrist", "right_wrist"])
        )
        self.robotwin_camera_names = (
            [to_robotwin_camera_name(name) for name in self.camera_names]
            if self.is_robotwin_env
            else []
        )
        self.camera_axis_mode = cfg.get("camera_axis_mode", "identity")
        self.view_height = int(cfg.get("view_height", self.image_size[0]))
        self.view_width = int(cfg.get("view_width", self.image_size[1]))
        self.camera_original_height = int(cfg.get("camera_original_height", 480))
        self.camera_original_width = int(cfg.get("camera_original_width", 640))
        self.prompt = cfg.get("prompt", "机械臂按照要求移动夹爪执行任务")
        self.traj_radius = int(cfg.get("traj_radius", 40))
        self.traj_radius_mode = cfg.get("traj_radius_mode", "perspective")
        self.traj_min_radius = int(cfg.get("traj_min_radius", 14))
        self.traj_max_radius = int(cfg.get("traj_max_radius", 58))
        self.traj_ref_depth = float(cfg.get("traj_ref_depth", 0.30))
        self.traj_near_depth = float(cfg.get("traj_near_depth", 0.19))
        self.traj_far_depth = float(cfg.get("traj_far_depth", 0.69))
        self.ray_o_vmin = float(cfg.get("ray_o_vmin", -1.5))
        self.ray_o_vmax = float(cfg.get("ray_o_vmax", 1.5))
        self.ray_d_vmin = float(cfg.get("ray_d_vmin", -1.0))
        self.ray_d_vmax = float(cfg.get("ray_d_vmax", 1.0))
        self.cfg_scale = float(cfg.get("cfg_scale", 1.0))
        self.tiled = bool(cfg.get("tiled", False))
        self.wm_infer_batch_size = int(cfg.get("wm_infer_batch_size", max(1, self.num_envs)))
        self.wm_infer_batch_size = max(1, self.wm_infer_batch_size)

        if self.is_robotwin_env:
            self.intrinsics = None
            self.extrinsics = None
        else:
            self.intrinsics = get_color_intrinsics()
            self.extrinsics = load_calib_extrinsics(
                Path(
                    cfg.get(
                        "head_left_calib",
                        "/data/zsq/outputs/calib_eye_to_hand_head_left/result_eye_to_hand.json",
                    )
                ),
                Path(
                    cfg.get(
                        "head_right_calib",
                        "/data/zsq/outputs/calib_eye_to_hand_head_right/result_eye_to_hand.json",
                    )
                ),
                Path(
                    cfg.get(
                        "left_calib",
                        "/data/zsq/outputs/calib_eye_in_hand_left/result_eye_in_hand.json",
                    )
                ),
                Path(
                    cfg.get(
                        "right_calib",
                        "/data/zsq/outputs/calib_eye_in_hand_right/result_eye_in_hand.json",
                    )
                ),
            )
        self.fk_fn = self._resolve_fk_fn(cfg.get("fk_fn", None))
        self.left_fk_fn = self._resolve_fk_fn(cfg.get("left_fk_fn", None)) or self.fk_fn
        self.right_fk_fn = self._resolve_fk_fn(cfg.get("right_fk_fn", None)) or self.fk_fn
        self.fk_pose_quat_order = cfg.get("fk_pose_quat_order", "qwxyz")
        self.fk_urdf_path = cfg.get("fk_urdf_path", None)
        self.fk_offset_path = cfg.get("fk_offset_path", None)
        self.fk_base_link = cfg.get("fk_base_link", "base_link")
        self.fk_tip_link = cfg.get("fk_tip_link", "link6")
        self.fk_offset_mode = cfg.get("fk_offset_mode", None)
        self.fk_state_unit = cfg.get("fk_state_unit", "rad")
        self._configure_fk_modules()
        self.robotwin_projector = (
            RobotWinCameraFKProjector(self.robotwin_camera_names)
            if self.is_robotwin_env
            else None
        )
        self.robotwin_episode_source_paths = (
            self._load_robotwin_episode_source_paths(cfg.initial_image_path)
            if self.is_robotwin_env
            else []
        )

        # load pipeline
        self.pipe = self._build_pipeline()

        # Load reward model if specified
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # Initialize state
        # Will be a tensor [num_envs, 3, 1, T, h, w]
        # TODO
        self.current_obs = None
        self.task_descriptions = [""] * self.num_envs # 数量是总的轨迹数
        self.init_ee_poses = [None] * self.num_envs
        self.current_action14 = torch.zeros(self.num_envs, 14, device=self.device)
        self.reference_frames = [None] * self.num_envs
        self.reference_action14 = torch.zeros(self.num_envs, 14, device=self.device)
        self.robotwin_camera_anchors = [None] * self.num_envs
        self.last_control_video = [[] for _ in range(self.num_envs)]
        self.last_control_video_chunks = []

        # Image queue for condition frames to generate video,
        # keep length of condition_frame_length
        self.image_queue = [
            [None] * self.condition_frame_length for _ in range(self.num_envs)
        ]

        # Condition action to generate video,
        # keep length of condition_frame_length
        self.condition_action = torch.zeros(
            self.num_envs,
            self.condition_frame_length,
            14,
        )

        self.reset_gripper_open = cfg.get("reset_gripper_open", True)
        # If reset_gripper_open is True and the environment is Libero, set the gripper open action to -1
        if self.reset_gripper_open and self.is_libero_env:
            self.condition_action[:, :, -1] = -1

        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        self._is_offloaded = False

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path,
            action_key=cfg.get("dataset_action_key", "action"),
            state_key=cfg.get("dataset_state_key", "state"),
            enable_kir=self.enable_kir,
        )

    def _load_robotwin_episode_source_paths(self, initial_image_path):
        manifest_path = Path(initial_image_path) / "manifest.json"
        if not manifest_path.exists():
            return []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mapping = manifest.get("episode_mapping", [])
        source_paths = []
        for item in mapping:
            source_path = item.get("source_path") or item.get("path")
            source_paths.append(str(source_path) if source_path else None)
        return source_paths

    def _robotwin_source_path_for_episode(self, episode_idx):
        episode_idx = int(episode_idx)
        if (
            episode_idx < len(self.robotwin_episode_source_paths)
            and self.robotwin_episode_source_paths[episode_idx]
        ):
            source_path = Path(self.robotwin_episode_source_paths[episode_idx])
            if source_path.exists():
                return str(source_path)
            raise FileNotFoundError(
                f"RobotWin manifest source_path for episode {episode_idx} does not exist: {source_path}"
            )
        raise ValueError(
            "RobotWin world model needs initial_image_path/manifest.json with "
            f"episode_mapping[*].source_path. Cannot resolve episode {episode_idx}."
        )

    def _as_plain_config_value(self, value):
        if isinstance(value, (list, tuple)):
            return [self._as_plain_config_value(item) for item in value]
        if value.__class__.__name__ == "ListConfig":
            return [self._as_plain_config_value(item) for item in value]
        return value

    def _build_pipeline(self):
        def split_dit_vace_state_dict(state_dict):
            dit_state, vace_state, unknown_keys = {}, {}, []
            for name, value in state_dict.items():
                if name.startswith("pipe.dit."):
                    dit_state[name[len("pipe.dit."):]] = value
                elif name.startswith("dit."):
                    dit_state[name[len("dit."):]] = value
                elif name.startswith("pipe.vace."):
                    vace_state[name[len("pipe.vace."):]] = value
                elif name.startswith("vace."):
                    vace_state[name[len("vace."):]] = value
                else:
                    unknown_keys.append(name)
            return dit_state, vace_state, unknown_keys
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda:0",
            model_configs=[
                # Paths are loaded from yaml
                ModelConfig(path=self._as_plain_config_value(self.cfg.base_model_path), offload_device="cpu"),
                ModelConfig(path=self._as_plain_config_value(self.cfg.VAE_path), offload_device="cpu"),
                ModelConfig(path=self._as_plain_config_value(self.cfg.T5_path), offload_device="cpu"),
            ],
        )
        # pipe.enable_vram_management()
        pipe.dit.to(self.device)
        pipe.vae.to(self.device)
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.to(self.device)
        pipe.units = [
            WanVideoUnit_TI2V2ContextEmbedder()
            if unit.__class__.__name__ == "WanVideoUnit_InputVideoEmbedder"
            else unit
            for unit in pipe.units
        ]

        pipe.vace = VaceWan22CodexModel(vace_in_dim=self.cfg.vace_in_dim).to(dtype=pipe.torch_dtype, device=pipe.device)
        state_dict = load_state_dict(self.cfg.vace_model_path)
        dit_state, vace_state, unknown_keys = split_dit_vace_state_dict(state_dict)
        if dit_state or vace_state:
            if dit_state:
                incompatible = pipe.dit.load_state_dict(dit_state, strict=False)
                print(
                    f"[INFO] loaded DiT weights from {self.cfg.vace_model_path}: keys={len(dit_state)} "
                    f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
                )
            if vace_state:
                if any(name.startswith("vace_global_") for name in vace_state):
                    pipe.vace.enable_global_cross_attn(global_context_dim=16)
                incompatible = pipe.vace.load_state_dict(vace_state, strict=False)
                print(
                    f"[INFO] loaded VACE weights from {self.cfg.vace_model_path}: keys={len(vace_state)} "
                    f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
                )
            if unknown_keys:
                print(f"[WARN] ignored {len(unknown_keys)} non-DiT/VACE checkpoint keys")
        else:
            if any(name.startswith("vace_global_") for name in state_dict):
                pipe.vace.enable_global_cross_attn(global_context_dim=16)
            pipe.vace.load_state_dict(state_dict)
            print(f"[INFO] loaded pure VACE checkpoint from {self.cfg.vace_model_path}: keys={len(state_dict)}")

        return pipe

    def _resolve_fk_fn(self, fn_or_path):
        if fn_or_path is None:
            return None
        if callable(fn_or_path):
            return fn_or_path
        if not isinstance(fn_or_path, str):
            raise TypeError(f"FK function must be callable or import path, got {type(fn_or_path)}")
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
            raise NotImplementedError(
                "ARXX5 VACE controls need FK. Set cfg.fk_fn, cfg.left_fk_fn/right_fk_fn, "
                "or override _call_fk to map 6-dim joints to xyz+qwxyz ee pose."
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
            raise ValueError(f"Unsupported fk_pose_quat_order: {self.fk_pose_quat_order}")
        return pose.astype(np.float32)

    def _action14_to_ee_pose_and_state(self, action14_seq):
        action14_seq = np.asarray(action14_seq, dtype=np.float32)
        if action14_seq.ndim != 2 or action14_seq.shape[-1] != 14:
            raise ValueError(f"Expected action sequence [T, 14], got {action14_seq.shape}")

        left_pose = self._call_fk(action14_seq[:, 0:6], "left")
        right_pose = self._call_fk(action14_seq[:, 7:13], "right")
        if left_pose.shape[0] != action14_seq.shape[0] or right_pose.shape[0] != action14_seq.shape[0]:
            raise ValueError(
                f"FK output length mismatch: actions={action14_seq.shape[0]}, "
                f"left={left_pose.shape[0]}, right={right_pose.shape[0]}"
            )

        ee_pose = np.concatenate([left_pose, right_pose], axis=-1).astype(np.float32)
        state = action14_seq.astype(np.float32).copy()
        return ee_pose, state

    def _tensor_frame_to_pil(self, frame):
        frame = frame.detach().cpu()
        if frame.ndim == 4:
            frame = frame[:, 0]
        img = frame.numpy().transpose(1, 2, 0)
        if img.max() <= 1.2:
            img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255)
        return Image.fromarray(img.astype(np.uint8))

    def _pad_action_sequence(self, action_seq):
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.shape[0] < self.num_frames:
            pad = np.repeat(
                action_seq[-1:], self.num_frames - action_seq.shape[0], axis=0
            )
            action_seq = np.concatenate([action_seq, pad], axis=0)
        elif action_seq.shape[0] > self.num_frames:
            raise ValueError(f"Action sequence length {action_seq.shape[0]} exceeds num_frames {self.num_frames}")
            action_seq = action_seq[-self.num_frames :]
        return action_seq

    def _build_arxx5_controls(self, action_seq):
        action_seq = self._pad_action_sequence(action_seq)
        ee_pose, state = self._action14_to_ee_pose_and_state(action_seq)
        traj_actions_by_cam, traj_w2c_by_cam = make_per_camera_sequences(
            ee_pose, state, self.extrinsics, self.camera_axis_mode
        )
        ray_w2c_by_cam = make_center_ray_w2c_sequences(
            ee_pose, self.extrinsics, self.camera_axis_mode
        )

        control_per_cam = {}
        ray_o_per_cam = {}
        ray_d_per_cam = {}
        for cam in self.camera_names:
            k = scale_intrinsic(
                self.intrinsics[cam],
                self.camera_original_height,
                self.camera_original_width,
                self.view_height,
                self.view_width,
            )
            control_per_cam[cam] = generate_traj_map(
                traj_actions_by_cam[cam],
                traj_w2c_by_cam[cam],
                k,
                self.view_height,
                self.view_width,
                radius=self.traj_radius,
                radius_mode=self.traj_radius_mode,
                min_radius=self.traj_min_radius,
                max_radius=self.traj_max_radius,
                ref_depth=self.traj_ref_depth,
                near_depth=self.traj_near_depth,
                far_depth=self.traj_far_depth,
            )

            ray_o_seq = []
            ray_d_seq = []
            for t in range(action_seq.shape[0]):
                c2w = np.linalg.inv(ray_w2c_by_cam[cam][t]).astype(np.float32)
                ray_o, ray_d = generate_raymap(k, c2w, self.view_height, self.view_width)
                ray_o_seq.append(to_uint8_img(ray_o, self.ray_o_vmin, self.ray_o_vmax))
                ray_d_seq.append(to_uint8_img(ray_d, self.ray_d_vmin, self.ray_d_vmax))
            ray_o_per_cam[cam] = ray_o_seq
            ray_d_per_cam[cam] = ray_d_seq

        control_video = []
        ray_map_o = []
        ray_map_d = []
        for t in range(action_seq.shape[0]):
            control_video.append(
                Image.fromarray(
                    np.concatenate(
                        [control_per_cam[cam][t] for cam in self.camera_names], axis=0
                    )
                )
            )
            ray_map_o.append(
                Image.fromarray(
                    np.concatenate(
                        [ray_o_per_cam[cam][t] for cam in self.camera_names], axis=0
                    )
                )
            )
            ray_map_d.append(
                Image.fromarray(
                    np.concatenate(
                        [ray_d_per_cam[cam][t] for cam in self.camera_names], axis=0
                    )
                )
            )
        return control_video, ray_map_o, ray_map_d

    def _build_robotwin_camera_anchor(self, env_idx, source_path, reference_action14):
        if self.robotwin_projector is None:
            raise RuntimeError("RobotWin camera projector is not initialized.")
        reference_action14 = np.asarray(reference_action14, dtype=np.float32).reshape(1, 14)
        ee_pose, _ = self._action14_to_ee_pose_and_state(reference_action14)
        left_pose0 = ee_pose[0, 0:7]
        right_pose0 = ee_pose[0, 7:14]
        self.robotwin_camera_anchors[env_idx] = self.robotwin_projector.build_anchor(
            source_path=source_path,
            left_pose0_wxyz=left_pose0,
            right_pose0_wxyz=right_pose0,
        )

    def _build_robotwin_controls(self, env_idx, action_seq):
        if self.robotwin_projector is None:
            raise RuntimeError("RobotWin camera projector is not initialized.")
        anchor = self.robotwin_camera_anchors[env_idx]
        if anchor is None:
            raise RuntimeError(f"Missing RobotWin camera anchor for env {env_idx}.")

        action_seq = self._pad_action_sequence(action_seq)
        ee_pose, _ = self._action14_to_ee_pose_and_state(action_seq)
        left_pose = ee_pose[:, 0:7]
        right_pose = ee_pose[:, 7:14]
        abs_actions = make_abs_actions_from_fk_wxyz(action_seq, left_pose, right_pose)
        w2c_by_robotwin_cam = self.robotwin_projector.build_w2c_sequences(
            anchor, left_pose, right_pose
        )

        control_per_cam = {}
        ray_o_per_cam = {}
        ray_d_per_cam = {}
        for cam, robotwin_cam in zip(self.camera_names, self.robotwin_camera_names):
            original_h, original_w = anchor.original_hw_by_cam[robotwin_cam]
            k = robotwin_scale_intrinsic(
                anchor.intrinsic_by_cam[robotwin_cam],
                original_h,
                original_w,
                self.view_height,
                self.view_width,
            )
            w2c_seq = w2c_by_robotwin_cam[robotwin_cam]
            control_per_cam[cam] = generate_traj_map(
                abs_actions,
                w2c_seq,
                k,
                self.view_height,
                self.view_width,
                radius=self.traj_radius,
                radius_mode=self.traj_radius_mode,
                min_radius=self.traj_min_radius,
                max_radius=self.traj_max_radius,
                ref_depth=self.traj_ref_depth,
                near_depth=self.traj_near_depth,
                far_depth=self.traj_far_depth,
            )

            ray_o_seq = []
            ray_d_seq = []
            for t in range(action_seq.shape[0]):
                c2w = np.linalg.inv(w2c_seq[t]).astype(np.float32)
                ray_o, ray_d = generate_raymap(k, c2w, self.view_height, self.view_width)
                ray_o_seq.append(to_uint8_img(ray_o, self.ray_o_vmin, self.ray_o_vmax))
                ray_d_seq.append(to_uint8_img(ray_d, self.ray_d_vmin, self.ray_d_vmax))
            ray_o_per_cam[cam] = ray_o_seq
            ray_d_per_cam[cam] = ray_d_seq

        control_video = []
        ray_map_o = []
        ray_map_d = []
        for t in range(action_seq.shape[0]):
            control_video.append(
                Image.fromarray(
                    np.concatenate(
                        [control_per_cam[cam][t] for cam in self.camera_names], axis=0
                    )
                )
            )
            ray_map_o.append(
                Image.fromarray(
                    np.concatenate(
                        [ray_o_per_cam[cam][t] for cam in self.camera_names], axis=0
                    )
                )
            )
            ray_map_d.append(
                Image.fromarray(
                    np.concatenate(
                        [ray_d_per_cam[cam][t] for cam in self.camera_names], axis=0
                    )
                )
            )
        return control_video, ray_map_o, ray_map_d


    def _load_reward_model(self):
        if self.cfg.reward_model.type == "ResnetRewModel":
            rew_model = ResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                threshold=self.cfg.reward_model.get("threshold", None),
            )
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            from diffsynth.models.reward_model import TaskEmbedResnetRewModel

            rew_model = TaskEmbedResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        elif self.cfg.reward_model.type == "DummyRewardModel":
            # raise NotImplementedError(f'Not setting rew')
            class DummyRewardModel(torch.nn.Module):
                def predict_rew(self, obs, instructions=None):
                    del instructions
                    return torch.zeros(obs.shape[0], device=obs.device, dtype=torch.float32)

            rew_model = DummyRewardModel()
        else:
            raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")
        return rew_model

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True # 指定环境的mask为True
            self.prev_step_reward[mask] = 0.0 # 指定环境的pre_step_reward清零
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
            # self._elapsed_steps = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward # [num_envs] 累积奖励
        # Update success_once based on terminations
        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = torch.full(
            (self.num_envs,),
            self.elapsed_steps,
            dtype=torch.float32,
            device=self.device,
        )
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        """Calculate step reward"""
        reward_diffs = torch.zeros(
            (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
        )
        for i in range(self.chunk):
            reward_diffs[:, i] = (
                self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            )
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]

        if self.use_rel_reward:
            return reward_diffs
        else:
            return chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        """
        Estimate success (terminations) based on reward values.
        Success is estimated when reward exceeds a threshold (default: 0.9).
        """
        # Get success threshold from config, default to 0.9
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)

        # Check if any reward in the chunk exceeds the threshold
        # chunk_rewards shape: [num_envs, chunk]
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]  # [num_envs] 每个环境沿时间维度的最大值
        success_estimated = max_reward_in_chunk >= success_threshold # [num_envs] 每个环境的最大reward超过阈值则认为成功

        return success_estimated.to(self.device)

    def update_reset_state_ids(self):
        """Updates the reset state IDs for environment initialization."""
        # Get total number of episodes available
        total_num_episodes = len(self.dataset)

        # Generate random reset state ids
        reset_state_ids = torch.randint(
            low=0,
            high=total_num_episodes,
            size=(self.num_group,),
            generator=self._generator,
        )
        # 对每一组都得到一个随机的reset state id，范围是[0, total_num_episodes)，shape是[num_group]
        # 每个组里的环境共享同一个reset state id
        # Repeat for each environment in the group
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None, # 选择dataset里哪条轨迹
    ):
        self.onload()
        self.elapsed_steps = 0

        # Handle first reset with fixed reset state ids
        if self.is_start: # TODO
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        num_envs = self.num_envs
        if len(self.dataset) < num_envs:
            raise ValueError(
                f"Not enough episodes in dataset. Found {len(self.dataset)}, need {num_envs}"
            )

        # If episode_indices not provided, randomly select
        if episode_indices is None:
            # Set random seed if provided
            if seed is not None:
                if isinstance(seed, list):
                    np.random.seed(seed[0])
                else:
                    np.random.seed(seed)

            # Randomly select episode indices
            episode_indices = np.random.choice(
                len(self.dataset), size=num_envs, replace=False
            )
        else:
            # Convert to numpy if tensor
            if isinstance(episode_indices, torch.Tensor):
                episode_indices = episode_indices.cpu().numpy()

        # Load first frame from each selected episode
        img_tensors = []
        task_descriptions = []
        init_ee_poses = []
        init_states14 = []
        condition_actions = []

        for env_idx, episode_idx in enumerate(episode_indices):
            # Get episode data from dataset wrapper
            # 对于所有环境，每个环境选的轨迹不一样
            # TODO dataset的定义在哪
            episode_data = self.dataset[episode_idx]

            # Get first frame from start_items
            if len(episode_data["start_items"]) == 0:
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]

            # Get task description
            task_desc = episode_data.get("task", "use the left arm to put the orange square into the green box and use the right arm to put the blue square into the green box")
            task_descriptions.append(str(task_desc))

            # Get image from frame
            if "image" not in first_frame:
                raise ValueError(f"No 'image' key in frame for episode {episode_idx}")

            img_tensor = first_frame[
                "image"
            ]  # Shape: [3, H, W], dtype: float, range: [0, 1]
            # [3, 480, 244], float32, [0,1]
            # Wan requires images in PIL format

            if "observation.ee_pose" in first_frame:
                init_ee_pose = first_frame["observation.ee_pose"]
                if isinstance(init_ee_pose, torch.Tensor):
                    init_ee_pose = init_ee_pose.numpy()
                init_ee_poses.append(init_ee_pose)
            else:
                init_ee_poses.append(None)

            init_state14 = None
            if "action" in first_frame:
                init_state14 = first_frame["action"]
            elif "observation.state" in first_frame:
                raise NotImplementedError("用action表示初始状态")
                init_state14 = first_frame["observation.state"]
            if isinstance(init_state14, torch.Tensor):
                init_state14 = init_state14.numpy()
            init_states14.append(init_state14)

            # Resize if needed
            if img_tensor.shape[1:] != self.image_size:
                img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
                img_tensor = F.interpolate(
                    img_tensor,
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                img_tensor = img_tensor.squeeze(0)  # [3, H, W]

            # Normalize to [-1, 1]
            img_tensor = self.trans_norm(img_tensor)

            # Repeat to fill condition frames: [3, H, W] -> [3, condition_frame_length, H, W]
            env_img_tensor = img_tensor.unsqueeze(1).repeat(
                1, self.condition_frame_length, 1, 1
            )  # [3, condition_frame_length, H, W]

            env_condition_action = np.zeros(
                (self.condition_frame_length, 14), dtype=np.float32
            ) # TODO

            if self.reset_gripper_open and self.is_libero_env:
                raise NotImplementedError("Gripper open action setting is not implemented for Libero env yet.")
                env_condition_action[:, -1] = -1

            # KIR trick: use the last four frames as condition frames, while
            # keeping the reference frame unchanged as the first frame.
            # target_items = episode_data.get("target_items", [])
            target_items = []
            # first condition frame is the reference frame,
            # so the length of target_items should be condition_frame_length - 1
            if len(target_items) == self.condition_frame_length - 1:
                for target_idx, target_frame in enumerate(target_items):
                    if "image" not in target_frame or "action" not in target_frame:
                        raise ValueError(
                            f"No 'image' or 'action' key in target frame for episode {episode_idx}"
                        )
                    target_img = target_frame["image"]
                    if target_img.shape[1:] != self.image_size:
                        target_img = target_img.unsqueeze(0)  # [1, 3, H, W]
                        target_img = F.interpolate(
                            target_img,
                            size=self.image_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        target_img = target_img.squeeze(0)  # [3, H, W]
                    target_img = self.trans_norm(target_img)

                    # keep first frame as reference frame, update the rest
                    env_img_tensor[:, target_idx + 1] = target_img
                    env_condition_action[target_idx + 1] = target_frame["action"]

            img_tensors.append(env_img_tensor)
            condition_actions.append(torch.from_numpy(env_condition_action))

        # Stack all environments: [num_envs, 3, condition_frame_length, H, W]
        # [8, 3, 5, 256, 256]
        stacked_imgs = torch.stack(img_tensors, dim=0).to(self.device)

        # Reshape to [num_envs, 3, 1, condition_frame_length, H, W] for compatibility
        # [8, 3, 1, 5, 256, 256]
        self.current_obs = stacked_imgs.unsqueeze(2).to(self.device)
        
        self.condition_action = torch.stack(condition_actions, dim=0).to(self.device)
        print(f'condition_action:{self.condition_action.shape}')

        num_envs, c, v, t, h, w = self.current_obs.shape
        assert t == self.condition_frame_length, (
            f"Unexpected current_obs shape: {self.current_obs.shape}, expected {num_envs, c, v, self.condition_frame_length, h, w}"
        )

        self._reset_metrics()

        # Initialize action buffer (if needed)
        # But we'll keep it for compatibility
        if hasattr(self.cfg, "action_dim"):
            action_dim = self.cfg.action_dim
        else:
            action_dim = 14

        # Initialize with zeros or from the 14-dim joint-gripper state.
        init_actions = []
        current_actions14 = []
        for init_state14 in init_states14:
            if init_state14 is not None:
                init_action = init_state14.flatten()
                # Pad or truncate to action_dim
                if len(init_action) < action_dim:
                    raise NotImplementedError("动作维度不对")
                    init_action = np.pad(
                        init_action, (0, action_dim - len(init_action))
                    )
                elif len(init_action) > action_dim:
                    raise NotImplementedError("动作维度不对")
                    init_action = init_action[:action_dim]
            else:
                init_action = np.zeros(action_dim, dtype=np.float32)
            init_actions.append(init_action) 

        for env_idx, init_state14 in enumerate(init_states14):
            cond_action = condition_actions[env_idx][-1].numpy()
            if np.any(cond_action):
                current_action = cond_action
            elif init_state14 is not None:
                current_action = init_state14.flatten()
                if len(current_action) < 14:
                    current_action = np.pad(current_action, (0, 14 - len(current_action)))
                elif len(current_action) > 14:
                    current_action = current_action[:14]
            else:
                current_action = np.zeros(14, dtype=np.float32)
            current_actions14.append(current_action.astype(np.float32))

        self.current_action14 = torch.from_numpy(np.stack(current_actions14, axis=0)).to(
            self.device
        )
        self.reference_action14 = self.current_action14.clone()
        if self.condition_frame_length > 0:
            self.condition_action = self.current_action14[:, None, :].repeat(
                1, self.condition_frame_length, 1
            ).to(self.condition_action.device, dtype=self.condition_action.dtype)

        if self.is_robotwin_env:
            for env_idx, episode_idx in enumerate(episode_indices):
                source_path = self._robotwin_source_path_for_episode(episode_idx)
                self._build_robotwin_camera_anchor(
                    env_idx,
                    source_path,
                    self.reference_action14[env_idx].detach().cpu().numpy(),
                )

        # TI2V2 context is anchored by the first frame. The first rollout uses
        # [first_frame] * context_length; later rollouts use
        # [first_frame] + recent generated frames.
        for env_idx in range(num_envs):
            reference_frame = self.current_obs[env_idx, :, 0, 0:1, :, :].clone()
            self.reference_frames[env_idx] = reference_frame
            self.image_queue[env_idx] = [
                reference_frame.clone() for _ in range(self.condition_frame_length)
            ]
            self.current_obs[env_idx, :, 0, :, :, :] = reference_frame.repeat(
                1, self.condition_frame_length, 1, 1
            )

        # Store init_ee_poses
        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses

        # Wrap observation to match libero_env format
        extracted_obs = self._wrap_obs()
        infos = {}

        return extracted_obs, infos

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("step in Wan Env is not impl, use chunk_step instead")

    def _infer_next_chunk_rewards(self):
        """Predict next reward using the reward model"""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        # Extract chunk observations
        num_envs, c, v, t, h, w = self.current_obs.shape
        extract_chunk_obs = self.current_obs.permute(
            0, 3, 1, 2, 4, 5
        )  # [num_envs, chunk + condition_frame_length, 3, v, h, w]

        if self.cfg.reward_model.type == "ResnetRewModel":
            extract_chunk_obs = extract_chunk_obs[
                :, -self.chunk :, :, :, :, :
            ]  # [num_envs, chunk, 3, v, h, w]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.chunk, 3, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(
                2
            )  # [num_envs * chunk, 3, h, w]
            extract_chunk_obs = extract_chunk_obs.to(self.device)

            rewards = self.reward_model.predict_rew(extract_chunk_obs)
            rewards = rewards.reshape(self.num_envs, self.chunk)
        elif self.cfg.reward_model.type == "DummyRewardModel":
            print(f'ONLY FOR TEST')
            extract_chunk_obs = extract_chunk_obs[
                :, -self.chunk :, :, :, :, :
            ]  # [num_envs, chunk, 3, v, h, w]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.chunk, 3, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(
                2
            )  # [num_envs * chunk, 3, h, w]
            extract_chunk_obs = extract_chunk_obs.to(self.device)

            rewards = self.reward_model.predict_rew(extract_chunk_obs)
            rewards = rewards.reshape(self.num_envs, self.chunk)
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            extract_chunk_obs = extract_chunk_obs[
                :, -self.chunk :, :, :, :, :
            ]  # [num_envs, chunk, 3, v, h, w]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.chunk, 3, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(
                2
            )  # [num_envs * chunk, 3, h, w]
            extract_chunk_obs = extract_chunk_obs.to(self.device)

            # Prepare instructions for each frame in the chunk
            # Each environment has one task description, repeat it for each frame in the chunk
            instructions = []
            for env_idx in range(self.num_envs):
                task_desc = self.task_descriptions[env_idx]
                # Repeat the instruction for each frame in the chunk
                instructions.extend([task_desc] * self.chunk)

            # Predict rewards with instruction conditioning
            rewards = self.reward_model.predict_rew(extract_chunk_obs, instructions)
            rewards = rewards.reshape(self.num_envs, self.chunk)
        else:
            raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")

        return rewards

    def _infer_next_chunk_frames_single_env_loop(self, actions):
        """Predict next frame chunk using the wan model"""
        num_envs = self.num_envs
        assert actions.shape[0] == self.num_envs, (
            f"Actions shape {actions.shape} does not match num_envs {self.num_envs}"
        )

        actions_tensor = (
            torch.from_numpy(actions).to(self.device)
            if isinstance(actions, np.ndarray)
            else actions.to(self.device)
        ).float()
        if actions_tensor.ndim != 3 or actions_tensor.shape[-1] != 14:
            raise ValueError(f"Expected actions [num_envs, chunk, 14], got {actions_tensor.shape}")

        self.condition_action = self.condition_action.to(
            device=actions_tensor.device, dtype=actions_tensor.dtype
        ) # [num_envs, condition_frame_length, 14]每个环境condition_frame_length步的动作条件
        self.current_action14 = self.current_action14.to(
            device=actions_tensor.device, dtype=actions_tensor.dtype
        ) # [num_envs, 14]每个环境上一个chunk的最后一步动作，作为下一个chunk的初始状态

        if self.condition_frame_length > 1:
            action_window = torch.cat([self.condition_action, actions_tensor], dim=1)
            # 表示context+horizon帧完整的动作序列，送入WAN进行生成
        elif self.retain_action:
            action_window = torch.cat([self.condition_action, actions_tensor], dim=1)
        else:
            action_window = torch.cat(
                [self.current_action14[:, None, :], actions_tensor], dim=1
            )

        all_samples = []
        last_control_video = []
        target_size = (self.view_width, self.view_height * len(self.camera_names))

        for env_idx in range(num_envs):
            context_frames = self.image_queue[env_idx]
            # self.image_queue存首帧锚定的5帧上下文，如果没有就用当前obs的最后一帧复制填充
            if not context_frames or context_frames[0] is None:
                raise ValueError(f"Image queue for env {env_idx} is empty or uninitialized")
                context_frames = [
                    self.current_obs[env_idx, :, 0, -1:, :, :]
                    for _ in range(self.condition_frame_length)
                ]
            input_video = []
            for context_frame in context_frames[-self.condition_frame_length :]:
                input_image = self._tensor_frame_to_pil(context_frame)
                if input_image.size != target_size:
                    input_image = input_image.resize(target_size, Image.BILINEAR)
                input_video.append(input_image)
            while len(input_video) < self.condition_frame_length:
                raise ValueError("context不足5")
                input_video.insert(0, input_video[0])

            action_seq = action_window[env_idx].detach().cpu().numpy()
            action_seq = self._pad_action_sequence(action_seq)
            if self.is_robotwin_env:
                control_video, ray_map_o, ray_map_d = self._build_robotwin_controls(
                    env_idx, action_seq
                )
            else:
                control_video, ray_map_o, ray_map_d = self._build_arxx5_controls(
                    action_seq
                )
            pred_control_video = control_video[
                self.condition_frame_length : self.condition_frame_length + self.chunk
            ]
            last_control_video.append(pred_control_video)
            prompt = self.task_descriptions[env_idx] or self.prompt

            kwargs = {
                "prompt": prompt,
                # "seed": int(self.cfg.get("seed", 0)) + int(self.elapsed_steps) + env_idx,
                "seed": 42,
                "tiled": self.tiled,
                "input_video": input_video,
                "vace_video": control_video,
                "ray_map_o": ray_map_o,
                "ray_map_d": ray_map_d,
                "height": target_size[1],
                "width": target_size[0],
                "num_frames": self.num_frames,
                "num_inference_steps": self.num_inference_steps,
                "cfg_scale": self.cfg_scale,
                "progress_bar_cmd": lambda x: x,
            }

            output = self.pipe(**kwargs)
            frames = []
            for img in output:
                arr = np.array(img) / 255.0
                arr = arr * 2.0 - 1.0
                frames.append(arr)

            video = np.stack(frames, axis=0)  # [T, H, W, 3]
            video = video.transpose(0, 3, 1, 2)  # [T, 3, H, W]
            video = torch.from_numpy(video).float()
            video = video.transpose(0, 1)  # [3, T, H, W]
            pred_video = video[
                :, self.condition_frame_length : self.condition_frame_length + self.chunk
            ]
            if pred_video.shape[1] < self.chunk:
                raise ValueError(f"Generated video has fewer frames than expected: {pred_video.shape[1]} vs chunk {self.chunk}")
                pred_video = video[:, -self.chunk :]

            reference_frame = self.reference_frames[env_idx]
            if reference_frame is None:
                raise NotImplementedError("没有锚定帧")
                reference_frame = self.current_obs[env_idx, :, 0, 0:1, :, :].detach().cpu()
                self.reference_frames[env_idx] = reference_frame
            recent_count = max(0, self.condition_frame_length - 1) # 除去reference frame后还需要的context帧数
            recent_start = max(0, pred_video.shape[1] - recent_count) # 从生成视频中取出最近的recent_count帧，如果生成的视频不足，就用reference frame填充
            recent_frames = [
                pred_video[:, t : t + 1] for t in range(recent_start, pred_video.shape[1])
            ]
            while len(recent_frames) < recent_count:
                raise ValueError(f"生成的视频帧数 {pred_video.shape[1]} 不足以提供 {recent_count} 帧的context")
                recent_frames.insert(0, reference_frame.clone())
            self.image_queue[env_idx] = [
                reference_frame.clone(),
                *recent_frames[-recent_count:],
            ][: self.condition_frame_length]

            all_samples.append(pred_video[:, -self.chunk :])
            self.current_action14[env_idx] = torch.from_numpy(action_seq[-1]).to(
                device=self.current_action14.device, dtype=self.current_action14.dtype
            )
            recent_actions = torch.from_numpy(
                action_seq[-max(1, self.condition_frame_length - 1) :]
            ).to(device=self.device, dtype=actions_tensor.dtype)
            if self.condition_frame_length > 1:
                self.condition_action[env_idx] = torch.cat(
                    [
                        self.reference_action14[env_idx : env_idx + 1].to(
                            device=self.device, dtype=actions_tensor.dtype
                        ),
                        recent_actions[-(self.condition_frame_length - 1) :],
                    ],
                    dim=0,
                )
            else:
                self.condition_action[env_idx] = self.current_action14[env_idx : env_idx + 1]

        # Stack all environments: [num_envs, C, T, H, W]
        x_samples = torch.stack(all_samples, dim=0).to(self.device)

        # Reshape to match current_obs format: [num_envs, C, 1, T, H, W]
        x_samples = x_samples.unsqueeze(2)

        # Update current observation: append new generated frames to the time dimension
        self.current_obs = torch.cat([self.current_obs, x_samples], dim=3)
        self.last_control_video = last_control_video
        if hasattr(self, "last_control_video_chunks"):
            self.last_control_video_chunks.append(last_control_video)

        # Keep only the last condition_frame_length + chunk frames
        # Note: chunk might be different from T in x_samples due to VAE decoding
        # We'll keep a sliding window of recent frames
        max_frames = self.condition_frame_length + self.chunk
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

    def _normalise_pipe_batch_output(self, output, expected_batch):
        if expected_batch == 1:
            if isinstance(output, list) and (not output or isinstance(output[0], Image.Image)):
                return [output]
            if isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
                return output
        if isinstance(output, list) and len(output) == expected_batch:
            if all(isinstance(item, list) for item in output):
                return output
        raise ValueError(
            f"Unexpected batched pipe output for batch={expected_batch}: "
            f"type={type(output)}, len={len(output) if isinstance(output, list) else 'NA'}"
        )

    def _pil_video_to_tensor(self, output_frames):
        frames = []
        for img in output_frames:
            arr = np.array(img) / 255.0
            arr = arr * 2.0 - 1.0
            frames.append(arr)
        video = np.stack(frames, axis=0)  # [T, H, W, 3]
        video = video.transpose(0, 3, 1, 2)  # [T, 3, H, W]
        video = torch.from_numpy(video).float()
        return video.transpose(0, 1)  # [3, T, H, W]

    def _infer_next_chunk_frames(self, actions):
        """Predict next frame chunk using batched Wan pipeline calls."""
        num_envs = self.num_envs
        assert actions.shape[0] == self.num_envs, (
            f"Actions shape {actions.shape} does not match num_envs {self.num_envs}"
        )

        actions_tensor = (
            torch.from_numpy(actions).to(self.device)
            if isinstance(actions, np.ndarray)
            else actions.to(self.device)
        ).float()
        if actions_tensor.ndim != 3 or actions_tensor.shape[-1] != 14:
            raise ValueError(f"Expected actions [num_envs, chunk, 14], got {actions_tensor.shape}")

        self.condition_action = self.condition_action.to(
            device=actions_tensor.device, dtype=actions_tensor.dtype
        )
        self.current_action14 = self.current_action14.to(
            device=actions_tensor.device, dtype=actions_tensor.dtype
        )

        if self.condition_frame_length > 1:
            action_window = torch.cat([self.condition_action, actions_tensor], dim=1)
        elif self.retain_action:
            action_window = torch.cat([self.condition_action, actions_tensor], dim=1)
        else:
            action_window = torch.cat(
                [self.current_action14[:, None, :], actions_tensor], dim=1
            )

        target_size = (self.view_width, self.view_height * len(self.camera_names))
        per_env_inputs = []
        last_control_video = [None] * num_envs

        for env_idx in range(num_envs):
            context_frames = self.image_queue[env_idx]
            if not context_frames or context_frames[0] is None:
                raise ValueError(f"Image queue for env {env_idx} is empty or uninitialized")

            input_video = []
            for context_frame in context_frames[-self.condition_frame_length :]:
                input_image = self._tensor_frame_to_pil(context_frame)
                if input_image.size != target_size:
                    input_image = input_image.resize(target_size, Image.BILINEAR)
                input_video.append(input_image)
            if len(input_video) < self.condition_frame_length:
                raise ValueError("context不足5")

            action_seq = action_window[env_idx].detach().cpu().numpy()
            action_seq = self._pad_action_sequence(action_seq)
            if self.is_robotwin_env:
                control_video, ray_map_o, ray_map_d = self._build_robotwin_controls(
                    env_idx, action_seq
                )
            else:
                control_video, ray_map_o, ray_map_d = self._build_arxx5_controls(
                    action_seq
                )
            pred_control_video = control_video[
                self.condition_frame_length : self.condition_frame_length + self.chunk
            ]
            last_control_video[env_idx] = pred_control_video
            per_env_inputs.append(
                {
                    "env_idx": env_idx,
                    "input_video": input_video,
                    "control_video": control_video,
                    "ray_map_o": ray_map_o,
                    "ray_map_d": ray_map_d,
                    "action_seq": action_seq,
                    "prompt": self.task_descriptions[env_idx] or self.prompt,
                }
            )

        all_samples = [None] * num_envs
        infer_batch_size = max(1, int(getattr(self, "wm_infer_batch_size", num_envs)))

        for batch_start in range(0, num_envs, infer_batch_size):
            batch_items = per_env_inputs[batch_start : batch_start + infer_batch_size]
            batch_size = len(batch_items)
            kwargs = {
                "prompt": [item["prompt"] for item in batch_items],
                "seed": [42] * batch_size,
                "tiled": self.tiled,
                "input_video": [item["input_video"] for item in batch_items],
                "vace_video": [item["control_video"] for item in batch_items],
                "ray_map_o": [item["ray_map_o"] for item in batch_items],
                "ray_map_d": [item["ray_map_d"] for item in batch_items],
                "height": target_size[1],
                "width": target_size[0],
                "num_frames": self.num_frames,
                "num_inference_steps": self.num_inference_steps,
                "cfg_scale": self.cfg_scale,
                "progress_bar_cmd": lambda x: x,
                "batch_size": batch_size,
            }

            output = self.pipe(**kwargs)
            output_batch = self._normalise_pipe_batch_output(output, batch_size)

            for item, output_frames in zip(batch_items, output_batch):
                env_idx = item["env_idx"]
                action_seq = item["action_seq"]
                video = self._pil_video_to_tensor(output_frames)
                pred_video = video[
                    :, self.condition_frame_length : self.condition_frame_length + self.chunk
                ]
                if pred_video.shape[1] < self.chunk:
                    raise ValueError(
                        f"Generated video has fewer frames than expected: "
                        f"{pred_video.shape[1]} vs chunk {self.chunk}"
                    )

                reference_frame = self.reference_frames[env_idx]
                if reference_frame is None:
                    raise NotImplementedError("没有锚定帧")
                recent_count = max(0, self.condition_frame_length - 1)
                recent_start = max(0, pred_video.shape[1] - recent_count)
                recent_frames = [
                    pred_video[:, t : t + 1]
                    for t in range(recent_start, pred_video.shape[1])
                ]
                while len(recent_frames) < recent_count:
                    raise ValueError(
                        f"生成的视频帧数 {pred_video.shape[1]} 不足以提供 "
                        f"{recent_count} 帧的context"
                    )
                self.image_queue[env_idx] = [
                    reference_frame.clone(),
                    *recent_frames[-recent_count:],
                ][: self.condition_frame_length]

                all_samples[env_idx] = pred_video[:, -self.chunk :]
                self.current_action14[env_idx] = torch.from_numpy(action_seq[-1]).to(
                    device=self.current_action14.device,
                    dtype=self.current_action14.dtype,
                )
                recent_actions = torch.from_numpy(
                    action_seq[-max(1, self.condition_frame_length - 1) :]
                ).to(device=self.device, dtype=actions_tensor.dtype)
                if self.condition_frame_length > 1:
                    self.condition_action[env_idx] = torch.cat(
                        [
                            self.reference_action14[env_idx : env_idx + 1].to(
                                device=self.device, dtype=actions_tensor.dtype
                            ),
                            recent_actions[-(self.condition_frame_length - 1) :],
                        ],
                        dim=0,
                    )
                else:
                    self.condition_action[env_idx] = self.current_action14[env_idx : env_idx + 1]

        if any(sample is None for sample in all_samples):
            raise RuntimeError("Batched WM inference did not produce samples for all envs.")

        x_samples = torch.stack(all_samples, dim=0).to(self.device)
        x_samples = x_samples.unsqueeze(2)
        self.current_obs = torch.cat([self.current_obs, x_samples], dim=3)
        self.last_control_video = last_control_video
        if hasattr(self, "last_control_video_chunks"):
            self.last_control_video_chunks.append(last_control_video)

        max_frames = self.condition_frame_length + self.chunk
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

    def _wrap_obs(self):
        """Wrap observation to match libero_env format"""
        num_envs = self.num_envs

        # Extract the last frame (most recent observation) for each environment
        # self.current_obs is [b, c, v, t, h, w]  v=1 for single view
        b, c, v, t, h, w = self.current_obs.shape
        assert b == num_envs, (
            f"Unexpected current_obs shape: {self.current_obs.shape}, expected {num_envs}"
        )

        last_frame = self.current_obs[
            :, :, 0, -1, :, :
        ]  # [b,3, v, t,h,w] -> [b, 3, 1, h, w] -> [b, 3, h, w]
        # [8, 3, 256, 256]

        full_image = last_frame.permute(0, 2, 3, 1)  # [b, H, W, 3]
        # Denormalize from [-1, 1] to [0, 255]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255)
        # print(f'full_image:{full_image.shape}')
        # print(f'image_size:{self.image_size}')
        # Resize to 256x256 to match libero_env format
        if full_image.shape[1:3] != self.image_size:
            # Reshape for interpolation: [num_envs, H, W, 3] -> [num_envs, 3, H, W]
            full_image = full_image.permute(0, 3, 1, 2)  # [num_envs, 3, H, W]
            # Resize using F.interpolate
            full_image = F.interpolate(
                full_image, size=self.image_size, mode="bilinear", align_corners=False
            )
            # Convert back: [num_envs, 3, 256, 256] -> [num_envs, 256, 256, 3]
            full_image = full_image.permute(0, 2, 3, 1)  # [num_envs, 256, 256, 3]

        # Convert to uint8 tensor (keep as tensor, not numpy)
        full_image = full_image.to(torch.uint8)

        expected_height = self.view_height * len(self.camera_names)
        if full_image.shape[1] != expected_height:
            raise ValueError(
                f"Unexpected multiview image height {full_image.shape[1]}, "
                f"expected {expected_height} for cameras {self.camera_names}"
            )
        view_images = {
            camera_name: full_image[
                :, view_idx * self.view_height : (view_idx + 1) * self.view_height
            ]
            for view_idx, camera_name in enumerate(self.camera_names)
        }
        main_image = view_images.get("head", full_image[:, : self.view_height])
        left_wrist_image = view_images.get("left_wrist", torch.zeros_like(main_image))
        right_wrist_image = view_images.get("right_wrist", torch.zeros_like(main_image))

        def _prepare_image_for_vla(image, view_name):
            # image: [num_envs, H, W, 3], uint8
            if self.is_robotwin_env:
                expected_hw = (256, 320)
                target_hw = (240, 320)
                env_name = "robotwin"
            else:
                expected_hw = (224, 224)
                target_hw = (224, 224)
                env_name = "arxx5"
            actual_hw = tuple(int(v) for v in image.shape[1:3])
            if actual_hw != expected_hw:
                raise ValueError(
                    f"{env_name} VLA observation view {view_name!r} expects "
                    f"{expected_hw}, got {actual_hw}. Check image_size/view_height/view_width."
                )
            if target_hw == expected_hw:
                return image.contiguous()
            image = image.permute(0, 3, 1, 2).float()
            image = F.interpolate(
                image,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            image = image.permute(0, 2, 3, 1)
            return image.round().clamp(0, 255).to(torch.uint8).contiguous()

        main_image = _prepare_image_for_vla(main_image, "main")
        left_wrist_image = _prepare_image_for_vla(left_wrist_image, "left_wrist")
        right_wrist_image = _prepare_image_for_vla(right_wrist_image, "right_wrist")

        wrist_images = torch.stack([left_wrist_image, right_wrist_image], dim=1)

        # # ARXX5 PI05 was trained without proprio/state input.
        # states = torch.zeros((num_envs, 0), device=self.device, dtype=torch.float32)
        # Now using states
        states = self.current_action14.to(
            device=self.device,
            dtype=torch.float32,
        ).clone()
        # Get task descriptions
        if hasattr(self, "task_descriptions"):
            task_descriptions = self.task_descriptions
        else:
            task_descriptions = ["use the left arm to put the orange square into the green box and use the right arm to put the blue square into the green box"] * num_envs

        # Wrap observation - format aligned with libero_env
        obs = {
            "main_images": main_image,  # [num_envs, H, W, 3]
            "wrist_images": wrist_images,  # [num_envs, 2, H, W, 3]
            "extra_view_images": None,
            "states": states,  # [num_envs, 14]
            "task_descriptions": task_descriptions,  # list of strings
        }

        return obs

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """Handle automatic reset on episode termination"""
        final_obs = extracted_obs
        final_info = infos

        extracted_obs, infos = self.reset()

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones

        return extracted_obs, infos

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """Execute a chunk of actions - optimized version that processes chunk actions together"""
        # chunk_actions: [num_envs, chunk_steps, action_dim=14]
        self.onload()

        actions_tensor = (
            torch.from_numpy(policy_output_action).to(self.device)
            if isinstance(policy_output_action, np.ndarray)
            else policy_output_action.to(self.device)
        ).float()
        # [num_envs, chunk_steps, 14]
        # chunk_steps可以是8的倍数
        if actions_tensor.ndim != 3 or actions_tensor.shape[-1] != 14:
            raise ValueError(
                f"Expected actions [num_envs, chunk, 14], got {actions_tensor.shape}"
            )
        if actions_tensor.shape[0] != self.num_envs:
            raise ValueError(
                f"Actions shape {actions_tensor.shape} does not match num_envs {self.num_envs}"
            )
        action_chunk_len = actions_tensor.shape[1]
        if action_chunk_len % self.chunk != 0:
            raise ValueError(
                f"Action chunk length {action_chunk_len} must be a multiple of "
                f"world-model chunk {self.chunk}"
            )

        reward_chunks = []
        termination_chunks = []
        truncation_chunks = []
        self.last_control_video_chunks = []
        extracted_obs = None
        num_rollout_chunks = action_chunk_len // self.chunk

        for chunk_idx in range(num_rollout_chunks):
            start = chunk_idx * self.chunk
            end = start + self.chunk
            sub_actions = actions_tensor[:, start:end, :]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                self._infer_next_chunk_frames(sub_actions)

            # Update elapsed steps after each world-model autoregressive rollout.
            self.elapsed_steps += self.chunk

            # Read the last frame from self.current_obs.
            extracted_obs = self._wrap_obs()

            # Get rewards for this generated world-model chunk.
            chunk_rewards = self._infer_next_chunk_rewards()
            # [B,T]
            chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)
            reward_chunks.append(chunk_rewards_tensors)

            # Simple strategy: mark success/truncation at the end of each generated sub-chunk,
            # but do not stop early inside a longer policy action chunk.
            estimated_success = self._estimate_success_from_rewards(chunk_rewards)
            raw_chunk_terminations = torch.zeros(
                self.num_envs, self.chunk, dtype=torch.bool, device=self.device
            )
            raw_chunk_terminations[:, -1] = estimated_success
            # 仅在每个sub-chunk的最后一步标记成功/终止，其他步骤不标记，即使reward已经超过阈值也继续执行后续动作，直到sub-chunk结束
            raw_chunk_truncations = torch.zeros(
                self.num_envs, self.chunk, dtype=torch.bool, device=self.device
            )
            truncations = torch.tensor(
                self.elapsed_steps >= self.cfg.max_episode_steps, device=self.device
            )
            if truncations.any():
                raw_chunk_truncations[:, -1] = truncations # 同样仅在sub-chunk的最后一步标记截断

            termination_chunks.append(raw_chunk_terminations)
            truncation_chunks.append(raw_chunk_truncations)

        chunk_rewards_tensors = torch.cat(reward_chunks, dim=1)
        raw_chunk_terminations = torch.cat(termination_chunks, dim=1)
        raw_chunk_truncations = torch.cat(truncation_chunks, dim=1)
        # 把每个sub-chunk的奖励、终止、截断结果拼接起来，得到完整的chunk级别的奖励和done信号
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
        # 得到每个环境在整个chunk动作执行完后是否已经终止或截断的结果

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}

        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, infos
        )

        chunk_terminations = raw_chunk_terminations
        chunk_truncations = raw_chunk_truncations

        # Get actions and rewards for rendering
        chunk_actions_for_render = actions_tensor.detach().cpu().numpy()
        chunk_rewards_for_render = chunk_rewards_tensors.detach().cpu().numpy()

        # Reshape for rendering: [num_envs, chunk, action_dim] -> [chunk, num_envs, action_dim]
        chunk_actions_for_render = chunk_actions_for_render.transpose(1, 0, 2)
        chunk_rewards_for_render = chunk_rewards_for_render.T  # [chunk, num_envs]

        return (
            [extracted_obs],
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    def offload(self):
        """Move heavy models and runtime tensors to CPU."""
        if self._is_offloaded:
            return
        self.pipe.vae = self.pipe.vae.to("cpu")
        self.pipe.dit = self.pipe.dit.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.current_action14 = self.current_action14.cpu()
        self.condition_action = self.condition_action.cpu()
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        """Move models and runtime tensors back to execution device."""
        if not self._is_offloaded:
            return
        self.pipe.dit = self.pipe.dit.to(self.device)
        self.pipe.vae = self.pipe.vae.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.current_action14 = self.current_action14.to(self.device)
        self.condition_action = self.condition_action.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False

    def get_state(self) -> bytes:
        """Serialize runtime state to CPU bytes buffer for offload."""
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "current_action14": self.current_action14.cpu(),
            "condition_action": self.condition_action.cpu(),
            "elapsed_steps": self.elapsed_steps,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "_is_start": self._is_start,
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
        }
        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )

        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()


# export PYTHONPATH=/data/zsq/diffsynth-studio-wujie-ppu:/data/zsq/RLinf:$PYTHONPATH python -m rlinf.envs.world_model.world_model_wan_multi_view_env
if __name__ == "__main__":
    import json
    from pathlib import Path

    import cv2
    import imageio.v2 as imageio
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir
    from omegaconf import open_dict

    # # Set required environment variable
    os.environ.setdefault("EMBODIED_PATH", "examples/embodiment")

    repo_root = Path(__file__).resolve().parents[3]

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    config_dir = Path(
        os.environ.get("EMBODIED_CONFIG_DIR", repo_root / "examples/embodiment/config")
    ).resolve()
    config_name = os.environ.get(
        "EMBODIED_CONFIG_NAME", "wanmultiview_robotwin_grpo_pi05"
    )

    print(f"Loading config: {config_name} from {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    rollout_episode_index = int(os.environ.get("WM_ROLLOUT_EPISODE_INDEX", "5"))
    rollout_dataset_dir = Path(
        os.environ.get("WM_ROLLOUT_DATASET_DIR", cfg.initial_image_path)
    )
    rollout_output_dir = Path(
        os.environ.get(
            "WM_ROLLOUT_OUTPUT_DIR",
            "/data/zsq/RLinf/outputs/wanmultiview_rollout",
        )
    )
    rollout_fps = int(os.environ.get("WM_ROLLOUT_FPS", "30"))

    if not rollout_dataset_dir.exists():
        raise FileNotFoundError(
            f"Rollout dataset not found: {rollout_dataset_dir}. "
            "Build it first with examples/embodiment/prepare_wanmultiview_rollout_dataset.py"
        )

    with open_dict(cfg):
        cfg.initial_image_path = str(rollout_dataset_dir)
        cfg.total_num_envs = 1
        cfg.group_size = 1
        cfg.auto_reset = False
        cfg.use_fixed_reset_state_ids = False

    env = WanmultiviewEnv(cfg, cfg.total_num_envs, seed_offset=0, total_num_processes=1)

    if rollout_episode_index < 0 or rollout_episode_index >= len(env.dataset):
        raise ValueError(
            f"WM_ROLLOUT_EPISODE_INDEX={rollout_episode_index} is outside "
            f"dataset range [0, {len(env.dataset) - 1}]"
        )

    trajectory = np.load(env.dataset.npy_files[rollout_episode_index], allow_pickle=True)
    actions = np.asarray(
        [
            frame.get("action", frame.get("state"))
            for frame in trajectory
        ],
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected episode actions [T, 14], got {actions.shape}")

    # Frame 0 is the reset condition image/state. Roll out frames 1..T-1.
    rollout_actions = actions[1:] # 除了第一帧的reset condition，其他都rollout
    max_steps_env = os.environ.get("WM_ROLLOUT_MAX_STEPS", None)
    if max_steps_env is not None:
        rollout_actions = rollout_actions[: int(max_steps_env)]

    with open_dict(cfg):
        cfg.max_episode_steps = int(len(rollout_actions))

    obs, info = env.reset(episode_indices=np.asarray([rollout_episode_index]))
    del obs, info

    def tensor_frames_to_uint8(frames_tensor):
        frames_tensor = frames_tensor.detach().float().cpu()
        frames_tensor = frames_tensor.permute(1, 2, 3, 0)  # [T, H, W, C]
        frames_tensor = (frames_tensor + 1.0) / 2.0 * 255.0
        frames_tensor = torch.clamp(frames_tensor, 0, 255)
        return frames_tensor.numpy().astype(np.uint8)

    def annotate_frame(frame, frame_idx, reward_value):
        annotated = np.ascontiguousarray(frame.copy())
        text = f"frame={frame_idx:04d} reward={reward_value:.3f}"
        color = (0, 255, 0) if reward_value >= 0.5 else (255, 80, 80)
        cv2.rectangle(annotated, (6, 6), (360, 38), (0, 0, 0), thickness=-1)
        cv2.putText(
            annotated,
            text,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        return annotated

    def control_frame_to_uint8(control_frame, target_shape):
        if control_frame is None:
            return np.zeros(target_shape, dtype=np.uint8)
        if isinstance(control_frame, Image.Image):
            control_img = control_frame.convert("RGB")
        else:
            control_img = Image.fromarray(np.asarray(control_frame, dtype=np.uint8)).convert("RGB")
        target_h, target_w = target_shape[:2]
        if control_img.size != (target_w, target_h):
            control_img = control_img.resize((target_w, target_h), Image.BILINEAR)
        return np.asarray(control_img, dtype=np.uint8)

    def append_traj_map(frame, traj_frame):
        traj = control_frame_to_uint8(traj_frame, frame.shape)
        return np.concatenate([frame, traj], axis=1)

    def build_initial_traj_frame():
        action_seq = env.current_action14[0].detach().float().cpu().numpy()[None, :]
        action_seq = np.repeat(action_seq, env.condition_frame_length + env.chunk, axis=0)
        action_seq = env._pad_action_sequence(action_seq)
        if env.is_robotwin_env:
            control_video, _, _ = env._build_robotwin_controls(0, action_seq)
        else:
            control_video, _, _ = env._build_arxx5_controls(action_seq)
        context_idx = min(max(env.condition_frame_length - 1, 0), len(control_video) - 1)
        return control_video[context_idx]

    generated_frames = [
        tensor_frames_to_uint8(env.current_obs[0, :, 0, -1:, :, :])[0] # [8, 3, 1, 5, 256, 256]
    ]
    traj_frames = [build_initial_traj_frame()]
    rewards = [0.0]

    num_chunks = int(np.ceil(len(rollout_actions) / cfg.chunk))
    print(
        f"Rolling out episode={rollout_episode_index}, "
        f"steps={len(rollout_actions)}, chunks={num_chunks}, dataset={rollout_dataset_dir}"
    )

    for chunk_idx in range(num_chunks):
        start = chunk_idx * cfg.chunk #  0, 8, 16, ...
        end = min(start + cfg.chunk, len(rollout_actions)) # 8, 16, ..., len(rollout_actions)
        chunk = rollout_actions[start:end]
        valid_len = len(chunk)
        if valid_len < cfg.chunk:
            pad = np.repeat(chunk[-1:], cfg.chunk - valid_len, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0) # [8, 14]

        _, chunk_rewards, _, _, _ = env.chunk_step(chunk[None, :, :]) # 输入[1, chunk, 14] 输出[1, chunk]
        chunk_reward_np = chunk_rewards[0, :valid_len].detach().cpu().numpy()
        chunk_frames = tensor_frames_to_uint8(
            env.current_obs[0, :, 0, -cfg.chunk:, :, :]
        )[:valid_len]
        chunk_traj_frames = env.last_control_video[0][:valid_len]
        generated_frames.extend(list(chunk_frames))
        traj_frames.extend(list(chunk_traj_frames))
        rewards.extend([float(x) for x in chunk_reward_np])
        print(
            f"chunk {chunk_idx + 1}/{num_chunks}: "
            f"frames={valid_len}, rewards={chunk_reward_np.tolist()}"
        )

    rollout_output_dir.mkdir(parents=True, exist_ok=True)
    annotated_frames = [
        append_traj_map(annotate_frame(frame, idx, reward), traj_frame)
        for idx, (frame, traj_frame, reward) in enumerate(
            zip(generated_frames, traj_frames, rewards)
        )
    ]
    video_path = (
        rollout_output_dir
        / f"episode_{rollout_episode_index:06d}_wm_rollout_reward.mp4"
    )
    rewards_path = (
        rollout_output_dir
        / f"episode_{rollout_episode_index:06d}_wm_rollout_rewards.json"
    )
    imageio.mimsave(video_path, annotated_frames, fps=rollout_fps, macro_block_size=1)
    rewards_path.write_text(
        json.dumps(
            {
                "episode_index": rollout_episode_index,
                "dataset": str(rollout_dataset_dir),
                "video": str(video_path),
                "fps": rollout_fps,
                "rewards": rewards,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved rollout video: {video_path}")
    print(f"Saved rewards: {rewards_path}")
