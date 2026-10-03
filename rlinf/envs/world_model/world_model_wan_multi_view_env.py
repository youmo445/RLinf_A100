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
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
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
from rlinf.envs.world_model.arxx5_action_map import generate_traj_map_alpha
from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.envs.world_model.reward_model_compat import (
    ResnetRewModel as CompatibleResnetRewModel,
)
from rlinf.envs.world_model.robotwin_camera_fk import (
    RobotWinCameraFKProjector,
    make_abs_actions_from_fk_wxyz,
    scale_intrinsic as robotwin_scale_intrinsic,
    to_robotwin_camera_name,
)

__all__ = ["WanmultiviewEnv"]


class InstructionRoutedResnetRewModel(torch.nn.Module):
    """Route each frame to a task-specific binary reward model by instruction."""

    def __init__(self, checkpoint_root, instruction_to_checkpoint, threshold=None):
        super().__init__()
        checkpoint_root = Path(checkpoint_root)
        instruction_to_checkpoint = dict(instruction_to_checkpoint)
        if not instruction_to_checkpoint:
            raise ValueError(
                "InstructionRoutedResnetRewModel requires "
                "reward_model.instruction_to_checkpoint"
            )

        self.models = torch.nn.ModuleDict()
        self.instruction_to_model_key = {}
        for model_idx, (instruction, checkpoint_value) in enumerate(
            instruction_to_checkpoint.items()
        ):
            normalized_instruction = str(instruction).strip()
            if not normalized_instruction:
                raise ValueError("Reward-model instruction cannot be empty")
            checkpoint_path = Path(str(checkpoint_value))
            if not checkpoint_path.is_absolute():
                checkpoint_path = checkpoint_root / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)

            model_key = f"task_{model_idx}"
            self.models[model_key] = ResnetRewModel(
                checkpoint_path=str(checkpoint_path),
                threshold=threshold,
            )
            self.instruction_to_model_key[normalized_instruction] = model_key

    @torch.no_grad()
    def predict_rew(self, obs, instructions=None):
        if instructions is None or len(instructions) != obs.shape[0]:
            raise ValueError(
                "Instruction-routed reward inference expects one instruction per frame: "
                f"frames={obs.shape[0]}, instructions="
                f"{None if instructions is None else len(instructions)}"
            )

        normalized = [str(item).strip() for item in instructions]
        unknown = sorted(
            set(normalized).difference(self.instruction_to_model_key)
        )
        if unknown:
            raise ValueError(
                f"No reward model configured for instructions {unknown}. "
                f"Known instructions: {sorted(self.instruction_to_model_key)}"
            )

        rewards = torch.empty(
            obs.shape[0], device=obs.device, dtype=torch.float32
        )
        grouped_indices = {}
        for sample_idx, instruction in enumerate(normalized):
            model_key = self.instruction_to_model_key[instruction]
            grouped_indices.setdefault(model_key, []).append(sample_idx)

        for model_key, sample_indices in grouped_indices.items():
            indices = torch.as_tensor(
                sample_indices, device=obs.device, dtype=torch.long
            )
            task_rewards = self.models[model_key].predict_rew(
                obs.index_select(0, indices)
            )
            rewards.index_copy_(0, indices, task_rewards.reshape(-1))
        return rewards


class EpisodeGroupRoutedResnetRewModel(torch.nn.Module):
    """Route each frame to the reward model assigned to its reset episode group."""

    def __init__(self, checkpoint_root, group_to_checkpoint, threshold=None):
        super().__init__()
        checkpoint_root = Path(checkpoint_root)
        group_to_checkpoint = dict(group_to_checkpoint)
        if not group_to_checkpoint:
            raise ValueError(
                "EpisodeGroupRoutedResnetRewModel requires "
                "reward_model.group_to_checkpoint"
            )

        self.models = torch.nn.ModuleDict()
        self.group_to_model_key = {}
        for model_idx, (group_id, checkpoint_value) in enumerate(
            group_to_checkpoint.items()
        ):
            normalized_group_id = int(group_id)
            checkpoint_path = Path(str(checkpoint_value))
            if not checkpoint_path.is_absolute():
                checkpoint_path = checkpoint_root / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)

            model_key = f"group_{model_idx}"
            self.models[model_key] = ResnetRewModel(
                checkpoint_path=str(checkpoint_path),
                threshold=threshold,
            )
            self.group_to_model_key[normalized_group_id] = model_key

    @property
    def configured_group_ids(self):
        return tuple(sorted(self.group_to_model_key))

    @torch.no_grad()
    def predict_rew(self, obs, group_ids=None):
        if group_ids is None or len(group_ids) != obs.shape[0]:
            raise ValueError(
                "Episode-group reward inference expects one group ID per frame: "
                f"frames={obs.shape[0]}, group_ids="
                f"{None if group_ids is None else len(group_ids)}"
            )

        normalized_group_ids = [int(group_id) for group_id in group_ids]
        unknown = sorted(
            set(normalized_group_ids).difference(self.group_to_model_key)
        )
        if unknown:
            raise ValueError(
                f"No reward model configured for episode groups {unknown}. "
                f"Known groups: {list(self.configured_group_ids)}"
            )

        grouped_indices = {}
        for sample_idx, group_id in enumerate(normalized_group_ids):
            model_key = self.group_to_model_key[group_id]
            grouped_indices.setdefault(model_key, []).append(sample_idx)

        rewards = torch.empty(
            obs.shape[0], device=obs.device, dtype=torch.float32
        )
        for model_key, sample_indices in grouped_indices.items():
            indices = torch.as_tensor(
                sample_indices, device=obs.device, dtype=torch.long
            )
            group_rewards = self.models[model_key].predict_rew(
                obs.index_select(0, indices)
            )
            rewards.index_copy_(0, indices, group_rewards.reshape(-1))
        return rewards


class InstructionRoutedGripperTintResnetRewModel(torch.nn.Module):
    """Route tinted head-view frames to task-specific reward models."""

    def __init__(self, checkpoint_root, task_routes, threshold=None):
        super().__init__()
        checkpoint_root = Path(checkpoint_root)
        task_routes = dict(task_routes)
        if not task_routes:
            raise ValueError(
                "InstructionRoutedGripperTintResnetRewModel requires "
                "reward_model.task_routes"
            )

        self.models = torch.nn.ModuleDict()
        self.routes = []
        for model_idx, (route_name, route_value) in enumerate(task_routes.items()):
            route_cfg = dict(route_value)
            checkpoint_value = route_cfg.get("checkpoint")
            if checkpoint_value is None:
                raise ValueError(
                    f"Reward-model route {route_name!r} is missing checkpoint"
                )
            checkpoint_path = Path(str(checkpoint_value))
            if not checkpoint_path.is_absolute():
                checkpoint_path = checkpoint_root / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)

            keywords = route_cfg.get("keywords", [route_name])
            if isinstance(keywords, str):
                keywords = [keywords]
            keywords = tuple(
                str(keyword).strip().lower()
                for keyword in keywords
                if str(keyword).strip()
            )
            if not keywords:
                raise ValueError(
                    f"Reward-model route {route_name!r} has no non-empty keywords"
                )

            model_key = f"task_{model_idx}"
            route_threshold = route_cfg.get("threshold", threshold)
            self.models[model_key] = CompatibleResnetRewModel(
                checkpoint_path=str(checkpoint_path),
                threshold=route_threshold,
            )
            self.routes.append((str(route_name), model_key, keywords))

    def _model_key_for_instruction(self, instruction):
        normalized = str(instruction).strip().lower()
        matches = [
            (route_name, model_key)
            for route_name, model_key, keywords in self.routes
            if any(keyword in normalized for keyword in keywords)
        ]
        if len(matches) != 1:
            route_summary = {
                route_name: list(keywords)
                for route_name, _, keywords in self.routes
            }
            raise ValueError(
                "Expected exactly one gripper-tint reward route for instruction "
                f"{instruction!r}, got {[name for name, _ in matches]}. "
                f"Configured routes: {route_summary}"
            )
        return matches[0][1]

    @torch.no_grad()
    def predict_rew(self, obs, instructions=None):
        if instructions is None or len(instructions) != obs.shape[0]:
            raise ValueError(
                "Instruction-routed gripper-tint reward inference expects one "
                f"instruction per frame: frames={obs.shape[0]}, instructions="
                f"{None if instructions is None else len(instructions)}"
            )

        grouped_indices = {}
        for sample_idx, instruction in enumerate(instructions):
            model_key = self._model_key_for_instruction(instruction)
            grouped_indices.setdefault(model_key, []).append(sample_idx)

        rewards = torch.empty(
            obs.shape[0], device=obs.device, dtype=torch.float32
        )
        for model_key, sample_indices in grouped_indices.items():
            indices = torch.as_tensor(
                sample_indices, device=obs.device, dtype=torch.long
            )
            task_rewards = self.models[model_key].predict_rew(
                obs.index_select(0, indices)
            )
            rewards.index_copy_(0, indices, task_rewards.reshape(-1))
        return rewards


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
        if len(input_video) < 5:
            raise ValueError(
                f"TI2V2 context inference expects 5 context frames, got {len(input_video)}."
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

        self.reward_group_sampling = str(
            cfg.get("reward_group_sampling", "episode_uniform")
        ).strip().lower()
        if self.reward_group_sampling not in {"episode_uniform", "balanced"}:
            raise ValueError(
                "reward_group_sampling must be 'episode_uniform' or 'balanced', "
                f"got {self.reward_group_sampling!r}"
            )
        self._reward_group_sampling_round = 0
        self.episode_reward_group_ids = None
        self.episode_indices_by_reward_group = {}
        self.reward_group_manifest_path = None
        self._init_episode_reward_groups()
        self.current_reward_group_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # Logging identity uses the episode actually selected by reset(), not
        # the pre-sampled reset_state_ids that a reset may choose not to use.
        self.current_reset_history_ids = None

        # Update reset state ids
        self.update_reset_state_ids()

        # Model hyperparameters
        self.num_inference_steps = cfg.num_inference_steps
        self.chunk = int(
            cfg.get("wm_chunk", cfg.get("chunk", cfg.get("horizon_frames", 8)))
        )  # Ta = 8
        self.condition_frame_length = int(
            cfg.get(
                "context_frames",
                cfg.get("condition_frame_length", cfg.get("dataset_context_frames", 5)),
            )
        )  # To = 5
        self.num_frames = int(
            cfg.get(
                "wm_num_frames",
                cfg.get("num_frames", self.condition_frame_length + self.chunk),
            )
        )  # Total number of frames to encode = 13
        assert self.num_frames == self.condition_frame_length + self.chunk, (
            "num_frames must be equal to condition_frame_length + action_chunk_length"
        )
        self.rollout_context_mode = self._normalize_rollout_context_mode(
            cfg.get("rollout_context_mode", "dense_history")
        )
        self.rollout_sparse_history_stride = int(
            cfg.get("rollout_sparse_history_stride", cfg.get("sparse_history_stride", 4))
        )
        if self.rollout_sparse_history_stride <= 0:
            raise ValueError(
                f"rollout_sparse_history_stride must be positive, got {self.rollout_sparse_history_stride}"
            )
        if (
            self.rollout_context_mode == "sparse_history"
            and self.condition_frame_length != 5
        ):
            raise ValueError("sparse_history rollout expects condition_frame_length=5")

        task_rollout_cfg = cfg.get("task_specific_rollout", {})
        self.task_specific_rollout_enabled = bool(
            task_rollout_cfg.get("enabled", False)
        )
        self.task_specific_stop_on_success = bool(
            task_rollout_cfg.get("stop_on_success", False)
        )
        self.task_horizon_by_instruction = {
            str(instruction).strip(): int(max_steps)
            for instruction, max_steps in dict(
                task_rollout_cfg.get("horizon_by_instruction", {})
            ).items()
        }
        self.task_horizon_by_reward_group = {
            int(group_id): int(max_steps)
            for group_id, max_steps in dict(
                task_rollout_cfg.get("horizon_by_reward_group", {})
            ).items()
        }
        configured_horizons = {
            **{
                f"instruction:{instruction}": max_steps
                for instruction, max_steps in self.task_horizon_by_instruction.items()
            },
            **{
                f"reward_group:{group_id}": max_steps
                for group_id, max_steps in self.task_horizon_by_reward_group.items()
            },
        }
        if self.task_specific_rollout_enabled and not configured_horizons:
            raise ValueError(
                "task_specific_rollout.enabled=true requires either "
                "horizon_by_instruction or horizon_by_reward_group"
            )
        if self.task_specific_rollout_enabled and self.auto_reset:
            raise ValueError(
                "task-specific padded rollout requires auto_reset=false so completed "
                "environments remain inactive until the next rollout reset"
            )
        invalid_horizons = {
            key: max_steps
            for key, max_steps in configured_horizons.items()
            if max_steps <= 0 or max_steps % self.chunk != 0
        }
        if invalid_horizons:
            raise ValueError(
                "Task-specific horizons must be positive multiples of the WM chunk "
                f"({self.chunk}), got {invalid_horizons}"
            )
        if (
            self.task_specific_rollout_enabled
            and max(configured_horizons.values()) > int(cfg.max_episode_steps)
        ):
            raise ValueError(
                "env.max_episode_steps must cover the longest task-specific horizon: "
                f"global={cfg.max_episode_steps}, "
                f"task_max={max(configured_horizons.values())}"
            )

        self.image_size = tuple(cfg.image_size)

        #
        self.retain_action = cfg.get("retain_action", True)  # Default True TODO
        self.latch_success_reward = bool(cfg.get("latch_success_reward", False))
        self.reward_success_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.enable_kir = cfg.get("enable_kir", True) # TODO
        self.kir_initial_frame = cfg.get("kir_initial_frame", None)
        if self.kir_initial_frame is not None:
            self.kir_initial_frame = int(self.kir_initial_frame)
        self.initial_joint_source = str(
            cfg.get("initial_joint_source", "action")
        ).lower()
        if self.initial_joint_source not in {"action", "state"}:
            raise ValueError(
                "initial_joint_source must be 'action' or 'state', got "
                f"{self.initial_joint_source!r}"
            )
        self.is_robotwin_env = cfg.get("wm_env_type", "robotwin") == "robotwin"

        self.camera_names = list(
            cfg.get("camera_names", ["head", "left_wrist", "right_wrist"])
        )
        reward_model_cfg = cfg.get("reward_model", {})
        self.reward_model_input_view = str(
            reward_model_cfg.get("input_view", "all")
        ).strip().lower()
        if self.reward_model_input_view in ("multiview", "three_view", "three_views"):
            self.reward_model_input_view = "all"
        if (
            self.reward_model_input_view != "all"
            and self.reward_model_input_view not in self.camera_names
        ):
            raise ValueError(
                "reward_model.input_view must be 'all' or one of camera_names "
                f"{self.camera_names}, got {self.reward_model_input_view!r}"
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
        self.prompt = str(
            cfg.get("prompt", "机械臂按照要求移动夹爪执行任务")
        )
        self.wm_prompt_mode = str(cfg.get("wm_prompt_mode", "fixed"))
        if self.wm_prompt_mode not in {"fixed", "task_instruction"}:
            raise ValueError(
                f"Unsupported wm_prompt_mode={self.wm_prompt_mode}. "
                "Expected fixed or task_instruction."
            )
        self.traj_radius = int(cfg.get("traj_radius", 40))
        self.traj_radius_mode = cfg.get("traj_radius_mode", "perspective")
        self.traj_min_radius = int(cfg.get("traj_min_radius", 14))
        self.traj_max_radius = int(cfg.get("traj_max_radius", 58))
        self.traj_ref_depth = float(cfg.get("traj_ref_depth", 0.30))
        self.traj_near_depth = float(cfg.get("traj_near_depth", 0.19))
        self.traj_far_depth = float(cfg.get("traj_far_depth", 0.69))
        self.traj_map_renderer = str(cfg.get("traj_map_renderer", "opaque"))
        if self.traj_map_renderer not in {"opaque", "alpha"}:
            raise ValueError(
                f"Unsupported traj_map_renderer={self.traj_map_renderer}. "
                "Expected opaque or alpha."
            )
        self.traj_circle_alpha = int(cfg.get("traj_circle_alpha", 160))
        if not 0 <= self.traj_circle_alpha <= 255:
            raise ValueError(
                f"traj_circle_alpha must be in [0, 255], got {self.traj_circle_alpha}"
            )
        self.traj_local_x_offset = float(cfg.get("traj_local_x_offset", 0.2))
        self.traj_axis_line_width = int(cfg.get("traj_axis_line_width", 8))
        if self.traj_axis_line_width <= 0:
            raise ValueError(
                "traj_axis_line_width must be positive, got "
                f"{self.traj_axis_line_width}"
            )
        self.ray_o_vmin = float(cfg.get("ray_o_vmin", -1.5))
        self.ray_o_vmax = float(cfg.get("ray_o_vmax", 1.5))
        self.ray_d_vmin = float(cfg.get("ray_d_vmin", -1.0))
        self.ray_d_vmax = float(cfg.get("ray_d_vmax", 1.0))
        self.cfg_scale = float(cfg.get("cfg_scale", 1.0))
        self.tiled = bool(cfg.get("tiled", False))
        self.vace_in_dim = int(cfg.get("vace_in_dim", 144))
        default_raymap_mode = "raw_stack4" if self.vace_in_dim == 72 else "image"
        self.raymap_mode = cfg.get("raymap_mode", default_raymap_mode)
        if self.raymap_mode not in ("image", "raw_stack4"):
            raise ValueError(f"Unsupported raymap_mode: {self.raymap_mode}")
        if self.raymap_mode == "raw_stack4" and self.vace_in_dim != 72:
            raise ValueError(
                f"raymap_mode=raw_stack4 expects vace_in_dim=72, got {self.vace_in_dim}"
            )
        self.vae_spatial_downsample = int(cfg.get("vae_spatial_downsample", 16))
        self.vae_temporal_downsample = int(cfg.get("vae_temporal_downsample", 4))
        default_view_embedding_num_views = (
            len(self.camera_names) if self.vace_in_dim == 72 else 0
        )
        self.vace_view_embedding_num_views = int(
            cfg.get("vace_view_embedding_num_views", default_view_embedding_num_views)
        )

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
        self.active_env_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.env_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.env_episode_max_steps = torch.full(
            (self.num_envs,),
            int(self.cfg.max_episode_steps),
            dtype=torch.long,
            device=self.device,
        )

        # Image queue for condition frames to generate video,
        # keep length of condition_frame_length
        self.image_queue = [
            [None] * self.condition_frame_length for _ in range(self.num_envs)
        ]
        self.frame_history = [[] for _ in range(self.num_envs)]
        self.action_history14 = [[] for _ in range(self.num_envs)]

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

    def _normalize_rollout_context_mode(self, mode):
        mode = str(mode)
        aliases = {
            "contiguous": "dense_history",
            "fixed_sparse_history": "sparse_history",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"dense_history", "sparse_history"}:
            raise ValueError(
                f"Unsupported rollout_context_mode={mode}. "
                "Expected dense_history or sparse_history."
            )
        return mode

    def _reset_task_specific_rollout_state(self, task_descriptions):
        """Initialize per-environment horizons without changing the fixed batch shape."""
        self.active_env_mask.fill_(True)
        self.env_elapsed_steps.zero_()
        self.env_episode_max_steps.fill_(int(self.cfg.max_episode_steps))
        if not self.task_specific_rollout_enabled:
            return

        horizons = []
        missing = []
        for env_idx, task in enumerate(task_descriptions):
            normalized_task = str(task).strip()
            group_id = int(self.current_reward_group_ids[env_idx].item())
            if group_id in self.task_horizon_by_reward_group:
                horizons.append(self.task_horizon_by_reward_group[group_id])
            elif normalized_task in self.task_horizon_by_instruction:
                horizons.append(self.task_horizon_by_instruction[normalized_task])
            else:
                missing.append(
                    {
                        "env_idx": env_idx,
                        "instruction": normalized_task,
                        "reward_group_id": group_id,
                    }
                )

        if missing:
            raise ValueError(
                "No task-specific rollout horizon configured for environments "
                f"{missing}. Known instructions: "
                f"{sorted(self.task_horizon_by_instruction)}; known reward groups: "
                f"{sorted(self.task_horizon_by_reward_group)}"
            )

        self.env_episode_max_steps.copy_(
            torch.tensor(horizons, dtype=torch.long, device=self.device)
        )

    def _sparse_history_ids(self, env_idx):
        history_len = len(self.frame_history[env_idx])
        if history_len <= 0:
            raise ValueError(f"Sparse frame history for env {env_idx} is empty")
        last_idx = history_len - 1
        stride = self.rollout_sparse_history_stride
        return [
            0,
            max(0, last_idx - 3 * stride),
            max(0, last_idx - 2 * stride),
            max(0, last_idx - stride),
            last_idx,
        ]

    def _sparse_history_context(self, env_idx):
        frame_history = self.frame_history[env_idx]
        action_history = self.action_history14[env_idx]
        if not action_history:
            reference_action = self.reference_action14[env_idx].detach().cpu().numpy()
            action_history = [reference_action.astype(np.float32)]
            self.action_history14[env_idx] = action_history

        ids = self._sparse_history_ids(env_idx)
        context_frames = [
            frame_history[min(idx, len(frame_history) - 1)].clone() for idx in ids
        ]
        context_actions = [
            np.asarray(action_history[min(idx, len(action_history) - 1)], dtype=np.float32).reshape(14)
            for idx in ids
        ]
        return context_frames, np.stack(context_actions, axis=0)

    def _append_sparse_history(self, env_idx, pred_video, actions_tensor):
        if not self.frame_history[env_idx]:
            reference_frame = self.reference_frames[env_idx]
            if reference_frame is None:
                raise ValueError(f"Missing reference frame for env {env_idx}")
            self.frame_history[env_idx] = [reference_frame.clone()]

        for t in range(pred_video.shape[1]):
            self.frame_history[env_idx].append(pred_video[:, t : t + 1].detach().clone())

        actions_np = actions_tensor.detach().cpu().numpy()
        for t in range(min(actions_np.shape[0], pred_video.shape[1])):
            self.action_history14[env_idx].append(
                np.asarray(actions_np[t], dtype=np.float32).reshape(14).copy()
            )

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path,
            action_key=cfg.get("dataset_action_key", "action"),
            state_key=cfg.get("dataset_state_key", "state"),
            enable_kir=self.enable_kir,
        )

    def _init_episode_reward_groups(self):
        reward_model_cfg = self.cfg.get("reward_model", {})
        reward_model_type = str(reward_model_cfg.get("type", ""))
        needs_group_metadata = (
            self.reward_group_sampling == "balanced"
            or reward_model_type == "EpisodeGroupRoutedResnetRewModel"
        )

        configured_manifest_path = self.cfg.get(
            "reward_group_manifest_path", None
        )
        manifest_path = (
            Path(str(configured_manifest_path))
            if configured_manifest_path
            else Path(str(self.cfg.initial_image_path)) / "manifest.json"
        )
        self.reward_group_manifest_path = manifest_path
        if not manifest_path.is_file():
            if needs_group_metadata:
                raise FileNotFoundError(
                    "Episode-group reward routing/sampling requires a manifest: "
                    f"{manifest_path}"
                )
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episode_mapping = manifest.get("episode_mapping", [])
        output_file_to_group = {}
        for item in episode_mapping:
            output_file = Path(str(item.get("output_file", ""))).name
            group_id = item.get("group_id", item.get("reward_group_id"))
            if not output_file or group_id is None:
                continue
            if output_file in output_file_to_group:
                raise ValueError(
                    f"Duplicate output_file {output_file!r} in {manifest_path}"
                )
            output_file_to_group[output_file] = int(group_id)

        if not output_file_to_group:
            if needs_group_metadata:
                raise ValueError(
                    f"No output_file/group_id mappings found in {manifest_path}"
                )
            return

        episode_group_ids = []
        missing_files = []
        for npy_file in self.dataset.npy_files:
            basename = Path(npy_file).name
            if basename not in output_file_to_group:
                missing_files.append(basename)
                continue
            episode_group_ids.append(output_file_to_group[basename])
        if missing_files:
            preview = missing_files[:8]
            raise ValueError(
                f"{manifest_path} is missing reward groups for {len(missing_files)} "
                f"dataset files, including {preview}"
            )

        self.episode_reward_group_ids = np.asarray(
            episode_group_ids, dtype=np.int64
        )
        for group_id in sorted(set(episode_group_ids)):
            indices = np.flatnonzero(
                self.episode_reward_group_ids == group_id
            ).astype(np.int64)
            self.episode_indices_by_reward_group[group_id] = torch.from_numpy(
                indices
            )

        if reward_model_type == "EpisodeGroupRoutedResnetRewModel":
            configured_groups = {
                int(group_id)
                for group_id in dict(
                    reward_model_cfg.get("group_to_checkpoint", {})
                )
            }
            dataset_groups = set(self.episode_indices_by_reward_group)
            missing_routes = sorted(dataset_groups.difference(configured_groups))
            if missing_routes:
                raise ValueError(
                    "reward_model.group_to_checkpoint is missing dataset groups "
                    f"{missing_routes}; dataset groups are {sorted(dataset_groups)}"
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

    def _decode_robotwin_rgb(self, raw):
        if isinstance(raw, np.bytes_):
            raw = raw.tobytes()
        elif isinstance(raw, np.ndarray):
            raw = raw.tobytes()
        elif not isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw)
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)

    def _read_robotwin_stacked_frame(self, source_path, frame_idx):
        import h5py

        views = []
        with h5py.File(source_path, "r") as f:
            for camera in self.robotwin_camera_names:
                key = f"observation/{camera}/rgb"
                if key not in f:
                    raise KeyError(f"Missing RobotWin camera key {key} in {source_path}")
                if frame_idx < 0 or frame_idx >= len(f[key]):
                    raise IndexError(
                        f"RobotWin frame {frame_idx} out of range for {key} "
                        f"with length {len(f[key])}: {source_path}"
                    )
                image = Image.fromarray(self._decode_robotwin_rgb(f[key][frame_idx]))
                target_size = (self.view_width, self.view_height)
                if image.size != target_size:
                    image = image.resize(target_size, Image.BILINEAR)
                views.append(np.asarray(image, dtype=np.uint8))
        stacked = np.concatenate(views, axis=0)
        return torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0

    def _build_kir_target_items(self, episode_data, episode_idx):
        if not self.enable_kir or self.kir_initial_frame is None:
            return []
        if self.condition_frame_length <= 1:
            return []

        dataset_meta = episode_data.get("dataset_meta", {})
        episode_length = int(dataset_meta.get("episode_length", 0))
        if episode_length <= 0:
            return []

        kir_frame = min(max(int(self.kir_initial_frame), 0), episode_length - 1)
        local_count = self.condition_frame_length - 1
        local_start = max(0, kir_frame - local_count + 1)
        frame_indices = list(range(local_start, kir_frame + 1))
        while len(frame_indices) < local_count:
            frame_indices.insert(0, frame_indices[0])

        file_path = dataset_meta.get("file_path", None)
        if file_path is None:
            raise ValueError("KIR needs dataset_meta['file_path'] to read actions.")
        trajectory = np.load(file_path, allow_pickle=True)

        source_path = None
        if self.is_robotwin_env:
            source_path = self._robotwin_source_path_for_episode(episode_idx)

        target_items = []
        for frame_idx in frame_indices[-local_count:]:
            raw_frame = trajectory[int(frame_idx)]
            if not isinstance(raw_frame, dict) and hasattr(raw_frame, "item"):
                raw_frame = raw_frame.item()
            joint_key = (
                "action" if self.initial_joint_source == "action" else "state"
            )
            if joint_key not in raw_frame:
                raise ValueError(
                    f"KIR frame {frame_idx} has no {joint_key} in {file_path}"
                )

            if self.is_robotwin_env:
                image_tensor = self._read_robotwin_stacked_frame(source_path, int(frame_idx))
            else:
                if "image" not in raw_frame:
                    raise ValueError(f"KIR frame {frame_idx} has no image in {file_path}")
                image = raw_frame["image"]
                if isinstance(image, np.ndarray):
                    if image.ndim == 3 and image.shape[2] == 3:
                        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                    else:
                        image_tensor = torch.from_numpy(image).float()
                else:
                    image_tensor = torch.as_tensor(image, dtype=torch.float32)

            action = raw_frame[joint_key]
            if isinstance(action, torch.Tensor):
                action_tensor = action.float()
            else:
                action_tensor = torch.as_tensor(action, dtype=torch.float32)

            target_items.append(
                {
                    "image": image_tensor,
                    "action": action_tensor,
                    "frame_index": int(frame_idx),
                }
            )
        return target_items

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

        pipe.vace = VaceWan22CodexModel(
            vace_in_dim=self.vace_in_dim,
            view_embedding_num_views=self.vace_view_embedding_num_views,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
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

    def _latent_frame_groups(self, num_frames):
        latent_t = (num_frames - 1) // self.vae_temporal_downsample + 1
        groups = [[0] * self.vae_temporal_downsample]
        for i in range(1, latent_t):
            start = 1 + (i - 1) * self.vae_temporal_downsample
            groups.append(
                [
                    min(start + j, num_frames - 1)
                    for j in range(self.vae_temporal_downsample)
                ]
            )
        return groups

    def _raw_stack4_latent_intrinsic(self, image_intrinsic, image_height, image_width):
        if (
            image_height % self.vae_spatial_downsample != 0
            or image_width % self.vae_spatial_downsample != 0
        ):
            raise ValueError(
                f"raw_stack4 requires view size divisible by {self.vae_spatial_downsample}, "
                f"got view_height={image_height}, view_width={image_width}."
            )
        latent_h = image_height // self.vae_spatial_downsample
        latent_w = image_width // self.vae_spatial_downsample
        latent_intrinsic = image_intrinsic.copy()
        latent_intrinsic[0, 0] *= latent_w / float(image_width)
        latent_intrinsic[0, 2] *= latent_w / float(image_width)
        latent_intrinsic[1, 1] *= latent_h / float(image_height)
        latent_intrinsic[1, 2] *= latent_h / float(image_height)
        return latent_intrinsic, latent_h, latent_w

    def _build_raw_stack4_raymap_tensors(self, ray_w2c_by_cam, intrinsic_by_cam, total_frames):
        groups = self._latent_frame_groups(total_frames)
        per_cam = {}
        for cam in self.camera_names:
            latent_intrinsic, latent_h, latent_w = self._raw_stack4_latent_intrinsic(
                intrinsic_by_cam[cam],
                self.view_height,
                self.view_width,
            )
            per_cam[cam] = {
                "latent_intrinsic": latent_intrinsic,
                "latent_h": latent_h,
                "latent_w": latent_w,
            }

        ray_o_slots = []
        ray_d_slots = []
        for group in groups:
            ray_o_views = []
            ray_d_views = []
            for cam in self.camera_names:
                cam_info = per_cam[cam]
                ray_o_group = []
                ray_d_group = []
                for frame_idx in group:
                    frame_idx = min(max(int(frame_idx), 0), total_frames - 1)
                    c2w = np.linalg.inv(ray_w2c_by_cam[cam][frame_idx]).astype(np.float32)
                    ray_o, ray_d = generate_raymap(
                        cam_info["latent_intrinsic"],
                        c2w,
                        cam_info["latent_h"],
                        cam_info["latent_w"],
                    )
                    ray_o_group.append(ray_o)
                    ray_d_group.append(ray_d)
                ray_o_views.append(np.concatenate(ray_o_group, axis=-1))
                ray_d_views.append(np.concatenate(ray_d_group, axis=-1))
            ray_o_slots.append(np.concatenate(ray_o_views, axis=0))
            ray_d_slots.append(np.concatenate(ray_d_views, axis=0))

        ray_o = torch.from_numpy(np.stack(ray_o_slots, axis=0)).permute(3, 0, 1, 2).float()
        ray_d = torch.from_numpy(np.stack(ray_d_slots, axis=0)).permute(3, 0, 1, 2).float()
        return ray_o, ray_d

    def _generate_traj_maps(self, abs_actions, w2c, intrinsic):
        kwargs = {
            "radius": self.traj_radius,
            "radius_mode": self.traj_radius_mode,
            "min_radius": self.traj_min_radius,
            "max_radius": self.traj_max_radius,
            "ref_depth": self.traj_ref_depth,
            "near_depth": self.traj_near_depth,
            "far_depth": self.traj_far_depth,
        }
        if self.traj_map_renderer == "alpha":
            return generate_traj_map_alpha(
                abs_actions,
                w2c,
                intrinsic,
                self.view_height,
                self.view_width,
                circle_alpha=self.traj_circle_alpha,
                local_x_offset=self.traj_local_x_offset,
                axis_line_width=self.traj_axis_line_width,
                **kwargs,
            )
        return generate_traj_map(
            abs_actions,
            w2c,
            intrinsic,
            self.view_height,
            self.view_width,
            **kwargs,
        )

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
        raw_intrinsic_by_cam = {}
        for cam in self.camera_names:
            k = scale_intrinsic(
                self.intrinsics[cam],
                self.camera_original_height,
                self.camera_original_width,
                self.view_height,
                self.view_width,
            )
            raw_intrinsic_by_cam[cam] = k
            control_per_cam[cam] = self._generate_traj_maps(
                traj_actions_by_cam[cam],
                traj_w2c_by_cam[cam],
                k,
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
        if self.raymap_mode == "raw_stack4":
            ray_map_o, ray_map_d = self._build_raw_stack4_raymap_tensors(
                ray_w2c_by_cam,
                raw_intrinsic_by_cam,
                action_seq.shape[0],
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
        raw_intrinsic_by_cam = {}
        raw_w2c_by_cam = {}
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
            raw_intrinsic_by_cam[cam] = k
            raw_w2c_by_cam[cam] = w2c_seq
            control_per_cam[cam] = self._generate_traj_maps(
                abs_actions,
                w2c_seq,
                k,
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
        if self.raymap_mode == "raw_stack4":
            ray_map_o, ray_map_d = self._build_raw_stack4_raymap_tensors(
                raw_w2c_by_cam,
                raw_intrinsic_by_cam,
                action_seq.shape[0],
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
        elif self.cfg.reward_model.type == "InstructionRoutedResnetRewModel":
            rew_model = InstructionRoutedResnetRewModel(
                checkpoint_root=self.cfg.reward_model.from_pretrained,
                instruction_to_checkpoint=(
                    self.cfg.reward_model.instruction_to_checkpoint
                ),
                threshold=self.cfg.reward_model.get("threshold", None),
            )
        elif self.cfg.reward_model.type == "EpisodeGroupRoutedResnetRewModel":
            rew_model = EpisodeGroupRoutedResnetRewModel(
                checkpoint_root=self.cfg.reward_model.from_pretrained,
                group_to_checkpoint=self.cfg.reward_model.group_to_checkpoint,
                threshold=self.cfg.reward_model.get("threshold", None),
            )
        elif (
            self.cfg.reward_model.type
            == "InstructionRoutedGripperTintResnetRewModel"
        ):
            rew_model = InstructionRoutedGripperTintResnetRewModel(
                checkpoint_root=self.cfg.reward_model.from_pretrained,
                task_routes=self.cfg.reward_model.task_routes,
                threshold=self.cfg.reward_model.get("threshold", None),
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

    def _select_reward_model_view(self, obs):
        """Select one vertically stacked camera view for a single-view RM."""
        if self.reward_model_input_view == "all":
            return obs
        if obs.ndim != 4:
            raise ValueError(
                f"Expected reward observations [B, C, H, W], got {obs.shape}"
            )
        expected_height = self.view_height * len(self.camera_names)
        if obs.shape[-2] != expected_height:
            raise ValueError(
                f"Expected vertically stacked reward observation height "
                f"{expected_height}, got {obs.shape[-2]}"
            )
        view_idx = self.camera_names.index(self.reward_model_input_view)
        start = view_idx * self.view_height
        return obs[:, :, start : start + self.view_height, :]

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
            self.reward_success_latched[mask] = False
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
            # self._elapsed_steps = 0
        else:
            self.prev_step_reward[:] = 0
            self.reward_success_latched[:] = False
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        # step_reward是[B]
        # terminations是二值[B]
        episode_info = {}
        self.returns += step_reward # [num_envs] 累积奖励
        # 直接加上了每个VLA-chunk的总奖励
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
        if self.task_specific_rollout_enabled:
            episode_info["episode_len"] = self.env_elapsed_steps.to(
                dtype=torch.float32
            ).clone()
        else:
            episode_info["episode_len"] = torch.full(
                (self.num_envs,),
                self.elapsed_steps,
                dtype=torch.float32,
                device=self.device,
            )
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _apply_success_reward_latch(self, chunk_rewards):
        """Make success an absorbing reward state inside and across chunks."""
        if not self.latch_success_reward:
            return chunk_rewards

        success_threshold = float(getattr(self.cfg, "success_reward_threshold", 0.9))
        latched_rewards = torch.empty_like(chunk_rewards)
        for i in range(chunk_rewards.shape[1]):
            self.reward_success_latched = self.reward_success_latched | (
                chunk_rewards[:, i] >= success_threshold
            )
            latched_rewards[:, i] = self.reward_success_latched.to(
                dtype=chunk_rewards.dtype
            )
        return latched_rewards

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
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]  # [num_envs] 每个环境的chunk步对应的奖励在时间维度的最大值
        success_estimated = max_reward_in_chunk >= success_threshold # [num_envs] 每个环境的最大reward超过阈值则认为成功

        return success_estimated.to(self.device)

    def update_reset_state_ids(self):
        """Updates the reset state IDs for environment initialization."""
        if self.reward_group_sampling == "balanced":
            if not self.episode_indices_by_reward_group:
                raise ValueError(
                    "Balanced reward-group sampling requires episode group metadata"
                )
            group_ids = sorted(self.episode_indices_by_reward_group)
            worker_rank = int(
                getattr(self.worker_info, "rank", 0)
                if self.worker_info is not None
                else 0
            )
            sampled_episode_ids = []
            for local_group_idx in range(self.num_group):
                group_position = (
                    worker_rank * self.num_group
                    + local_group_idx
                    + self._reward_group_sampling_round
                ) % len(group_ids)
                group_id = group_ids[group_position]
                candidates = self.episode_indices_by_reward_group[group_id]
                candidate_idx = torch.randint(
                    low=0,
                    high=len(candidates),
                    size=(1,),
                    generator=self._generator,
                ).item()
                sampled_episode_ids.append(int(candidates[candidate_idx]))
            reset_state_ids = torch.tensor(
                sampled_episode_ids, dtype=torch.long
            )
            self._reward_group_sampling_round += 1
        else:
            total_num_episodes = len(self.dataset)
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
        episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        if episode_indices.shape[0] != num_envs:
            raise ValueError(
                f"Expected {num_envs} episode indices, got {episode_indices.shape[0]}"
            )
        self.current_reset_history_ids = torch.as_tensor(
            episode_indices, dtype=torch.long
        ).clone()
        if self.episode_reward_group_ids is not None:
            selected_group_ids = self.episode_reward_group_ids[episode_indices]
            self.current_reward_group_ids = torch.as_tensor(
                selected_group_ids, dtype=torch.long, device=self.device
            )
        else:
            self.current_reward_group_ids = torch.full(
                (num_envs,), -1, dtype=torch.long, device=self.device
            )

        # Load first frame from each selected episode
        img_tensors = []
        task_descriptions = []
        init_ee_poses = []
        init_states14 = []
        condition_actions = []
        kir_context_enabled = []

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

            initial_joint_key = (
                "action"
                if self.initial_joint_source == "action"
                else "observation.state"
            )
            init_state14 = first_frame.get(initial_joint_key)
            if init_state14 is None and self.initial_joint_source == "state":
                raise ValueError(
                    f"No '{initial_joint_key}' key in frame for episode "
                    f"{episode_idx}; initial_joint_source=state requires dataset state."
                )
            if (
                init_state14 is None
                and self.initial_joint_source == "action"
                and "observation.state" in first_frame
            ):
                raise NotImplementedError("用action表示初始状态")
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
            if init_state14 is not None:
                init_action14 = np.asarray(init_state14, dtype=np.float32).flatten()
                if init_action14.shape[0] != 14:
                    raise ValueError(f"Expected initial action/state dim 14, got {init_action14.shape}")
                env_condition_action[0] = init_action14

            if self.reset_gripper_open and self.is_libero_env:
                raise NotImplementedError("Gripper open action setting is not implemented for Libero env yet.")
                env_condition_action[:, -1] = -1

            # KIR trick: use the last four frames as condition frames, while
            # keeping the reference frame unchanged as the first frame.
            # target_items = episode_data.get("target_items", [])
            target_items = self._build_kir_target_items(episode_data, episode_idx)
            env_has_kir_context = len(target_items) == self.condition_frame_length - 1
            # first condition frame is the reference frame,
            # so the length of target_items should be condition_frame_length - 1
            if env_has_kir_context:
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
                    target_action = target_frame["action"]
                    if isinstance(target_action, torch.Tensor):
                        target_action = target_action.detach().cpu().numpy()
                    env_condition_action[target_idx + 1] = np.asarray(
                        target_action, dtype=np.float32
                    )
            elif self.enable_kir and self.kir_initial_frame is not None:
                raise ValueError(
                    f"KIR expected {self.condition_frame_length - 1} target frames "
                    f"for episode {episode_idx}, got {len(target_items)}"
                )

            img_tensors.append(env_img_tensor)
            condition_actions.append(torch.from_numpy(env_condition_action))
            kir_context_enabled.append(env_has_kir_context)

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
            if kir_context_enabled[env_idx]:
                current_action = cond_action
            elif np.any(cond_action):
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
        reference_actions14 = []
        for init_state14 in init_states14:
            if init_state14 is None:
                reference_action = np.zeros(14, dtype=np.float32)
            else:
                reference_action = np.asarray(init_state14, dtype=np.float32).flatten()
                if reference_action.shape[0] != 14:
                    raise ValueError(f"Expected reference action dim 14, got {reference_action.shape}")
            reference_actions14.append(reference_action.astype(np.float32))
        self.reference_action14 = torch.from_numpy(np.stack(reference_actions14, axis=0)).to(
            self.device
        )
        if self.condition_frame_length > 0:
            repeated_condition_action = self.current_action14[:, None, :].repeat(
                1, self.condition_frame_length, 1
            ).to(self.condition_action.device, dtype=self.condition_action.dtype)
            if any(kir_context_enabled):
                kir_mask = torch.tensor(
                    kir_context_enabled,
                    device=self.condition_action.device,
                    dtype=torch.bool,
                )[:, None, None]
                self.condition_action = torch.where(
                    kir_mask, self.condition_action, repeated_condition_action
                )
            else:
                self.condition_action = repeated_condition_action

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
            context_action_np = (
                condition_actions[env_idx].detach().cpu().numpy().astype(np.float32)
            )
            if kir_context_enabled[env_idx]:
                context_frames = self.current_obs[env_idx, :, 0, :, :, :].clone()
                self.image_queue[env_idx] = [
                    context_frames[:, t : t + 1].clone()
                    for t in range(self.condition_frame_length)
                ]
                self.frame_history[env_idx] = [
                    context_frames[:, t : t + 1].clone()
                    for t in range(self.condition_frame_length)
                ]
                self.action_history14[env_idx] = [
                    context_action_np[t].reshape(14).copy()
                    for t in range(self.condition_frame_length)
                ]
            else:
                self.image_queue[env_idx] = [
                    reference_frame.clone() for _ in range(self.condition_frame_length)
                ]
                self.frame_history[env_idx] = [reference_frame.clone()]
                self.action_history14[env_idx] = [
                    self.reference_action14[env_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(14)
                    .copy()
                ]
                self.current_obs[env_idx, :, 0, :, :, :] = reference_frame.repeat(
                    1, self.condition_frame_length, 1, 1
                )

        # Store init_ee_poses
        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses
        self._reset_task_specific_rollout_state(task_descriptions)

        # Wrap observation to match libero_env format
        extracted_obs = self._wrap_obs()
        infos = {}
        if self.episode_reward_group_ids is not None:
            infos["reward_group_id"] = self.current_reward_group_ids.clone()
        infos.update(self.get_reset_metadata())

        return extracted_obs, infos

    def get_reset_metadata(self) -> dict[str, torch.Tensor]:
        """Return optional logging IDs for the current policy observation.

        These IDs do not condition the policy or affect sampling. Environments
        without reward-group metadata retain the legacy empty metadata path.
        """
        if (
            self.episode_reward_group_ids is None
            or self.current_reset_history_ids is None
        ):
            return {}
        return {
            "reset_stage_ids": self.current_reward_group_ids.detach()
            .cpu()
            .reshape(-1, 1)
            .clone(),
            "reset_history_ids": self.current_reset_history_ids.detach()
            .cpu()
            .reshape(-1, 1)
            .clone(),
        }

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("step in Wan Env is not impl, use chunk_step instead")

    @staticmethod
    def _project_reward_point(point_xyz, extrinsic, intrinsic, height, width):
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

    def _select_gripper_tint_norms(self, actions):
        if actions is None:
            raise ValueError(
                "Gripper-tint reward inference requires the actions aligned with "
                "the generated chunk"
            )
        if not self.is_robotwin_env:
            raise ValueError("Gripper-tint reward inference currently requires RoboTwin")
        if self.robotwin_projector is None:
            raise RuntimeError("RobotWin camera projector is not initialized")
        if "head" not in self.camera_names:
            raise ValueError(
                f"Gripper-tint reward inference requires a head camera, got {self.camera_names}"
            )

        actions_tensor = (
            torch.from_numpy(actions).to(self.device)
            if isinstance(actions, np.ndarray)
            else actions.to(self.device)
        ).float()
        if actions_tensor.shape != (self.num_envs, self.chunk, 14):
            raise ValueError(
                "Expected gripper-tint reward actions "
                f"{(self.num_envs, self.chunk, 14)}, got {tuple(actions_tensor.shape)}"
            )
        actions_np = actions_tensor.detach().cpu().numpy()

        head_view_idx = self.camera_names.index("head")
        head_robotwin_cam = self.robotwin_camera_names[head_view_idx]
        value_min = float(self.cfg.reward_model.get("gripper_value_min", 0.0))
        value_max = float(self.cfg.reward_model.get("gripper_value_max", 1.0))
        if value_max <= value_min:
            raise ValueError(
                f"gripper_value_max must exceed gripper_value_min, got {value_min}, {value_max}"
            )

        selected_norms = np.empty(
            (self.num_envs, self.chunk), dtype=np.float32
        )
        for env_idx in range(self.num_envs):
            anchor = self.robotwin_camera_anchors[env_idx]
            if anchor is None:
                raise RuntimeError(
                    f"Missing RobotWin camera anchor for env {env_idx}"
                )
            action_seq = actions_np[env_idx]
            ee_pose, _ = self._action14_to_ee_pose_and_state(action_seq)
            left_pose = ee_pose[:, 0:7]
            right_pose = ee_pose[:, 7:14]
            w2c_seq = self.robotwin_projector.build_w2c_sequences(
                anchor, left_pose, right_pose
            )[head_robotwin_cam]
            original_h, original_w = anchor.original_hw_by_cam[head_robotwin_cam]
            intrinsic = robotwin_scale_intrinsic(
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
                    projection = self._project_reward_point(
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
                    selected = min(visible, key=lambda choice: choice["center_dist"])
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

    def _prepare_gripper_tint_reward_obs(self, stacked_obs, actions):
        if self.cfg.reward_model.get("input_view", "head") != "head":
            raise ValueError("Gripper-tint reward model currently supports input_view=head")
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
        head_obs = stacked_obs[:, :, head_start:head_end, :]
        head_obs = head_obs.reshape(
            self.num_envs, self.chunk, 3, self.view_height, self.view_width
        )

        gripper_norms = self._select_gripper_tint_norms(actions)
        alpha = float(self.cfg.reward_model.get("tint_alpha", 0.42))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        green = float(self.cfg.reward_model.get("tint_green", 35.0))
        colors = torch.stack(
            [
                255.0 * gripper_norms,
                torch.full_like(gripper_norms, green),
                255.0 * (1.0 - gripper_norms),
            ],
            dim=-1,
        )[:, :, :, None, None]

        # Match the training pipeline: uint8 RGB -> alpha blend -> uint8 -> [-1, 1].
        head_uint8 = torch.clamp((head_obs + 1.0) * 127.5, 0.0, 255.0).to(
            torch.uint8
        )
        tinted_uint8 = (
            head_uint8.to(torch.float32) * (1.0 - alpha) + colors * alpha
        ).clamp(0.0, 255.0).to(torch.uint8)
        tinted_obs = tinted_uint8.to(torch.float32) / 127.5 - 1.0
        return tinted_obs.reshape(
            expected_frames, 3, self.view_height, self.view_width
        )

    def _infer_next_chunk_rewards(self, actions=None):
        """Predict next reward using the reward model"""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        # Extract chunk observations
        num_envs, c, v, t, h, w = self.current_obs.shape
        extract_chunk_obs = self.current_obs.permute(
            0, 3, 1, 2, 4, 5
        )  # [num_envs, chunk + condition_frame_length, 3, v, h, w]

        if self.cfg.reward_model.type in {
            "ResnetRewModel",
            "InstructionRoutedResnetRewModel",
            "EpisodeGroupRoutedResnetRewModel",
            "InstructionRoutedGripperTintResnetRewModel",
        }:
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

            if (
                self.cfg.reward_model.type
                == "InstructionRoutedGripperTintResnetRewModel"
            ):
                extract_chunk_obs = self._prepare_gripper_tint_reward_obs(
                    extract_chunk_obs, actions
                )
            else:
                extract_chunk_obs = self._select_reward_model_view(
                    extract_chunk_obs
                )

            if self.cfg.reward_model.type in {
                "InstructionRoutedResnetRewModel",
                "InstructionRoutedGripperTintResnetRewModel",
            }:
                instructions = []
                for env_idx in range(self.num_envs):
                    instructions.extend(
                        [self.task_descriptions[env_idx]] * self.chunk
                    )
                rewards = self.reward_model.predict_rew(
                    extract_chunk_obs, instructions
                )
            elif (
                self.cfg.reward_model.type
                == "EpisodeGroupRoutedResnetRewModel"
            ):
                if (self.current_reward_group_ids < 0).any():
                    raise ValueError(
                        "Episode-group reward routing is active, but reset did not "
                        "assign a valid reward group to every environment"
                    )
                frame_group_ids = self.current_reward_group_ids.repeat_interleave(
                    self.chunk
                ).tolist()
                rewards = self.reward_model.predict_rew(
                    extract_chunk_obs, frame_group_ids
                )
            else:
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

    def _infer_next_chunk_frames(self, actions, active_env_mask=None):
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
        if active_env_mask is None:
            active_env_mask = torch.ones(
                num_envs, dtype=torch.bool, device=self.device
            )
        else:
            active_env_mask = active_env_mask.to(
                device=self.device, dtype=torch.bool
            )
            if active_env_mask.shape != (num_envs,):
                raise ValueError(
                    f"Expected active_env_mask [{num_envs}], got "
                    f"{tuple(active_env_mask.shape)}"
                )

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
            if not bool(active_env_mask[env_idx].item()):
                padding_frame = self.current_obs[
                    env_idx, :, 0, -1:, :, :
                ].detach().cpu()
                all_samples.append(padding_frame.repeat(1, self.chunk, 1, 1))
                previous_control = (
                    self.last_control_video[env_idx]
                    if env_idx < len(self.last_control_video)
                    else []
                )
                last_control_video.append(previous_control)
                continue

            if self.rollout_context_mode == "sparse_history":
                context_frames, context_action_seq = self._sparse_history_context(env_idx)
            else:
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

            if self.rollout_context_mode == "sparse_history":
                action_seq = np.concatenate(
                    [
                        context_action_seq,
                        actions_tensor[env_idx].detach().cpu().numpy(),
                    ],
                    axis=0,
                )
            else:
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
            if self.wm_prompt_mode == "fixed":
                prompt = self.prompt
            else:
                prompt = self.task_descriptions[env_idx] or self.prompt

            kwargs = {
                "prompt": "机械臂按照要求移动夹爪执行任务",
                # "seed": int(self.cfg.get("seed", 0)) + int(self.elapsed_steps) + env_idx,
                "seed": 0,
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

            all_samples.append(pred_video[:, -self.chunk :])
            self.current_action14[env_idx] = torch.from_numpy(action_seq[-1]).to(
                device=self.current_action14.device, dtype=self.current_action14.dtype
            )
            if self.rollout_context_mode == "sparse_history":
                self._append_sparse_history(
                    env_idx, pred_video[:, -self.chunk :], actions_tensor[env_idx]
                )
                sparse_frames, sparse_actions = self._sparse_history_context(env_idx)
                self.image_queue[env_idx] = sparse_frames
                self.condition_action[env_idx] = torch.from_numpy(sparse_actions).to(
                    device=self.device, dtype=actions_tensor.dtype
                )
            else:
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
        rm_reward_chunks = []
        termination_chunks = []
        truncation_chunks = []
        self.last_control_video_chunks = []
        extracted_obs = None
        num_rollout_chunks = action_chunk_len // self.chunk
        task_specific_mode = self.task_specific_rollout_enabled
        active_env_mask = (
            self.active_env_mask.clone()
            if task_specific_mode
            else torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )
        reward_group_ids_for_video = self.current_reward_group_ids.clone()
        success_in_action_chunk = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if task_specific_mode:
            misaligned_horizons = self.env_episode_max_steps % action_chunk_len != 0
            if misaligned_horizons.any():
                bad = self.env_episode_max_steps[misaligned_horizons].tolist()
                raise ValueError(
                    "Task-specific horizons must align with the full VLA action chunk "
                    f"({action_chunk_len}), got {bad}"
                )
        # 这里计算VLA的action_chunk是wm的chunk的几倍
        for chunk_idx in range(num_rollout_chunks):
            start = chunk_idx * self.chunk
            end = start + self.chunk
            sub_actions = actions_tensor[:, start:end, :]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                self._infer_next_chunk_frames(
                    sub_actions,
                    active_env_mask=active_env_mask if task_specific_mode else None,
                )

            # Update elapsed steps after each world-model autoregressive rollout.
            self.elapsed_steps += self.chunk 
            # 0/8/16，就是一个数，目前还不能表示组内每条轨迹

            # Read the last frame from self.current_obs.
            extracted_obs = self._wrap_obs()

            # Get rewards for this generated world-model chunk.
            if task_specific_mode and not active_env_mask.any():
                chunk_rewards = torch.zeros(
                    self.num_envs,
                    self.chunk,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                chunk_rewards = self._infer_next_chunk_rewards(sub_actions)
            # import pdb;pdb.set_trace()
            # [B,T]-[1,8],是一个WM chunk的值
            rm_chunk_rewards = chunk_rewards.clone()
            if task_specific_mode:
                rm_chunk_rewards = torch.where(
                    active_env_mask[:, None],
                    rm_chunk_rewards,
                    torch.zeros_like(rm_chunk_rewards),
                )
            rm_reward_chunks.append(rm_chunk_rewards)
            chunk_rewards = self._apply_success_reward_latch(chunk_rewards)
            if task_specific_mode:
                chunk_rewards = torch.where(
                    active_env_mask[:, None],
                    chunk_rewards,
                    torch.zeros_like(chunk_rewards),
                )

            chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)
            if task_specific_mode:
                chunk_rewards_tensors = torch.where(
                    active_env_mask[:, None],
                    chunk_rewards_tensors,
                    torch.zeros_like(chunk_rewards_tensors),
                )
            # 目前等价
            reward_chunks.append(chunk_rewards_tensors)

            # Simple strategy: mark success/truncation at the end of each generated sub-chunk,
            # but do not stop early inside a longer policy action chunk.
            estimated_success = self._estimate_success_from_rewards(chunk_rewards)
            raw_chunk_terminations = torch.zeros(
                self.num_envs, self.chunk, dtype=torch.bool, device=self.device
            )
            raw_chunk_truncations = torch.zeros(
                self.num_envs, self.chunk, dtype=torch.bool, device=self.device
            )
            if task_specific_mode:
                success_in_action_chunk |= estimated_success & active_env_mask
            else:
                raw_chunk_terminations[:, -1] = estimated_success
                # 仅在每个sub-chunk的最后一步标记成功/终止，其他步骤不标记，即使reward已经超过阈值也继续执行后续动作，直到sub-chunk结束
                truncations = torch.tensor(
                    self.elapsed_steps >= self.cfg.max_episode_steps,
                    device=self.device,
                )
                if truncations.any():
                    raw_chunk_truncations[:, -1] = truncations # 同样仅在sub-chunk的最后一步标记截断

            termination_chunks.append(raw_chunk_terminations)
            truncation_chunks.append(raw_chunk_truncations)

        if task_specific_mode:
            self.env_elapsed_steps[active_env_mask] += action_chunk_len
            newly_terminated = (
                success_in_action_chunk & active_env_mask
                if self.task_specific_stop_on_success
                else torch.zeros_like(active_env_mask)
            )
            newly_truncated = (
                active_env_mask
                & (self.env_elapsed_steps >= self.env_episode_max_steps)
                & ~newly_terminated
            )
            termination_chunks[-1][:, -1] = newly_terminated
            truncation_chunks[-1][:, -1] = newly_truncated
            self.active_env_mask &= ~(newly_terminated | newly_truncated)

        chunk_rewards_tensors = torch.cat(reward_chunks, dim=1)
        rm_rewards_tensors = torch.cat(rm_reward_chunks, dim=1)
        # 这个存每帧的奖励
        raw_chunk_terminations = torch.cat(termination_chunks, dim=1)
        # 这个可以表示每个WM-chunk是否成功
        raw_chunk_truncations = torch.cat(truncation_chunks, dim=1)
        # 这个表示每个WM-chunk是否超出最大限制长度
        # print(f'chunk_rewards_tensors:{chunk_rewards_tensors},raw_chunk_termiations:{raw_chunk_terminations}')
        '''
        类似这种
        chunk_rewards_tensors:tensor([[0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0'),raw_chunk_termiations:tensor([[False, False, False, False, False, False, False, False]],
       device='cuda:0')
        '''
        # import pdb;pdb.set_trace()
        # 把每个sub-chunk的奖励、终止、截断结果拼接起来，得到完整的chunk级别的奖励和done信号
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
        # 得到每个环境在整个chunk动作执行完后是否已经终止或截断的结果
        # import pdb;pdb.set_trace()
        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}

        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, infos
        )
        # These chunk-aligned fields are optional video metadata. RecordVideo only
        # renders them when requested through video_cfg.extra_info_on_video.
        infos["stage"] = reward_group_ids_for_video
        # The current IDs describe the returned observation after any reset;
        # the separate video stage above describes the action just executed.
        infos.update(self.get_reset_metadata())
        infos["rm_reward"] = rm_rewards_tensors.max(dim=1).values
        infos["train_reward"] = chunk_rewards_tensors.sum(dim=1)
        success_threshold = float(
            getattr(self.cfg, "success_reward_threshold", 0.9)
        )
        infos["rm_success"] = (rm_rewards_tensors >= success_threshold).any(dim=1)
        # info存的是每个VLA-chunk的平均奖励/是否成功
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
        self.reward_success_latched = self.reward_success_latched.cpu()
        self.active_env_mask = self.active_env_mask.cpu()
        self.env_elapsed_steps = self.env_elapsed_steps.cpu()
        self.env_episode_max_steps = self.env_episode_max_steps.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        self.current_reward_group_ids = self.current_reward_group_ids.cpu()
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
        self.reward_success_latched = self.reward_success_latched.to(self.device)
        self.active_env_mask = self.active_env_mask.to(self.device)
        self.env_elapsed_steps = self.env_elapsed_steps.to(self.device)
        self.env_episode_max_steps = self.env_episode_max_steps.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        self.current_reward_group_ids = self.current_reward_group_ids.to(self.device)
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
            "active_env_mask": self.active_env_mask.cpu(),
            "env_elapsed_steps": self.env_elapsed_steps.cpu(),
            "env_episode_max_steps": self.env_episode_max_steps.cpu(),
            "prev_step_reward": self.prev_step_reward.cpu(),
            "reward_success_latched": self.reward_success_latched.cpu(),
            "_is_start": self._is_start,
            "reset_state_ids": self.reset_state_ids.cpu(),
            "current_reward_group_ids": self.current_reward_group_ids.cpu(),
            "reward_group_sampling_round": self._reward_group_sampling_round,
            "generator_state": self._generator.get_state(),
        }
        if self.episode_reward_group_ids is not None:
            env_state["current_reset_history_ids"] = self.current_reset_history_ids
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


# export PYTHONPATH=/data/zsq/diffsynth-studio-wujie-ppu:/data/zsq/RLinf_A100:$PYTHONPATH python -m rlinf.envs.world_model.world_model_wan_multi_view_env
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
        "EMBODIED_CONFIG_NAME", "wanmultiview_arxx5_newstructure_new_cam_0723_task_specific_grpo_pi05"
    )

    print(f"Loading config: {config_name} from {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    rollout_episode_index = int(os.environ.get("WM_ROLLOUT_EPISODE_INDEX", "10"))
    rollout_dataset_dir = Path(
        os.environ.get("WM_ROLLOUT_DATASET_DIR", cfg.initial_image_path)
    )
    rollout_output_dir = Path(
        os.environ.get(
            "WM_ROLLOUT_OUTPUT_DIR",
            "/data/zsq/RLinf_A100/outputs/wanmultiview_rollout",
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

    obs, info = env.reset(episode_indices=np.asarray([rollout_episode_index]))
    del obs, info

    if env.enable_kir and env.kir_initial_frame is not None:
        rollout_start_frame = min(max(int(env.kir_initial_frame) + 1, 1), len(actions))
        initial_frame_idx = min(max(int(env.kir_initial_frame), 0), len(actions) - 1)
    else:
        # Frame 0 is the reset condition image/state. Roll out frames 1..T-1.
        rollout_start_frame = 1
        initial_frame_idx = 0

    rollout_actions = actions[rollout_start_frame:]
    rollout_frame_indices = list(range(rollout_start_frame, rollout_start_frame + len(rollout_actions)))
    max_steps_env = os.environ.get("WM_ROLLOUT_MAX_STEPS", None)
    if max_steps_env is not None:
        max_steps = int(max_steps_env)
        rollout_actions = rollout_actions[:max_steps]
        rollout_frame_indices = rollout_frame_indices[:max_steps]

    with open_dict(cfg):
        cfg.max_episode_steps = int(len(rollout_actions))

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
    output_frame_indices = [initial_frame_idx]

    num_chunks = int(np.ceil(len(rollout_actions) / env.chunk))
    print(
        f"Rolling out episode={rollout_episode_index}, "
        f"kir={bool(env.enable_kir and env.kir_initial_frame is not None)}, "
        f"initial_frame={initial_frame_idx}, action_start_frame={rollout_start_frame}, "
        f"steps={len(rollout_actions)}, chunks={num_chunks}, dataset={rollout_dataset_dir}"
    )

    for chunk_idx in range(num_chunks):
        start = chunk_idx * env.chunk #  0, 8, 16, ...
        end = min(start + env.chunk, len(rollout_actions)) # 8, 16, ..., len(rollout_actions)
        chunk = rollout_actions[start:end]
        chunk_frame_indices = rollout_frame_indices[start:end]
        valid_len = len(chunk)
        if valid_len < env.chunk:
            pad = np.repeat(chunk[-1:], env.chunk - valid_len, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0) # [8, 14]

        _, chunk_rewards, _, _, _ = env.chunk_step(chunk[None, :, :]) # 输入[1, chunk, 14] 输出[1, chunk]
        chunk_reward_np = chunk_rewards[0, :valid_len].detach().cpu().numpy()
        chunk_frames = tensor_frames_to_uint8(
            env.current_obs[0, :, 0, -env.chunk:, :, :]
        )[:valid_len]
        chunk_traj_frames = env.last_control_video[0][:valid_len]
        generated_frames.extend(list(chunk_frames))
        traj_frames.extend(list(chunk_traj_frames))
        rewards.extend([float(x) for x in chunk_reward_np])
        output_frame_indices.extend(chunk_frame_indices)
        print(
            f"chunk {chunk_idx + 1}/{num_chunks}: "
            f"frames={valid_len}, frame_range="
            f"{chunk_frame_indices[0] if chunk_frame_indices else 'NA'}.."
            f"{chunk_frame_indices[-1] if chunk_frame_indices else 'NA'}, "
            f"rewards={chunk_reward_np.tolist()}"
        )

    rollout_output_dir.mkdir(parents=True, exist_ok=True)
    annotated_frames = [
        append_traj_map(annotate_frame(frame, frame_idx, reward), traj_frame)
        for frame_idx, frame, traj_frame, reward in zip(
            output_frame_indices, generated_frames, traj_frames, rewards
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
                "enable_kir": bool(env.enable_kir),
                "kir_initial_frame": env.kir_initial_frame,
                "initial_frame_idx": initial_frame_idx,
                "rollout_start_frame": rollout_start_frame,
                "frame_indices": output_frame_indices,
                "rewards": rewards,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved rollout video: {video_path}")
    print(f"Saved rewards: {rewards_path}")
