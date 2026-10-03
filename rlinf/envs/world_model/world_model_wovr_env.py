# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboTwin environment for the original action-conditioned WoVR models.

The WoVR checkpoints use a fixed reference frame, four rolling context frames,
and eight predicted frames. They are intentionally kept separate from the VACE
multi-view environment: the model loading and conditioning contracts differ.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.reward_model_compat import (
    InstructionRoutedGripperTintResnetRewModel,
    ResnetRewModel,
)
from rlinf.envs.world_model.world_model_wan_env import WanEnv

__all__ = ["WovrEnv"]


class InstructionRoutedResnetRewModel(torch.nn.Module):
    """Route frames to task-specific reward models using exact instructions."""

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
                "InstructionRoutedResnetRewModel expects one instruction per frame"
            )

        rewards = torch.empty(obs.shape[0], device=obs.device, dtype=torch.float32)
        grouped_indices = {}
        for frame_idx, instruction in enumerate(instructions):
            normalized_instruction = str(instruction).strip()
            if normalized_instruction not in self.instruction_to_model_key:
                available = sorted(self.instruction_to_model_key)
                raise KeyError(
                    f"No reward model for instruction {normalized_instruction!r}; "
                    f"available instructions: {available}"
                )
            model_key = self.instruction_to_model_key[normalized_instruction]
            grouped_indices.setdefault(model_key, []).append(frame_idx)

        for model_key, indices in grouped_indices.items():
            index_tensor = torch.tensor(indices, device=obs.device, dtype=torch.long)
            rewards[index_tensor] = self.models[model_key].predict_rew(
                obs.index_select(0, index_tensor)
            ).reshape(-1)
        return rewards


class WovrEnv(WanEnv):
    """WoVR world-model environment with the Wan multi-view RLinf contract."""

    _VALID_VIEW_MODES = {"single", "three_vertical"}

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
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            record_metrics=record_metrics,
            worker_info=worker_info,
        )

        self.wm_env_type = str(cfg.get("wm_env_type", "robotwin")).strip().lower()
        if self.wm_env_type not in {"robotwin", "arxx5", "arxx5_newcam"}:
            raise ValueError(
                "WovrEnv supports wm_env_type=robotwin, arxx5, or "
                f"arxx5_newcam, got {self.wm_env_type!r}"
            )
        self.is_robotwin_env = self.wm_env_type == "robotwin"

        self.action_dim = int(cfg.get("action_dim", 14))
        if self.action_dim != 14:
            raise ValueError(f"WoVR expects action_dim=14, got {self.action_dim}")
        if self.condition_frame_length != 5 or self.chunk != 8 or self.num_frames != 13:
            raise ValueError(
                "WoVR checkpoints require condition_frame_length=5, chunk=8, "
                f"num_frames=13; got {self.condition_frame_length}, "
                f"{self.chunk}, {self.num_frames}"
            )

        self.wovr_view_mode = str(cfg.get("wovr_view_mode", "single")).lower()
        if self.wovr_view_mode not in self._VALID_VIEW_MODES:
            raise ValueError(
                f"Unsupported wovr_view_mode={self.wovr_view_mode!r}; "
                f"expected one of {sorted(self._VALID_VIEW_MODES)}"
            )

        self.view_height = int(cfg.get("view_height", 256))
        self.view_width = int(cfg.get("view_width", 320))
        self.camera_names = list(
            cfg.get("camera_names", ["head", "left_wrist", "right_wrist"])
        )
        if self.wovr_view_mode == "three_vertical" and self.camera_names != [
            "head",
            "left_wrist",
            "right_wrist",
        ]:
            raise ValueError(
                "three_vertical WoVR expects camera_names="
                "[head, left_wrist, right_wrist] in that order"
            )

        self._gripper_tint_enabled = (
            self.cfg.reward_model.type
            == "InstructionRoutedGripperTintResnetRewModel"
        )
        self._gripper_tint_support = None
        if self._gripper_tint_enabled:
            self._init_gripper_tint_support()

        expected_height = self.view_height * (
            3 if self.wovr_view_mode == "three_vertical" else 1
        )
        if self.image_size != (expected_height, self.view_width):
            raise ValueError(
                f"{self.wovr_view_mode} WoVR expects image_size="
                f"[{expected_height}, {self.view_width}], got {self.image_size}"
            )

        self.initial_joint_source = str(
            cfg.get("initial_joint_source", "action")
        ).strip().lower()
        if self.initial_joint_source not in {"action", "state"}:
            raise ValueError(
                "initial_joint_source must be 'action' or 'state', got "
                f"{self.initial_joint_source!r}"
            )

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
        # WanEnv initializes reset IDs before WoVR-specific grouping is known.
        # Refresh them now that the optional manifest has been loaded.
        self.update_reset_state_ids()

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
        if self.task_specific_rollout_enabled and not self.task_horizon_by_instruction:
            raise ValueError(
                "task_specific_rollout.enabled=true requires "
                "task_specific_rollout.horizon_by_instruction"
            )
        if self.task_specific_rollout_enabled and self.auto_reset:
            raise ValueError(
                "task-specific padded rollout requires auto_reset=false so completed "
                "environments remain inactive until the next rollout reset"
            )
        invalid_horizons = {
            instruction: max_steps
            for instruction, max_steps in self.task_horizon_by_instruction.items()
            if max_steps <= 0 or max_steps % self.chunk != 0
        }
        if invalid_horizons:
            raise ValueError(
                "Task-specific horizons must be positive multiples of the WM chunk "
                f"({self.chunk}), got {invalid_horizons}"
            )
        if (
            self.task_specific_rollout_enabled
            and max(self.task_horizon_by_instruction.values())
            > int(cfg.max_episode_steps)
        ):
            raise ValueError(
                "env.max_episode_steps must cover the longest task-specific horizon: "
                f"global={cfg.max_episode_steps}, "
                f"task_max={max(self.task_horizon_by_instruction.values())}"
            )

        self.latch_success_reward = bool(cfg.get("latch_success_reward", False))
        self.reward_success_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.kir_initial_frame = None
        self.current_action14 = torch.zeros(self.num_envs, 14, device=self.device)
        self.reference_action14 = torch.zeros(self.num_envs, 14, device=self.device)
        self.reference_frames = [None] * self.num_envs
        self.last_control_video = [[] for _ in range(self.num_envs)]
        self.last_control_video_chunks = []
        self.condition_action = torch.zeros(
            self.num_envs,
            self.condition_frame_length,
            14,
            dtype=torch.float32,
            device=self.device,
        )
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

        dit = self.pipe.dit
        if getattr(dit, "action_mode", None) != "both":
            raise ValueError(
                f"WoVR checkpoint must use action_mode='both', got "
                f"{getattr(dit, 'action_mode', None)!r}"
            )
        if int(getattr(dit, "length_conditonal_frames", -1)) != 5:
            raise ValueError(
                "WoVR checkpoint must use length_conditonal_frames=5, got "
                f"{getattr(dit, 'length_conditonal_frames', None)!r}"
            )
        action_in_features = int(dit.action_mlp1[0].in_features)
        if action_in_features != 14:
            raise ValueError(
                f"WoVR checkpoint action input is {action_in_features}D, expected 14D"
            )

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path,
            action_key=cfg.get("dataset_action_key", "action"),
            state_key=cfg.get("dataset_state_key", "state"),
            enable_kir=False,
        )

    def _init_episode_reward_groups(self):
        """Load optional task groups used for balanced reset sampling."""
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
            if self.reward_group_sampling == "balanced":
                raise FileNotFoundError(
                    "Balanced reward-group sampling requires a manifest: "
                    f"{manifest_path}"
                )
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mapping = manifest.get("episode_mapping") or manifest.get("episodes", [])
        output_file_to_group = {}
        for item in mapping:
            output_file = Path(str(item.get("output_file", ""))).name
            group_id = item.get(
                "group_id",
                item.get("reward_group_id", item.get("task_index")),
            )
            if not output_file or group_id is None:
                continue
            if output_file in output_file_to_group:
                raise ValueError(
                    f"Duplicate output_file {output_file!r} in {manifest_path}"
                )
            output_file_to_group[output_file] = int(group_id)

        if not output_file_to_group:
            if self.reward_group_sampling == "balanced":
                raise ValueError(
                    "Balanced reward-group sampling found no output_file/group ID "
                    f"mapping in {manifest_path}"
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
            raise ValueError(
                f"{manifest_path} is missing reward groups for {len(missing_files)} "
                f"dataset files, including {missing_files[:8]}"
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

    def update_reset_state_ids(self):
        """Sample one initial episode per GRPO group, optionally task-balanced."""
        # WanEnv calls this virtual method before WovrEnv has initialized its
        # optional reward-group metadata.
        if not hasattr(self, "reward_group_sampling"):
            self.reset_state_ids = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            return

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
            groups_per_round = self.total_num_processes * self.num_group
            sampled_episode_ids = []
            for local_group_idx in range(self.num_group):
                global_group_slot = (
                    self._reward_group_sampling_round * groups_per_round
                    + worker_rank * self.num_group
                    + local_group_idx
                )
                group_id = group_ids[global_group_slot % len(group_ids)]
                candidates = self.episode_indices_by_reward_group[group_id]
                candidate_idx = torch.randint(
                    low=0,
                    high=len(candidates),
                    size=(1,),
                    generator=self._generator,
                ).item()
                sampled_episode_ids.append(int(candidates[candidate_idx]))
            reset_state_ids = torch.tensor(sampled_episode_ids, dtype=torch.long)
            self._reward_group_sampling_round += 1
        else:
            reset_state_ids = torch.randint(
                low=0,
                high=len(self.dataset),
                size=(self.num_group,),
                generator=self._generator,
            )

        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _reset_task_specific_rollout_state(self, task_descriptions):
        """Reset per-env horizons while keeping the distributed tensor shape fixed."""
        self.active_env_mask.fill_(True)
        self.env_elapsed_steps.zero_()
        self.env_episode_max_steps.fill_(int(self.cfg.max_episode_steps))
        if not self.task_specific_rollout_enabled:
            return

        normalized_tasks = [str(task).strip() for task in task_descriptions]
        missing = sorted(
            set(normalized_tasks).difference(self.task_horizon_by_instruction)
        )
        if missing:
            raise ValueError(
                "No task-specific rollout horizon configured for instructions "
                f"{missing}. Known instructions: "
                f"{sorted(self.task_horizon_by_instruction)}"
            )
        self.env_episode_max_steps.copy_(
            torch.tensor(
                [
                    self.task_horizon_by_instruction[task]
                    for task in normalized_tasks
                ],
                dtype=torch.long,
                device=self.device,
            )
        )

    def _build_pipeline(self):
        previous_action_dim = os.environ.get("WAN_ACTION_DIM")
        os.environ["WAN_ACTION_DIM"] = "14"
        try:
            pipe = super()._build_pipeline()
        finally:
            if previous_action_dim is None:
                os.environ.pop("WAN_ACTION_DIM", None)
            else:
                os.environ["WAN_ACTION_DIM"] = previous_action_dim
        return pipe

    def _init_gripper_tint_support(self):
        """Initialize tint preprocessing without importing the VACE environment."""
        from rlinf.envs.world_model.robotwin_gripper_tint import (
            RobotWinGripperTintSupport,
        )

        self._gripper_tint_support = RobotWinGripperTintSupport(
            self.cfg,
            num_envs=self.num_envs,
            chunk=self.chunk,
            camera_names=self.camera_names,
            view_height=self.view_height,
            view_width=self.view_width,
            device=self.device,
        )

    def _load_reward_model(self):
        reward_type = self.cfg.reward_model.type
        threshold = self.cfg.reward_model.get("threshold", None)
        if reward_type == "ResnetRewModel":
            return ResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                threshold=threshold,
            )
        if reward_type == "InstructionRoutedResnetRewModel":
            return InstructionRoutedResnetRewModel(
                checkpoint_root=self.cfg.reward_model.from_pretrained,
                instruction_to_checkpoint=(
                    self.cfg.reward_model.instruction_to_checkpoint
                ),
                threshold=threshold,
            )
        if reward_type == "InstructionRoutedGripperTintResnetRewModel":
            return InstructionRoutedGripperTintResnetRewModel(
                checkpoint_root=self.cfg.reward_model.from_pretrained,
                task_routes=self.cfg.reward_model.task_routes,
                threshold=threshold,
            )
        if reward_type == "DummyRewardModel":
            class DummyRewardModel(torch.nn.Module):
                def predict_rew(self, obs, instructions=None):
                    del instructions
                    return torch.zeros(
                        obs.shape[0], device=obs.device, dtype=torch.float32
                    )

            return DummyRewardModel()
        return super()._load_reward_model()

    def _prepare_initial_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected initial image [3,H,W], got {tuple(image.shape)}")

        image = image.float()
        source_height = int(image.shape[1])
        if self.wovr_view_mode == "single":
            if source_height >= self.view_height * 3:
                image = image[:, : source_height // 3]
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.view_height, self.view_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            return image

        if source_height < 3:
            raise ValueError(f"Invalid three-view image height {source_height}")
        if source_height % 3 != 0:
            raise ValueError(
                f"three_vertical initial image height must be divisible by 3, got "
                f"{source_height}"
            )
        source_view_height = source_height // 3
        views = []
        for view_idx in range(3):
            view = image[
                :,
                view_idx * source_view_height : (view_idx + 1) * source_view_height,
            ]
            view = F.interpolate(
                view.unsqueeze(0),
                size=(self.view_height, self.view_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            views.append(view)
        return torch.cat(views, dim=1)

    @staticmethod
    def _to_action14(value, *, name):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape[0] != 14:
            raise ValueError(f"Expected {name} dim 14, got {value.shape}")
        return value

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        del options
        self.onload()
        self.elapsed_steps = 0

        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        if len(self.dataset) < self.num_envs:
            raise ValueError(
                f"Not enough episodes in dataset: found {len(self.dataset)}, "
                f"need {self.num_envs}"
            )

        if episode_indices is None:
            rng_seed = self.seed
            if seed is not None:
                rng_seed = int(seed[0] if isinstance(seed, list) else seed)
            rng = np.random.default_rng(rng_seed)
            episode_indices = rng.choice(
                len(self.dataset), size=self.num_envs, replace=False
            )
        elif isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.detach().cpu().numpy()
        episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        if episode_indices.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} episode indices, got "
                f"{episode_indices.shape[0]}"
            )
        if self.episode_reward_group_ids is not None:
            selected_group_ids = self.episode_reward_group_ids[episode_indices]
            self.current_reward_group_ids = torch.as_tensor(
                selected_group_ids, dtype=torch.long, device=self.device
            )
        else:
            self.current_reward_group_ids = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )

        image_tensors = []
        task_descriptions = []
        init_ee_poses = []
        current_actions14 = []

        for episode_idx in episode_indices:
            episode_data = self.dataset[int(episode_idx)]
            if not episode_data["start_items"]:
                raise ValueError(f"Empty start_items for episode {episode_idx}")
            first_frame = episode_data["start_items"][0]
            if "image" not in first_frame:
                raise ValueError(f"No image in episode {episode_idx} first frame")

            task_descriptions.append(str(episode_data.get("task", "")))
            init_ee_poses.append(None)

            image = self._prepare_initial_image(first_frame["image"])
            image = self.trans_norm(image)
            image_tensors.append(
                image.unsqueeze(1).repeat(1, self.condition_frame_length, 1, 1)
            )

            initial_joint_key = (
                "action"
                if self.initial_joint_source == "action"
                else "observation.state"
            )
            initial_action = first_frame.get(initial_joint_key)
            if initial_action is None and self.initial_joint_source == "action":
                initial_action = first_frame.get("observation.state")
            if initial_action is None:
                raise ValueError(
                    f"Episode {episode_idx} has no {initial_joint_key!r}; "
                    f"initial_joint_source={self.initial_joint_source!r}"
                )
            current_actions14.append(
                self._to_action14(initial_action, name="initial action/state")
            )

        stacked_images = torch.stack(image_tensors, dim=0).to(self.device)
        self.current_obs = stacked_images.unsqueeze(2)
        self.condition_action = torch.zeros(
            self.num_envs,
            self.condition_frame_length,
            14,
            device=self.device,
            dtype=torch.float32,
        )
        self.current_action14 = torch.from_numpy(
            np.stack(current_actions14, axis=0)
        ).to(self.device)
        self.reference_action14 = self.current_action14.clone()

        if self._gripper_tint_enabled:
            self._gripper_tint_support.reset(
                episode_indices, self.reference_action14
            )

        self.image_queue = []
        self.reference_frames = []
        for env_idx in range(self.num_envs):
            reference = self.current_obs[env_idx, :, 0, 0:1].clone()
            self.reference_frames.append(reference)
            self.image_queue.append(
                [reference.clone() for _ in range(self.condition_frame_length)]
            )

        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses
        self._reset_task_specific_rollout_state(task_descriptions)
        self._reset_metrics()
        infos = {}
        if self.episode_reward_group_ids is not None:
            infos["reward_group_id"] = self.current_reward_group_ids.clone()
        return self._wrap_obs(), infos

    def set_initial_action(self, action14):
        action = torch.as_tensor(action14, device=self.device, dtype=torch.float32)
        action = action.reshape(self.num_envs, 14)
        self.current_action14 = action.clone()
        self.reference_action14 = action.clone()

    @staticmethod
    def _frame_to_pil(frame: torch.Tensor) -> Image.Image:
        image = frame[:, 0].detach().float().cpu().permute(1, 2, 0).numpy()
        image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image)

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions, active_env_mask=None):
        actions_tensor = torch.as_tensor(
            actions, device=self.device, dtype=torch.float32
        )
        expected_shape = (self.num_envs, self.chunk, 14)
        if tuple(actions_tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected actions {expected_shape}, got {tuple(actions_tensor.shape)}"
            )
        if active_env_mask is None:
            active_env_mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        else:
            active_env_mask = active_env_mask.to(
                device=self.device, dtype=torch.bool
            )
            if active_env_mask.shape != (self.num_envs,):
                raise ValueError(
                    f"Expected active_env_mask [{self.num_envs}], got "
                    f"{tuple(active_env_mask.shape)}"
                )
        active_indices = torch.nonzero(active_env_mask, as_tuple=False).flatten()

        self.condition_action = self.condition_action.to(
            device=actions_tensor.device, dtype=actions_tensor.dtype
        )
        if self.retain_action:
            action_window = torch.cat(
                [self.condition_action, actions_tensor], dim=1
            )
        else:
            action_window = actions_tensor
        if action_window.shape[1] != self.num_frames:
            raise ValueError(
                f"WoVR action window must have {self.num_frames} frames, got "
                f"{action_window.shape[1]}; retain_action must normally be true"
            )

        reference_images = []
        context_images = []
        for env_idx in active_indices.tolist():
            frames = [self._frame_to_pil(frame) for frame in self.image_queue[env_idx]]
            reference_images.append(frames[0])
            context_images.append(frames[-4:])

        output = []
        if active_indices.numel() > 0:
            active_action_window = action_window.index_select(0, active_indices)
            output = self.pipe(
                seed=int(self.cfg.get("wm_seed", 0)),
                tiled=bool(self.cfg.get("tiled", False)),
                input_image=reference_images,
                input_image4=context_images,
                action=active_action_window,
                height=int(self.image_size[0]),
                width=int(self.image_size[1]),
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                cfg_scale=float(self.cfg.get("cfg_scale", 1.0)),
                sigma_shift=float(self.cfg.get("sigma_shift", 5.0)),
                progress_bar_cmd=lambda x: x,
                batch_size=int(active_indices.numel()),
                bs_1=False,
            )
            if len(output) != int(active_indices.numel()):
                raise RuntimeError(
                    f"WoVR pipeline returned batch {len(output)}, expected "
                    f"{int(active_indices.numel())} active environments"
                )

        predicted_chunks = [None] * self.num_envs
        for env_idx in range(self.num_envs):
            if not bool(active_env_mask[env_idx].item()):
                predicted_chunks[env_idx] = self.current_obs[
                    env_idx, :, 0, -1:
                ].repeat(1, self.chunk, 1, 1)

        for output_idx, output_frames in enumerate(output):
            env_idx = int(active_indices[output_idx].item())
            if len(output_frames) != self.num_frames:
                raise RuntimeError(
                    f"WoVR pipeline returned {len(output_frames)} frames, expected "
                    f"{self.num_frames}"
                )
            frames = []
            for image in output_frames:
                array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
                frames.append(array)
            video = torch.from_numpy(np.stack(frames, axis=0)).permute(3, 0, 1, 2)
            video = video.to(self.device)

            context = video[:, -4:].clone()
            self.image_queue[env_idx] = [
                self.reference_frames[env_idx].clone(),
                *[context[:, idx : idx + 1].clone() for idx in range(4)],
            ]
            predicted_chunks[env_idx] = video[:, -self.chunk :]

        if active_indices.numel() > 0:
            self.condition_action[active_indices] = 0
            self.condition_action[active_indices, 1:] = actions_tensor[
                active_indices, -4:
            ]
            self.current_action14[active_indices] = actions_tensor[
                active_indices, -1
            ]

        predicted = torch.stack(predicted_chunks, dim=0).unsqueeze(2)
        self.current_obs = torch.cat([self.current_obs, predicted], dim=3)
        self.current_obs = self.current_obs[:, :, :, -self.num_frames :]

    def _load_reward_frames(self):
        frames = self.current_obs[:, :, :, -self.chunk :]
        reward_frames = frames.permute(0, 3, 1, 2, 4, 5).reshape(
            self.num_envs * self.chunk,
            3,
            1,
            self.image_size[0],
            self.image_size[1],
        ).squeeze(2)
        if self.is_robotwin_env:
            return reward_frames

        if self.wovr_view_mode == "single":
            return F.interpolate(
                reward_frames,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        views = torch.chunk(reward_frames, 3, dim=2)
        resized_views = [
            F.interpolate(
                view,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            for view in views
        ]
        return torch.cat(resized_views, dim=2)

    @torch.no_grad()
    def _infer_next_chunk_rewards(self, actions=None):
        reward_frames = self._load_reward_frames().to(self.device)
        reward_type = self.cfg.reward_model.type
        if reward_type == "InstructionRoutedGripperTintResnetRewModel":
            reward_frames = self._gripper_tint_support.prepare_reward_obs(
                reward_frames, actions
            )
            instructions = []
            for instruction in self.task_descriptions:
                instructions.extend([instruction] * self.chunk)
            rewards = self.reward_model.predict_rew(reward_frames, instructions)
        elif reward_type == "InstructionRoutedResnetRewModel":
            instructions = []
            for instruction in self.task_descriptions:
                instructions.extend([instruction] * self.chunk)
            rewards = self.reward_model.predict_rew(reward_frames, instructions)
        elif reward_type in {
            "ResnetRewModel",
            "DummyRewardModel",
        }:
            rewards = self.reward_model.predict_rew(reward_frames)
        elif reward_type == "TaskEmbedResnetRewModel":
            instructions = []
            for instruction in self.task_descriptions:
                instructions.extend([instruction] * self.chunk)
            rewards = self.reward_model.predict_rew(reward_frames, instructions)
        else:
            raise ValueError(f"Unknown reward model type: {reward_type}")
        return rewards.reshape(self.num_envs, self.chunk)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.reward_success_latched[mask] = False
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.reward_success_latched[:] = False
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        if not self.record_metrics:
            return infos

        self.returns += step_reward
        if isinstance(terminations, torch.Tensor):
            terminations_tensor = terminations.to(
                device=self.device, dtype=torch.bool
            )
        else:
            terminations_tensor = torch.as_tensor(
                terminations, device=self.device, dtype=torch.bool
            )
        self.success_once |= terminations_tensor

        if self.task_specific_rollout_enabled:
            episode_len = self.env_elapsed_steps.to(torch.float32).clone()
        else:
            episode_len = torch.full(
                (self.num_envs,),
                self.elapsed_steps,
                dtype=torch.float32,
                device=self.device,
            )
        infos["episode"] = {
            "success_once": self.success_once.clone(),
            "return": self.returns.clone(),
            "episode_len": episode_len,
            "reward": self.returns.clone() / episode_len.clamp_min(1.0),
        }
        return infos

    def _apply_success_reward_latch(self, chunk_rewards):
        if not self.latch_success_reward:
            return chunk_rewards
        threshold = float(getattr(self.cfg, "success_reward_threshold", 0.9))
        latched_rewards = torch.empty_like(chunk_rewards)
        for frame_idx in range(chunk_rewards.shape[1]):
            self.reward_success_latched |= chunk_rewards[:, frame_idx] >= threshold
            latched_rewards[:, frame_idx] = self.reward_success_latched.to(
                chunk_rewards.dtype
            )
        return latched_rewards

    def _split_views(self, full_image):
        if self.wovr_view_mode == "single":
            main = full_image
            return main, torch.zeros_like(main), torch.zeros_like(main)

        expected_height = self.view_height * 3
        if full_image.shape[1:3] != (expected_height, self.view_width):
            raise ValueError(
                f"Expected generated image {(expected_height, self.view_width)}, got "
                f"{tuple(full_image.shape[1:3])}"
            )
        views = [
            full_image[:, idx * self.view_height : (idx + 1) * self.view_height]
            for idx in range(3)
        ]
        return views[0], views[1], views[2]

    def _prepare_vla_image(self, image):
        source_hw = tuple(int(value) for value in image.shape[1:3])
        expected_hw = (self.view_height, self.view_width)
        if source_hw != expected_hw:
            raise ValueError(
                f"WoVR VLA view expects generated size {expected_hw}, got {source_hw}"
            )
        target_hw = (240, 320) if self.is_robotwin_env else (224, 224)
        if source_hw == target_hw:
            return image.contiguous()
        image = image.permute(0, 3, 1, 2).float()
        image = F.interpolate(
            image, size=target_hw, mode="bilinear", align_corners=False
        )
        return (
            image.permute(0, 2, 3, 1)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .contiguous()
        )

    def _wrap_obs(self):
        last_frame = self.current_obs[:, :, 0, -1].permute(0, 2, 3, 1)
        full_image = ((last_frame + 1.0) * 127.5).round().clamp(0, 255)
        full_image = full_image.to(torch.uint8)
        main, left_wrist, right_wrist = self._split_views(full_image)

        main = self._prepare_vla_image(main)
        left_wrist = self._prepare_vla_image(left_wrist)
        right_wrist = self._prepare_vla_image(right_wrist)
        wrist_images = torch.stack([left_wrist, right_wrist], dim=1)

        return {
            "main_images": main,
            "wrist_images": wrist_images,
            "extra_view_images": None,
            "states": self.current_action14.to(
                device=self.device, dtype=torch.float32
            ).clone(),
            "task_descriptions": self.task_descriptions,
        }

    def _handle_auto_reset(self, dones, extracted_obs, infos):
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
        self.onload()
        actions_tensor = torch.as_tensor(
            policy_output_action, device=self.device, dtype=torch.float32
        )
        if actions_tensor.ndim != 3 or actions_tensor.shape[-1] != 14:
            raise ValueError(
                f"Expected actions [num_envs, chunk, 14], got {actions_tensor.shape}"
            )
        if actions_tensor.shape[0] != self.num_envs:
            raise ValueError(
                f"Actions batch {actions_tensor.shape[0]} does not match "
                f"num_envs {self.num_envs}"
            )
        if actions_tensor.shape[1] % self.chunk != 0:
            raise ValueError(
                f"Action chunk length {actions_tensor.shape[1]} must be a multiple "
                f"of world-model chunk {self.chunk}"
            )

        reward_chunks = []
        rm_reward_chunks = []
        termination_chunks = []
        truncation_chunks = []
        extracted_obs = None
        self.last_control_video_chunks = []

        action_chunk_len = int(actions_tensor.shape[1])
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

        for start in range(0, action_chunk_len, self.chunk):
            sub_actions = actions_tensor[:, start : start + self.chunk]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                self._infer_next_chunk_frames(
                    sub_actions,
                    active_env_mask=active_env_mask if task_specific_mode else None,
                )

            self.elapsed_steps += self.chunk
            extracted_obs = self._wrap_obs()
            if task_specific_mode and not active_env_mask.any():
                raw_rewards = torch.zeros(
                    self.num_envs,
                    self.chunk,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                raw_rewards = self._infer_next_chunk_rewards(sub_actions)
            rm_rewards = raw_rewards.clone()
            if task_specific_mode:
                rm_rewards = torch.where(
                    active_env_mask[:, None],
                    rm_rewards,
                    torch.zeros_like(rm_rewards),
                )
            rm_reward_chunks.append(rm_rewards)

            latched_rewards = self._apply_success_reward_latch(raw_rewards)
            if task_specific_mode:
                latched_rewards = torch.where(
                    active_env_mask[:, None],
                    latched_rewards,
                    torch.zeros_like(latched_rewards),
                )
            step_rewards = self._calc_step_reward(latched_rewards)
            if task_specific_mode:
                step_rewards = torch.where(
                    active_env_mask[:, None],
                    step_rewards,
                    torch.zeros_like(step_rewards),
                )
            reward_chunks.append(step_rewards)

            success = self._estimate_success_from_rewards(latched_rewards)
            terminations = torch.zeros(
                self.num_envs, self.chunk, dtype=torch.bool, device=self.device
            )
            truncations = torch.zeros_like(terminations)
            if task_specific_mode:
                success_in_action_chunk |= success & active_env_mask
            else:
                terminations[:, -1] = success
                if self.elapsed_steps >= self.cfg.max_episode_steps:
                    truncations[:, -1] = True
            termination_chunks.append(terminations)
            truncation_chunks.append(truncations)

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

        rewards = torch.cat(reward_chunks, dim=1)
        rm_rewards = torch.cat(rm_reward_chunks, dim=1)
        terminations = torch.cat(termination_chunks, dim=1)
        truncations = torch.cat(truncation_chunks, dim=1)
        past_terminations = terminations.any(dim=1)
        past_dones = past_terminations | truncations.any(dim=1)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}
        infos = self._record_metrics(
            rewards.sum(dim=1), past_terminations, infos
        )
        infos["stage"] = reward_group_ids_for_video
        infos["rm_reward"] = rm_rewards.max(dim=1).values
        infos["train_reward"] = rewards.sum(dim=1)
        success_threshold = float(
            getattr(self.cfg, "success_reward_threshold", 0.9)
        )
        infos["rm_success"] = (rm_rewards >= success_threshold).any(dim=1)

        return [extracted_obs], rewards, terminations, truncations, [infos]

    def offload(self):
        if self._is_offloaded:
            return
        self.pipe.dit = self.pipe.dit.to("cpu")
        self.pipe.vae = self.pipe.vae.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.current_action14 = self.current_action14.cpu()
        self.reference_action14 = self.reference_action14.cpu()
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
        self._is_offloaded = True
        torch.cuda.empty_cache()

    def onload(self):
        if not self._is_offloaded:
            return
        self.pipe.dit = self.pipe.dit.to(self.device)
        self.pipe.vae = self.pipe.vae.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.current_action14 = self.current_action14.to(self.device)
        self.reference_action14 = self.reference_action14.to(self.device)
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
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "current_action14": self.current_action14.cpu(),
            "reference_action14": self.reference_action14.cpu(),
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
