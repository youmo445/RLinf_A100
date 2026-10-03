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

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union


def _activate_wovr_diffsynth_backend() -> Path:
    """Select the WoVR DiffSynth implementation before importing WanEnv."""
    backend_root = Path(
        os.environ.get(
            "WOVR_DIFFSYNTH_PATH", "/data/zsq/diffsynth-studio-wovr-ppu"
        )
    ).expanduser().resolve()
    if not backend_root.is_dir():
        raise FileNotFoundError(
            "WoVR DiffSynth backend not found: "
            f"{backend_root}. Set WOVR_DIFFSYNTH_PATH to the correct checkout."
        )

    backend_str = str(backend_root)
    sys.path[:] = [entry for entry in sys.path if entry != backend_str]
    sys.path.insert(0, backend_str)

    loaded_diffsynth = sys.modules.get("diffsynth")
    if loaded_diffsynth is not None:
        expected_package = (backend_root / "diffsynth").resolve()
        loaded_paths = [
            Path(path).resolve()
            for path in getattr(loaded_diffsynth, "__path__", ())
        ]
        if expected_package not in loaded_paths:
            raise RuntimeError(
                "diffsynth was imported before the WoVR backend was selected. "
                f"Loaded paths: {loaded_paths}; expected: {expected_package}. "
                "Start WoVR in a fresh Python process."
            )
    return backend_root


_WOVR_DIFFSYNTH_ROOT = _activate_wovr_diffsynth_backend()

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.reward_model_compat import ResnetRewModel
from rlinf.envs.world_model.world_model_wan_env import WanEnv

import diffsynth


def _validate_wovr_diffsynth_backend() -> None:
    expected_package = (_WOVR_DIFFSYNTH_ROOT / "diffsynth").resolve()
    loaded_file = Path(diffsynth.__file__).resolve()
    if not loaded_file.is_relative_to(expected_package):
        raise RuntimeError(
            "WoVR loaded the wrong DiffSynth backend: "
            f"{loaded_file}; expected a module below {expected_package}."
        )


_validate_wovr_diffsynth_backend()

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

        if cfg.get("wm_env_type", "robotwin") != "robotwin":
            raise ValueError("WovrEnv currently supports wm_env_type=robotwin only")

        self.action_dim = int(cfg.get("action_dim", 14))
        if self.action_dim != 14:
            raise ValueError(f"WoVR RobotTwin expects action_dim=14, got {self.action_dim}")
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

        expected_height = self.view_height * (
            3 if self.wovr_view_mode == "three_vertical" else 1
        )
        if self.image_size != (expected_height, self.view_width):
            raise ValueError(
                f"{self.wovr_view_mode} WoVR expects image_size="
                f"[{expected_height}, {self.view_width}], got {self.image_size}"
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

    def _build_pipeline(self):
        for config_key in ("model_path", "VAE_path"):
            checkpoint_path = Path(str(self.cfg.get(config_key, ""))).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"WoVR {config_key} does not exist: {checkpoint_path}"
                )

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

            initial_action = first_frame.get("action")
            if initial_action is None:
                initial_action = first_frame.get("observation.state")
            if initial_action is None:
                raise ValueError(
                    f"Episode {episode_idx} has neither action nor observation.state"
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
        self._reset_metrics()
        return self._wrap_obs(), {}

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
    def _infer_next_chunk_frames(self, actions):
        actions_tensor = torch.as_tensor(
            actions, device=self.device, dtype=torch.float32
        )
        expected_shape = (self.num_envs, self.chunk, 14)
        if tuple(actions_tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected actions {expected_shape}, got {tuple(actions_tensor.shape)}"
            )

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
        for env_idx in range(self.num_envs):
            frames = [self._frame_to_pil(frame) for frame in self.image_queue[env_idx]]
            reference_images.append(frames[0])
            context_images.append(frames[-4:])

        output = self.pipe(
            seed=int(self.cfg.get("wm_seed", 0)),
            tiled=bool(self.cfg.get("tiled", False)),
            input_image=reference_images,
            input_image4=context_images,
            action=action_window,
            height=int(self.image_size[0]),
            width=int(self.image_size[1]),
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            cfg_scale=float(self.cfg.get("cfg_scale", 1.0)),
            sigma_shift=float(self.cfg.get("sigma_shift", 5.0)),
            progress_bar_cmd=lambda x: x,
            batch_size=self.num_envs,
            bs_1=False,
        )
        if len(output) != self.num_envs:
            raise RuntimeError(
                f"WoVR pipeline returned batch {len(output)}, expected {self.num_envs}"
            )

        predicted_chunks = []
        for env_idx, output_frames in enumerate(output):
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
            predicted_chunks.append(video[:, -self.chunk :])

        self.condition_action.zero_()
        self.condition_action[:, 1:] = actions_tensor[:, -4:]
        self.current_action14 = actions_tensor[:, -1].clone()

        predicted = torch.stack(predicted_chunks, dim=0).unsqueeze(2)
        self.current_obs = torch.cat([self.current_obs, predicted], dim=3)
        self.current_obs = self.current_obs[:, :, :, -self.num_frames :]

    def _load_reward_frames(self):
        frames = self.current_obs[:, :, :, -self.chunk :]
        return frames.permute(0, 3, 1, 2, 4, 5).reshape(
            self.num_envs * self.chunk,
            3,
            1,
            self.image_size[0],
            self.image_size[1],
        ).squeeze(2)

    @torch.no_grad()
    def _infer_next_chunk_rewards(self):
        reward_frames = self._load_reward_frames().to(self.device)
        reward_type = self.cfg.reward_model.type
        if reward_type == "InstructionRoutedResnetRewModel":
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

    @staticmethod
    def _prepare_robotwin_vla_image(image):
        if tuple(image.shape[1:3]) != (256, 320):
            raise ValueError(
                f"RoboTwin VLA view expects (256, 320), got {tuple(image.shape[1:3])}"
            )
        image = image.permute(0, 3, 1, 2).float()
        image = F.interpolate(
            image, size=(240, 320), mode="bilinear", align_corners=False
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

        main = self._prepare_robotwin_vla_image(main)
        left_wrist = self._prepare_robotwin_vla_image(left_wrist)
        right_wrist = self._prepare_robotwin_vla_image(right_wrist)
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
        termination_chunks = []
        truncation_chunks = []
        extracted_obs = None
        self.last_control_video_chunks = []

        for start in range(0, actions_tensor.shape[1], self.chunk):
            sub_actions = actions_tensor[:, start : start + self.chunk]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                self._infer_next_chunk_frames(sub_actions)

            self.elapsed_steps += self.chunk
            extracted_obs = self._wrap_obs()
            raw_rewards = self._infer_next_chunk_rewards()
            latched_rewards = self._apply_success_reward_latch(raw_rewards)
            step_rewards = self._calc_step_reward(latched_rewards)
            reward_chunks.append(step_rewards)

            success = self._estimate_success_from_rewards(latched_rewards)
            terminations = torch.zeros(
                self.num_envs, self.chunk, dtype=torch.bool, device=self.device
            )
            terminations[:, -1] = success
            truncations = torch.zeros_like(terminations)
            if self.elapsed_steps >= self.cfg.max_episode_steps:
                truncations[:, -1] = True
            termination_chunks.append(terminations)
            truncation_chunks.append(truncations)

        rewards = torch.cat(reward_chunks, dim=1)
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
            "prev_step_reward": self.prev_step_reward.cpu(),
            "reward_success_latched": self.reward_success_latched.cpu(),
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


def _parse_main_args():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Run a dataset-action smoke rollout through a WoVR environment."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=repo_root / "examples/embodiment/config",
    )
    parser.add_argument(
        "--config-name",
        default="wovr_robotwin_dual_task_grpo_pi05_head_only_no_state",
        help="Single-view or three-view WoVR Hydra config, with or without .yaml.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--action-source",
        choices=("dataset", "hold", "random"),
        default="dataset",
    )
    parser.add_argument("--action-start-frame", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Number of generated frames; use 0 for the rest of the episode.",
    )
    parser.add_argument("--random-scale", type=float, default=0.02)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--skip-reward", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs/wovr_env_smoke",
    )
    return parser.parse_args()


def _main_tensor_video_to_uint8(frames):
    frames = frames.detach().float().cpu()
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(f"Expected [3,T,H,W], got {tuple(frames.shape)}")
    frames = frames.permute(1, 2, 3, 0)
    return ((frames + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()


def _main_annotate_frame(frame, frame_idx, reward, view_name):
    import cv2

    frame = np.ascontiguousarray(frame.copy())
    color = (40, 220, 40) if reward >= 0.5 else (255, 110, 70)
    text = f"{view_name} frame={frame_idx:04d} reward={reward:.3f}"
    cv2.rectangle(frame, (4, 4), (min(frame.shape[1] - 4, 380), 36), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def _main_split_views(frame, view_mode):
    if view_mode == "single":
        return {"head": frame}
    if frame.shape[0] % 3 != 0:
        raise ValueError(f"Three-view frame height is not divisible by 3: {frame.shape}")
    view_height = frame.shape[0] // 3
    names = ("head", "left_wrist", "right_wrist")
    return {
        name: frame[idx * view_height : (idx + 1) * view_height]
        for idx, name in enumerate(names)
    }


def _main_episode_actions(env, episode_index, action_key):
    trajectory = np.load(env.dataset.npy_files[episode_index], allow_pickle=True)
    values = []
    for frame_idx, frame in enumerate(trajectory):
        value = frame.get(action_key, frame.get("action", frame.get("state")))
        if value is None:
            raise KeyError(
                f"Episode {episode_index} frame {frame_idx} has no {action_key!r}, "
                "'action', or 'state' value"
            )
        values.append(np.asarray(value, dtype=np.float32).reshape(-1))
    actions = np.stack(values, axis=0)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected episode actions [T,14], got {actions.shape}")
    return actions


@torch.no_grad()
def _main():
    import imageio.v2 as imageio
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    args = _parse_main_args()
    config_dir = args.config_dir.resolve()
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    os.environ.setdefault("REPO_PATH", str(Path(__file__).resolve().parents[3]))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        full_cfg = compose(config_name=args.config_name.removesuffix(".yaml"))

    with open_dict(full_cfg):
        cfg = full_cfg.env.train
        if cfg.env_type != "wovr_wm":
            raise ValueError(f"Expected env_type='wovr_wm', got {cfg.env_type!r}")
        cfg.total_num_envs = 1
        cfg.group_size = 1
        cfg.auto_reset = False
        cfg.use_fixed_reset_state_ids = False
        cfg.video_cfg.save_video = False
        cfg.seed = args.seed
        cfg.wm_seed = args.seed
        if args.num_inference_steps is not None:
            cfg.num_inference_steps = int(args.num_inference_steps)
        if args.model_path is not None:
            cfg.model_path = args.model_path
        if args.skip_reward:
            cfg.reward_model.type = "DummyRewardModel"

    env = WovrEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1)
    if not 0 <= args.episode_index < len(env.dataset):
        raise ValueError(
            f"episode-index {args.episode_index} is outside [0,{len(env.dataset) - 1}]"
        )

    obs, _ = env.reset(
        episode_indices=np.asarray([args.episode_index], dtype=np.int64)
    )
    del obs
    episode_actions = _main_episode_actions(
        env, args.episode_index, str(cfg.get("dataset_action_key", "action"))
    )
    if args.action_source == "dataset":
        start = int(args.action_start_frame)
        if start < 1 or start >= len(episode_actions):
            raise ValueError(
                f"action-start-frame must be in [1,{len(episode_actions) - 1}], "
                f"got {start}"
            )
        actions = episode_actions[start:]
        first_frame_idx = start
    else:
        num_steps = args.max_steps if args.max_steps > 0 else env.chunk
        initial = env.current_action14[0].detach().float().cpu().numpy()
        if args.action_source == "hold":
            actions = np.repeat(initial[None], num_steps, axis=0)
        else:
            rng = np.random.default_rng(args.seed)
            actions = initial[None] + rng.normal(
                0.0, args.random_scale, size=(num_steps, 14)
            ).astype(np.float32)
        first_frame_idx = 1
    if args.max_steps > 0:
        actions = actions[: args.max_steps]
    if len(actions) == 0:
        raise ValueError("No actions selected for rollout")
    cfg.max_episode_steps = max(int(len(actions)), env.chunk)

    frames = [_main_tensor_video_to_uint8(env.current_obs[0, :, 0, -1:])[0]]
    rewards = [0.0]
    frame_indices = [first_frame_idx - 1]
    executed_actions = []
    num_chunks = int(np.ceil(len(actions) / env.chunk))
    for chunk_idx in range(num_chunks):
        start = chunk_idx * env.chunk
        valid = actions[start : start + env.chunk]
        valid_len = len(valid)
        padded = valid
        if valid_len < env.chunk:
            padded = np.concatenate(
                [valid, np.repeat(valid[-1:], env.chunk - valid_len, axis=0)], axis=0
            )
        _, chunk_rewards, terminations, truncations, _ = env.chunk_step(
            padded[None]
        )
        chunk_frames = _main_tensor_video_to_uint8(
            env.current_obs[0, :, 0, -env.chunk:]
        )[:valid_len]
        reward_values = chunk_rewards[0, :valid_len].detach().float().cpu().numpy()
        frames.extend(list(chunk_frames))
        rewards.extend(float(value) for value in reward_values)
        frame_indices.extend(range(first_frame_idx + start, first_frame_idx + start + valid_len))
        executed_actions.append(valid.astype(np.float32))
        print(
            f"chunk {chunk_idx + 1}/{num_chunks}: valid={valid_len} "
            f"reward={reward_values.tolist()} "
            f"done={bool(terminations.any() or truncations.any())}"
        )

    if not (len(frames) == len(frame_indices) == len(rewards)):
        raise RuntimeError(
            "WoVR smoke output lengths diverged: "
            f"frames={len(frames)} indices={len(frame_indices)} rewards={len(rewards)}"
        )

    output_dir = args.output_dir / (
        f"{env.wovr_view_mode}_episode_{args.episode_index:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(full_cfg, output_dir / "resolved_config.yaml")
    annotated_full = [
        _main_annotate_frame(frame, frame_idx, reward, env.wovr_view_mode)
        for frame, frame_idx, reward in zip(frames, frame_indices, rewards)
    ]
    imageio.mimsave(
        output_dir / "wovr_rollout.mp4",
        annotated_full,
        fps=args.fps,
        macro_block_size=1,
    )
    for view_name in _main_split_views(frames[0], env.wovr_view_mode):
        view_frames = []
        for frame, frame_idx, reward in zip(frames, frame_indices, rewards):
            view = _main_split_views(frame, env.wovr_view_mode)[view_name]
            view_frames.append(_main_annotate_frame(view, frame_idx, reward, view_name))
        imageio.mimsave(
            output_dir / f"{view_name}.mp4",
            view_frames,
            fps=args.fps,
            macro_block_size=1,
        )
    all_actions = np.concatenate(executed_actions, axis=0)
    np.save(output_dir / "actions.npy", all_actions)
    (output_dir / "rollout.json").write_text(
        json.dumps(
            {
                "config_name": args.config_name,
                "view_mode": env.wovr_view_mode,
                "episode_index": args.episode_index,
                "action_source": args.action_source,
                "num_inference_steps": int(env.num_inference_steps),
                "frame_indices": frame_indices,
                "rewards": rewards,
                "num_generated_frames": len(frames) - 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    _main()
