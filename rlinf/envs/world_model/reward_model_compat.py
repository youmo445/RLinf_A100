"""Reward-model compatibility helpers independent of a DiffSynth checkout."""

import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = (
            nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
            if use_1x1conv
            else None
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return F.relu(y + x)


def _resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blocks = []
    for block_idx in range(num_residuals):
        if block_idx == 0 and not first_block:
            blocks.append(
                Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blocks.append(Residual(num_channels, num_channels))
    return blocks


def _checkpoint_dummy_train(*args, **kwargs):
    return None


class _RewardCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "train" and module in {
            "__main__",
            "train_robotwin_rm_multidataset",
        }:
            return _checkpoint_dummy_train
        return super().find_class(module, name)


class _RewardCheckpointPickleModule:
    __name__ = "pickle"
    Unpickler = _RewardCheckpointUnpickler
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL

    @staticmethod
    def load(file, **kwargs):
        return _RewardCheckpointUnpickler(file, **kwargs).load()


def _torch_load_reward_checkpoint(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (AttributeError, ModuleNotFoundError) as exc:
        is_missing_train = (
            "train" in str(exc)
            or "train_robotwin_rm_multidataset" in str(exc)
        )
        if not is_missing_train:
            raise
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            pickle_module=_RewardCheckpointPickleModule,
        )


class ResnetRewModel(nn.Module):
    """Binary ResNet reward model matching the current WanmultiviewEnv behavior."""

    def __init__(self, checkpoint_path=None, threshold=None):
        super().__init__()
        self.threshold = 0.5
        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        stage2 = nn.Sequential(*_resnet_block(64, 64, 2, first_block=True))
        stage3 = nn.Sequential(*_resnet_block(64, 128, 2))
        stage4 = nn.Sequential(*_resnet_block(128, 256, 2))
        stage5 = nn.Sequential(*_resnet_block(256, 512, 2))
        self.net = nn.Sequential(
            stem,
            stage2,
            stage3,
            stage4,
            stage5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        if threshold is not None:
            self.threshold = float(threshold)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = _torch_load_reward_checkpoint(checkpoint_path)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict)

        if isinstance(checkpoint, dict):
            threshold = checkpoint.get("threshold")
            if threshold is None and isinstance(checkpoint.get("metrics"), dict):
                threshold = checkpoint["metrics"].get("best_threshold")
            if threshold is not None:
                self.threshold = float(threshold)

    @torch.no_grad()
    def predict_score(self, obs):
        obs = obs.clamp(-1.0, 1.0)
        return self.net(obs.to(dtype=torch.float32))

    @torch.no_grad()
    def predict_rew(self, obs, threshold=None):
        scores = self.predict_score(obs)
        threshold = self.threshold if threshold is None else float(threshold)
        return (scores >= threshold).to(dtype=scores.dtype)

    def forward(self, obs=None):
        return self.predict_rew(obs)


class InstructionRoutedGripperTintResnetRewModel(nn.Module):
    """Route gripper-tinted head frames to task-specific reward models."""

    def __init__(self, checkpoint_root, task_routes, threshold=None):
        super().__init__()
        checkpoint_root = Path(checkpoint_root)
        task_routes = dict(task_routes)
        if not task_routes:
            raise ValueError(
                "InstructionRoutedGripperTintResnetRewModel requires "
                "reward_model.task_routes"
            )

        self.models = nn.ModuleDict()
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
            self.models[model_key] = ResnetRewModel(
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
