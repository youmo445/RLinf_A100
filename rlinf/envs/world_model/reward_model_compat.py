"""Reward-model compatibility helpers independent of a DiffSynth checkout."""

import pickle

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
        if module == "__main__" and name == "train":
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
    except AttributeError as exc:
        if "Can't get attribute 'train'" not in str(exc):
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
