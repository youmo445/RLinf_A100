# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Ctrl-World backed multi-view world-model environment.

This module keeps the RLinf Wan multi-view environment interface and replaces
only the video generator with Ctrl-World.  The inherited class still owns reset,
reward evaluation, observation wrapping, metrics, auto-reset, and action-chunk
splitting.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rlinf.envs.utils import recursive_to_device
from rlinf.envs.world_model.world_model_wan_multi_view_env import WanmultiviewEnv

__all__ = ["CtrlWorldEnv"]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _plain(value: Any) -> Any:
    if value.__class__.__name__ in {"ListConfig", "DictConfig"}:
        try:
            from omegaconf import OmegaConf

            return OmegaConf.to_container(value, resolve=True)
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = str(value).lower()
    if name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if name in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported Ctrl-World dtype: {value}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class CtrlWorldEnv(WanmultiviewEnv):
    """RLinf multi-view WM environment using Ctrl-World as the generator."""

    def _build_pipeline(self):
        self._init_ctrlworld_runtime_config()
        ctrlworld_root = Path(self.ctrlworld_path).expanduser().resolve()
        if not ctrlworld_root.exists():
            raise FileNotFoundError(
                "Ctrl-World repo not found. Set cfg.ctrlworld_path or CTRLWORLD_PATH. "
                f"Current value: {ctrlworld_root}"
            )
        if str(ctrlworld_root) not in sys.path:
            sys.path.insert(0, str(ctrlworld_root))

        from dataset.robotwin_fk import RoboTwinFK
        from models.ctrl_world import CrtlWorld

        args = self._build_ctrlworld_args(ctrlworld_root)
        ckpt_path = self._resolve_ctrlworld_checkpoint(args)
        args.ckpt_path = str(ckpt_path)

        self.ctrlworld_args = args
        self.ctrlworld_dtype = _torch_dtype(_cfg_get(self.cfg, "ctrlworld_dtype", args.dtype))
        self.ctrlworld_view_height = int(args.height)
        self.ctrlworld_view_width = int(args.width)
        self.ctrlworld_num_history = int(args.num_history)
        self.ctrlworld_num_frames = int(args.num_frames)
        self.ctrlworld_action_dim = int(args.action_dim)
        self.ctrlworld_history_skip = int(args.robotwin_history_skip)
        self.ctrlworld_future_skip = int(args.robotwin_future_skip)
        self.ctrlworld_skip_first_pred_frame = _as_bool(
            _cfg_get(self.cfg, "ctrlworld_skip_first_pred_frame", False)
        )

        if self.ctrlworld_action_dim != 16:
            raise ValueError(f"Ctrl-World RobotWin action_dim must be 16, got {self.ctrlworld_action_dim}")
        if self.ctrlworld_num_frames < self.chunk:
            raise ValueError(
                f"Ctrl-World num_frames={self.ctrlworld_num_frames} must be >= wm chunk={self.chunk}"
            )

        print(f"[CtrlWorldEnv] loading Ctrl-World checkpoint: {ckpt_path}")
        model = CrtlWorld(args)
        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = self._unwrap_state_dict(state_dict)
        model.load_state_dict(
            state_dict,
            strict=_as_bool(_cfg_get(self.cfg, "ctrlworld_strict_load", True)),
        )
        model.to(self.device).to(self.ctrlworld_dtype)
        model.eval()

        self.ctrlworld_model = model
        self.ctrlworld_fk = RoboTwinFK(str(args.robotwin_fk_config_path))
        self.ctrlworld_state_p01, self.ctrlworld_state_p99 = self._load_ctrlworld_action_stats(
            Path(args.robotwin_stat_path)
        )
        self.ctrlworld_state_p01 = self.ctrlworld_state_p01.astype(np.float32)
        self.ctrlworld_state_p99 = self.ctrlworld_state_p99.astype(np.float32)
        return model

    def _init_ctrlworld_runtime_config(self) -> None:
        self.ctrlworld_path = _plain(
            _cfg_get(self.cfg, "ctrlworld_path", os.environ.get("CTRLWORLD_PATH", "/data/zsq/Ctrl-World"))
        )

    def _load_ctrlworld_config_module(self, ctrlworld_root: Path):
        config_path = Path(_cfg_get(self.cfg, "ctrlworld_config_py", ctrlworld_root / "config.py"))
        if not config_path.is_absolute():
            config_path = ctrlworld_root / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Ctrl-World config.py not found: {config_path}")
        spec = importlib.util.spec_from_file_location("_ctrlworld_runtime_config", str(config_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Ctrl-World config module from {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _build_ctrlworld_args(self, ctrlworld_root: Path) -> SimpleNamespace:
        module = self._load_ctrlworld_config_module(ctrlworld_root)
        args_cls = getattr(module, "wm_args", None)
        args = args_cls() if callable(args_cls) else SimpleNamespace()

        def set_attr(name: str, default: Any) -> None:
            value = _cfg_get(self.cfg, f"ctrlworld_{name}", getattr(args, name, default))
            setattr(args, name, _plain(value))

        set_attr("svd_model_path", getattr(args, "svd_model_path", None))
        set_attr("clip_model_path", getattr(args, "clip_model_path", None))
        set_attr("robotwin_fk_config_path", getattr(args, "robotwin_fk_config_path", None))
        set_attr("robotwin_stat_path", getattr(args, "robotwin_stat_path", None))
        set_attr("robotwin_camera_names", "head_camera,left_camera,right_camera")
        set_attr("width", 320)
        set_attr("height", 240)
        set_attr("num_frames", self.chunk + 1)
        set_attr("num_history", 6)
        set_attr("action_dim", 16)
        set_attr("num_inference_steps", _cfg_get(self.cfg, "num_inference_steps", 50))
        set_attr("decode_chunk_size", 7)
        set_attr("guidance_scale", 1.0)
        set_attr("fps", 30)
        set_attr("motion_bucket_id", 127)
        set_attr("frame_level_cond", True)
        set_attr("his_cond_zero", False)
        set_attr("text_cond", True)
        set_attr("dtype", torch.bfloat16)
        set_attr("vae_encode_batch_size", 64)
        set_attr("robotwin_history_skip", 4)
        set_attr("robotwin_future_skip", 1)

        args.ckpt_path = _plain(_cfg_get(self.cfg, "ctrlworld_ckpt_path", getattr(args, "ckpt_path", None)))
        args.ckpt_dir = _plain(_cfg_get(self.cfg, "ctrlworld_ckpt_dir", getattr(args, "ckpt_dir", None)))

        env_overrides = {
            "svd_model_path": "CTRLWORLD_SVD_MODEL_PATH",
            "clip_model_path": "CTRLWORLD_CLIP_MODEL_PATH",
            "robotwin_fk_config_path": "CTRLWORLD_ROBOTWIN_FK_CONFIG_PATH",
            "robotwin_stat_path": "CTRLWORLD_ROBOTWIN_STAT_PATH",
            "ckpt_path": "CTRLWORLD_CKPT_PATH",
            "ckpt_dir": "CTRLWORLD_CKPT_DIR",
        }
        for attr_name, env_name in env_overrides.items():
            env_value = os.environ.get(env_name)
            if env_value:
                setattr(args, attr_name, env_value)

        for path_attr in (
            "svd_model_path",
            "clip_model_path",
            "robotwin_fk_config_path",
            "robotwin_stat_path",
        ):
            value = getattr(args, path_attr, None)
            if value is None:
                continue
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = ctrlworld_root / path
            setattr(args, path_attr, str(path))

        return args

    def _resolve_ctrlworld_checkpoint(self, args: SimpleNamespace) -> Path:
        ckpt = getattr(args, "ckpt_path", None) or getattr(args, "ckpt_dir", None)
        if ckpt is None:
            raise ValueError("Set cfg.ctrlworld_ckpt_path or cfg.ctrlworld_ckpt_dir")
        path = Path(str(ckpt)).expanduser()
        if not path.is_absolute():
            path = Path(self.ctrlworld_path).expanduser().resolve() / path
        if path.is_file():
            return path
        if path.is_dir():
            checkpoints = sorted(
                path.glob("checkpoint-*.pt"),
                key=lambda p: int(p.stem.split("-")[-1]) if p.stem.split("-")[-1].isdigit() else -1,
            )
            if not checkpoints:
                checkpoints = sorted(path.glob("*.pt"))
            if checkpoints:
                return checkpoints[-1]
        raise FileNotFoundError(f"Ctrl-World checkpoint not found: {path}")

    def _unwrap_state_dict(self, state_dict: Any) -> dict:
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module"):
                value = state_dict.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported Ctrl-World checkpoint type: {type(state_dict)}")
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {
                (k[len("module.") :] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
        return state_dict

    def _load_ctrlworld_action_stats(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.is_absolute():
            path = Path(self.ctrlworld_path).expanduser().resolve() / path
        if not path.exists():
            raise FileNotFoundError(f"Ctrl-World RobotWin action stat not found: {path}")
        data = _read_json(path)
        return (
            np.asarray(data["state_01"], dtype=np.float32)[None, :],
            np.asarray(data["state_99"], dtype=np.float32)[None, :],
        )

    def _normalize_ctrlworld_action(self, action16: np.ndarray) -> np.ndarray:
        eps = 1e-8
        normed = 2 * (action16 - self.ctrlworld_state_p01) / (
            self.ctrlworld_state_p99 - self.ctrlworld_state_p01 + eps
        ) - 1
        return np.clip(normed, -1.0, 1.0).astype(np.float32)

    def _action14_to_ctrlworld_action16(self, action14: np.ndarray) -> np.ndarray:
        action14 = np.asarray(action14, dtype=np.float32)
        if action14.ndim != 2 or action14.shape[-1] != 14:
            raise ValueError(f"Expected action14 [T, 14], got {action14.shape}")
        left_arm = action14[:, 0:6]
        left_gripper = action14[:, 6]
        right_arm = action14[:, 7:13]
        right_gripper = action14[:, 13]
        return self.ctrlworld_fk.abs_actions_from_joint_action(
            left_arm, left_gripper, right_arm, right_gripper
        ).astype(np.float32)

    def _frame_to_uint8_stacked(self, frame_tensor: torch.Tensor) -> np.ndarray:
        frame = frame_tensor.detach().float().cpu()
        if frame.ndim == 4:
            frame = frame[:, 0]
        if frame.ndim != 3:
            raise ValueError(f"Expected frame tensor [3, H, W] or [3, 1, H, W], got {frame_tensor.shape}")
        frame = frame.permute(1, 2, 0)
        frame = ((frame + 1.0) * 0.5 * 255.0).clamp(0, 255)
        return frame.numpy().astype(np.uint8)

    def _resize_view_uint8(self, view: np.ndarray, width: int, height: int) -> np.ndarray:
        if view.shape[0] == height and view.shape[1] == width:
            return view
        return np.asarray(Image.fromarray(view).resize((width, height), Image.BILINEAR), dtype=np.uint8)

    def _ctrlworld_pixel_values_from_frames(self, frames: Iterable[torch.Tensor]) -> torch.Tensor:
        views_per_frame = []
        expected_height = self.view_height * len(self.camera_names)
        for frame in frames:
            stacked = self._frame_to_uint8_stacked(frame)
            if stacked.shape[0] != expected_height:
                raise ValueError(
                    f"Stacked multiview frame height {stacked.shape[0]} does not match "
                    f"{expected_height}={self.view_height}*{len(self.camera_names)}"
                )
            views = []
            for view_idx in range(len(self.camera_names)):
                view = stacked[
                    view_idx * self.view_height : (view_idx + 1) * self.view_height,
                    : self.view_width,
                ]
                views.append(
                    self._resize_view_uint8(
                        view,
                        self.ctrlworld_view_width,
                        self.ctrlworld_view_height,
                    )
                )
            views_per_frame.append(np.stack(views, axis=0))
        pixel_values = np.stack(views_per_frame, axis=0)[None]  # [1, T, V, H, W, C]
        return torch.from_numpy(pixel_values).to(torch.uint8)

    @torch.no_grad()
    def _encode_ctrlworld_frames(self, frames: Iterable[torch.Tensor]) -> torch.Tensor:
        pixel_values = self._ctrlworld_pixel_values_from_frames(frames)
        latents = self.ctrlworld_model.get_batch_latents({"pixel_values": pixel_values})
        return latents.to(device=self.device, dtype=self.ctrlworld_dtype)

    @torch.no_grad()
    def _decode_ctrlworld_latents_to_env_frames(self, latents: torch.Tensor) -> torch.Tensor:
        view_count = len(self.camera_names)
        bsz, frame_num, channels, latent_h_total, latent_w = latents.shape
        if bsz != 1:
            raise ValueError("CtrlWorldEnv currently decodes one env at a time")
        if latent_h_total % view_count != 0:
            raise ValueError(
                f"Latent height {latent_h_total} is not divisible by view_count={view_count}"
            )
        per_view_latent_h = latent_h_total // view_count
        latents = latents.reshape(
            bsz, frame_num, channels, view_count, per_view_latent_h, latent_w
        )
        latents = latents.permute(0, 3, 1, 2, 4, 5).reshape(
            bsz * view_count, frame_num, channels, per_view_latent_h, latent_w
        )

        x = latents.flatten(0, 1)
        decoded = []
        decode_kwargs = {}
        decode_chunk_size = int(self.ctrlworld_args.decode_chunk_size)
        vae = self.ctrlworld_model.pipeline.vae
        for start in range(0, x.shape[0], decode_chunk_size):
            chunk = x[start : start + decode_chunk_size] / vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded.append(vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded, dim=0)
        videos = videos.reshape(bsz * view_count, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255.0)
        videos = videos.detach().to(torch.float32).cpu().numpy()
        videos = videos.transpose(0, 1, 3, 4, 2).astype(np.uint8)  # [V, T, H, W, C]

        stacked_frames = []
        for frame_idx in range(frame_num):
            resized_views = [
                self._resize_view_uint8(
                    videos[view_idx, frame_idx],
                    self.view_width,
                    self.view_height,
                )
                for view_idx in range(view_count)
            ]
            stacked_frames.append(np.concatenate(resized_views, axis=0))
        stacked = np.stack(stacked_frames, axis=0)  # [T, H_total, W, C]
        tensor = torch.from_numpy(stacked).float() / 255.0 * 2.0 - 1.0
        return tensor.permute(3, 0, 1, 2).contiguous()  # [3, T, H_total, W]

    def _wrap_obs(self):
        """Expose Ctrl-World 240x320 per-view frames directly to the VLA."""
        b, c, v, t, h, w = self.current_obs.shape
        if b != self.num_envs or c != 3 or v != 1:
            raise ValueError(
                f"Unexpected current_obs shape: {self.current_obs.shape}; "
                f"expected [{self.num_envs}, 3, 1, T, H, W]"
            )

        expected_height = self.view_height * len(self.camera_names)
        expected_hw = (expected_height, self.view_width)
        if (h, w) != expected_hw:
            raise ValueError(
                f"Ctrl-World current_obs expects stacked image size {expected_hw}, "
                f"got {(h, w)}. Check image_size/view_height/view_width."
            )

        last_frame = self.current_obs[:, :, 0, -1, :, :]
        full_image = last_frame.permute(0, 2, 3, 1)
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255).to(torch.uint8)

        view_images = {
            camera_name: full_image[
                :, view_idx * self.view_height : (view_idx + 1) * self.view_height
            ].contiguous()
            for view_idx, camera_name in enumerate(self.camera_names)
        }
        main_image = view_images.get("head")
        if main_image is None:
            main_image = full_image[:, : self.view_height].contiguous()
        left_wrist_image = view_images.get("left_wrist", torch.zeros_like(main_image))
        right_wrist_image = view_images.get("right_wrist", torch.zeros_like(main_image))
        wrist_images = torch.stack(
            [left_wrist_image.contiguous(), right_wrist_image.contiguous()], dim=1
        )

        states = self.current_action14.to(device=self.device, dtype=torch.float32).clone()
        task_descriptions = getattr(
            self,
            "task_descriptions",
            [self.prompt] * self.num_envs,
        )
        return {
            "main_images": main_image.contiguous(),
            "wrist_images": wrist_images.contiguous(),
            "extra_view_images": None,
            "states": states,
            "task_descriptions": task_descriptions,
        }

    def _history_frame_ids(self, env_idx: int) -> list[int]:
        history_len = len(self.frame_history[env_idx])
        if history_len <= 0:
            raise ValueError(f"Frame history for env {env_idx} is empty")
        last_idx = history_len - 1
        return [
            max(0, last_idx - i * self.ctrlworld_history_skip)
            for i in range(self.ctrlworld_num_history, 0, -1)
        ]

    def _build_ctrlworld_window(self, env_idx: int, actions_tensor: torch.Tensor):
        frame_history = self.frame_history[env_idx]
        action_history = self.action_history14[env_idx]
        if not frame_history:
            raise ValueError(f"Frame history for env {env_idx} is empty")
        if not action_history:
            action_history = [
                self.reference_action14[env_idx].detach().cpu().numpy().astype(np.float32)
            ]
            self.action_history14[env_idx] = action_history

        history_ids = self._history_frame_ids(env_idx)
        history_frames = [frame_history[min(idx, len(frame_history) - 1)].clone() for idx in history_ids]
        current_frame = frame_history[-1].clone()

        history_actions = [
            np.asarray(action_history[min(idx, len(action_history) - 1)], dtype=np.float32).reshape(14)
            for idx in history_ids
        ]
        current_action = self.current_action14[env_idx].detach().cpu().numpy().astype(np.float32).reshape(14)
        future_actions = [current_action]
        future_actions.extend(
            actions_tensor[env_idx].detach().cpu().numpy().astype(np.float32).reshape(self.chunk, 14)
        )
        while len(future_actions) < self.ctrlworld_num_frames:
            future_actions.append(future_actions[-1].copy())
        future_actions = future_actions[: self.ctrlworld_num_frames]

        action14_window = np.stack(history_actions + future_actions, axis=0)
        action16 = self._action14_to_ctrlworld_action16(action14_window)
        action16 = self._normalize_ctrlworld_action(action16)
        action = torch.from_numpy(action16[None]).to(
            device=self.device,
            dtype=self.ctrlworld_dtype,
        )
        return history_frames, current_frame, action

    @torch.no_grad()
    def _predict_ctrlworld_latent_window(
        self,
        action: torch.Tensor,
        text: list[str],
        current_latent: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        action_latent = self.ctrlworld_model.action_encoder(
            action,
            text,
            self.ctrlworld_model.tokenizer,
            self.ctrlworld_model.text_encoder,
            frame_level_cond=bool(self.ctrlworld_args.frame_level_cond),
        )
        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            self.ctrlworld_model.pipeline,
            image=current_latent,
            text=action_latent,
            width=int(self.ctrlworld_args.width),
            height=int(self.ctrlworld_args.height) * len(self.camera_names),
            num_frames=int(self.ctrlworld_args.num_frames),
            history=history,
            num_inference_steps=int(self.ctrlworld_args.num_inference_steps),
            decode_chunk_size=int(self.ctrlworld_args.decode_chunk_size),
            max_guidance_scale=float(self.ctrlworld_args.guidance_scale),
            fps=int(self.ctrlworld_args.fps),
            motion_bucket_id=int(self.ctrlworld_args.motion_bucket_id),
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=bool(self.ctrlworld_args.frame_level_cond),
            his_cond_zero=bool(self.ctrlworld_args.his_cond_zero),
        )
        return pred_latents.to(device=self.device, dtype=self.ctrlworld_dtype)

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions):
        """Predict the next WM chunk with Ctrl-World."""
        actions_tensor = (
            torch.from_numpy(actions).to(self.device)
            if isinstance(actions, np.ndarray)
            else actions.to(self.device)
        ).float()
        if actions_tensor.ndim != 3 or actions_tensor.shape != (self.num_envs, self.chunk, 14):
            raise ValueError(
                f"Expected actions [num_envs, {self.chunk}, 14], got {tuple(actions_tensor.shape)}"
            )

        all_samples = []
        last_control_video = []
        for env_idx in range(self.num_envs):
            history_frames, current_frame, action = self._build_ctrlworld_window(env_idx, actions_tensor)
            latents = self._encode_ctrlworld_frames([*history_frames, current_frame])
            history_latents = latents[:, : self.ctrlworld_num_history]
            current_latent = latents[:, self.ctrlworld_num_history]
            prompt = self.task_descriptions[env_idx] or self.prompt

            pred_latents = self._predict_ctrlworld_latent_window(
                action=action,
                text=[prompt],
                current_latent=current_latent,
                history=history_latents,
            )
            pred_video = self._decode_ctrlworld_latents_to_env_frames(pred_latents)
            if self.ctrlworld_skip_first_pred_frame:
                pred_video = pred_video[:, 1 : 1 + self.chunk]
            else:
                pred_video = pred_video[:, : self.chunk]
            if pred_video.shape[1] < self.chunk:
                raise ValueError(
                    f"Ctrl-World generated {pred_video.shape[1]} frames, expected {self.chunk}"
                )
            pred_video = pred_video[:, : self.chunk].to(self.device)
            all_samples.append(pred_video)

            actions_np = actions_tensor[env_idx].detach().cpu().numpy().astype(np.float32)
            for frame_idx in range(self.chunk):
                self.frame_history[env_idx].append(pred_video[:, frame_idx : frame_idx + 1].detach().clone())
                self.action_history14[env_idx].append(actions_np[frame_idx].reshape(14).copy())

            self.current_action14[env_idx] = actions_tensor[env_idx, -1].to(
                device=self.current_action14.device,
                dtype=self.current_action14.dtype,
            )

            reference_frame = self.reference_frames[env_idx]
            if reference_frame is None:
                reference_frame = self.current_obs[env_idx, :, 0, 0:1, :, :].detach().clone()
                self.reference_frames[env_idx] = reference_frame

            if self.rollout_context_mode == "sparse_history":
                sparse_frames, sparse_actions = self._sparse_history_context(env_idx)
                self.image_queue[env_idx] = sparse_frames
                self.condition_action[env_idx] = torch.from_numpy(sparse_actions).to(
                    device=self.device,
                    dtype=actions_tensor.dtype,
                )
            else:
                recent_count = max(0, self.condition_frame_length - 1)
                recent_frames = [
                    pred_video[:, t : t + 1].detach().clone()
                    for t in range(max(0, pred_video.shape[1] - recent_count), pred_video.shape[1])
                ]
                if len(recent_frames) < recent_count:
                    raise ValueError(
                        f"Ctrl-World generated {pred_video.shape[1]} frames; "
                        f"cannot build {recent_count} recent context frames"
                    )
                self.image_queue[env_idx] = [
                    reference_frame.detach().clone(),
                    *recent_frames[-recent_count:],
                ][: self.condition_frame_length]
                if self.condition_frame_length > 1:
                    recent_actions = actions_tensor[env_idx, -recent_count:].to(
                        device=self.device,
                        dtype=actions_tensor.dtype,
                    )
                    self.condition_action[env_idx] = torch.cat(
                        [
                            self.reference_action14[env_idx : env_idx + 1].to(
                                device=self.device,
                                dtype=actions_tensor.dtype,
                            ),
                            recent_actions,
                        ],
                        dim=0,
                    )
                else:
                    self.condition_action[env_idx] = self.current_action14[env_idx : env_idx + 1]

            blank = np.zeros(
                (
                    self.view_height * len(self.camera_names),
                    self.view_width,
                    3,
                ),
                dtype=np.uint8,
            )
            last_control_video.append([blank.copy() for _ in range(self.chunk)])

        x_samples = torch.stack(all_samples, dim=0).to(self.device).unsqueeze(2)
        self.current_obs = torch.cat([self.current_obs, x_samples], dim=3)
        self.last_control_video = last_control_video
        if hasattr(self, "last_control_video_chunks"):
            self.last_control_video_chunks.append(last_control_video)

        max_frames = self.condition_frame_length + self.chunk
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

    def offload(self):
        if self._is_offloaded:
            return
        self.pipe = self.pipe.to("cpu")
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = recursive_to_device(self.current_obs, "cpu")
        self.current_action14 = self.current_action14.cpu()
        self.condition_action = self.condition_action.cpu()
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reward_success_latched = self.reward_success_latched.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded:
            return
        self.pipe = self.pipe.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = recursive_to_device(self.current_obs, self.device)
        self.current_action14 = self.current_action14.to(self.device)
        self.condition_action = self.condition_action.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reward_success_latched = self.reward_success_latched.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False


def _tensor_frames_to_uint8(frames_tensor: torch.Tensor) -> np.ndarray:
    frames_tensor = frames_tensor.detach().float().cpu()
    frames_tensor = frames_tensor.permute(1, 2, 3, 0)
    frames_tensor = (frames_tensor + 1.0) / 2.0 * 255.0
    frames_tensor = torch.clamp(frames_tensor, 0, 255)
    return frames_tensor.numpy().astype(np.uint8)

# export PYTHONPATH=/data/zsq/diffsynth-studio-wujie-ppu:/data/zsq/RLinf_A100:$PYTHONPATH python -m rlinf.envs.world_model.world_model_ctrlworld_env
'''
 - diffusers==0.38.0
 + diffusers==0.34.0
'''
if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import initialize_config_dir
    from omegaconf import open_dict

    parser = argparse.ArgumentParser(description="Smoke rollout Ctrl-World inside the RLinf WM env API.")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.environ.setdefault("EMBODIED_PATH", "examples/embodiment")
    repo_root = Path(__file__).resolve().parents[3]
    config_dir = Path(
        args.config_dir
        or os.environ.get("EMBODIED_CONFIG_DIR", repo_root / "examples/embodiment/config")
    ).resolve()
    config_name = args.config_name or os.environ.get(
        "EMBODIED_CONFIG_NAME",
        "ctrlworld_robotwin_dual_task_new_structure_grpo_pi05",
    )

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg_ = compose(config_name=config_name)
        cfg = cfg_["env"]["train"]

    rollout_episode_index = int(
        args.episode_index
        if args.episode_index is not None
        else os.environ.get("WM_ROLLOUT_EPISODE_INDEX", "0")
    )
    rollout_output_dir = Path(
        args.output_dir
        or os.environ.get(
            "WM_ROLLOUT_OUTPUT_DIR",
            "/data/zsq/RLinf_A100/outputs/ctrlworld_rollout",
        )
    )

    with open_dict(cfg):
        cfg.total_num_envs = 1
        cfg.group_size = 1
        cfg.auto_reset = False
        cfg.use_fixed_reset_state_ids = False

    env = CtrlWorldEnv(cfg, cfg.total_num_envs, seed_offset=0, total_num_processes=1)
    if rollout_episode_index < 0 or rollout_episode_index >= len(env.dataset):
        raise ValueError(
            f"episode-index {rollout_episode_index} outside dataset range [0, {len(env.dataset) - 1}]"
        )

    trajectory = np.load(env.dataset.npy_files[rollout_episode_index], allow_pickle=True)
    actions = np.asarray(
        [frame.get("action", frame.get("state")) for frame in trajectory],
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected episode actions [T, 14], got {actions.shape}")

    env.reset(episode_indices=np.asarray([rollout_episode_index]))
    rollout_actions = actions[1:]
    if args.max_steps is not None:
        rollout_actions = rollout_actions[: int(args.max_steps)]
    if len(rollout_actions) == 0:
        raise ValueError("No rollout actions available after reset frame")

    frames = [_tensor_frames_to_uint8(env.current_obs[0, :, 0, -1:, :, :])[0]]
    rewards = [0.0]
    num_chunks = int(np.ceil(len(rollout_actions) / env.chunk))
    print(
        f"[CtrlWorldEnv] config={config_name} episode={rollout_episode_index} "
        f"steps={len(rollout_actions)} chunks={num_chunks}"
    )
    for chunk_idx in range(num_chunks):
        start = chunk_idx * env.chunk
        end = min(start + env.chunk, len(rollout_actions))
        chunk = rollout_actions[start:end]
        valid_len = len(chunk)
        if valid_len < env.chunk:
            pad = np.repeat(chunk[-1:], env.chunk - valid_len, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        _, chunk_rewards, _, _, _ = env.chunk_step(chunk[None])
        chunk_frames = _tensor_frames_to_uint8(env.current_obs[0, :, 0, -env.chunk :, :, :])[:valid_len]
        frames.extend(list(chunk_frames))
        rewards.extend([float(x) for x in chunk_rewards[0, :valid_len].detach().cpu().numpy()])
        print(
            f"[CtrlWorldEnv] chunk {chunk_idx + 1}/{num_chunks}: "
            f"frames={valid_len} reward_mean={float(chunk_rewards[0, :valid_len].mean()):.4f}"
        )

    rollout_output_dir.mkdir(parents=True, exist_ok=True)
    video_path = rollout_output_dir / f"episode_{rollout_episode_index:06d}_ctrlworld_rollout.mp4"
    rewards_path = rollout_output_dir / f"episode_{rollout_episode_index:06d}_ctrlworld_rewards.json"
    imageio.mimsave(video_path, frames, fps=int(args.fps), macro_block_size=1)
    rewards_path.write_text(
        json.dumps(
            {
                "episode_index": rollout_episode_index,
                "config_name": config_name,
                "video": str(video_path),
                "rewards": rewards,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[CtrlWorldEnv] saved video: {video_path}")
    print(f"[CtrlWorldEnv] saved rewards: {rewards_path}")
