"""Ordered 0--K stage rewards for full-start Wan multi-view WM rollouts.

This module is intentionally separate from ``world_model_wan_multi_view_env``.
Legacy ``wan_multiview_wm`` configs continue to instantiate ``WanmultiviewEnv``
and therefore retain their original binary reward behavior.
"""

from pathlib import Path

import torch

from rlinf.envs.world_model.reward_model_compat import ResnetRewModel
from rlinf.envs.world_model.world_model_wan_multi_view_env import WanmultiviewEnv

__all__ = ["WanmultiviewStageRewardEnv"]


class OrderedStageResnetRewardRouter(torch.nn.Module):
    """Route each environment to exactly one ordered binary stage RM.

    ``stage_progress`` is an absorbing integer state in ``[0, num_stages]``.
    A positive prediction advances it by exactly one.  The active stage is
    snapshotted before processing a frame, so a frame can never trigger two
    stages.  All configured stages use single-frame confirmation.
    """

    def __init__(self, checkpoint_root, stage_configs, num_envs):
        super().__init__()
        checkpoint_root = Path(str(checkpoint_root))
        stage_configs = list(stage_configs)
        if not stage_configs:
            raise ValueError(
                "OrderedStageResnetRewModel requires reward_model.stages"
            )

        self.models = torch.nn.ModuleList()
        self.stage_names = []
        for stage_idx, raw_stage_cfg in enumerate(stage_configs):
            stage_cfg = dict(raw_stage_cfg)
            expected_stage = stage_idx + 1
            configured_stage = int(stage_cfg.get("stage", expected_stage))
            if configured_stage != expected_stage:
                raise ValueError(
                    "reward_model.stages must be ordered and contiguous: "
                    f"expected stage {expected_stage}, got {configured_stage}"
                )

            confirmation_frames = int(stage_cfg.get("confirmation_frames", 1))
            if confirmation_frames != 1:
                raise ValueError(
                    "This full-start 0--K variant intentionally supports only "
                    "single-frame confirmation; got confirmation_frames="
                    f"{confirmation_frames} for stage {configured_stage}"
                )

            checkpoint_value = stage_cfg.get("checkpoint")
            if checkpoint_value is None:
                raise ValueError(
                    f"Reward stage {configured_stage} is missing checkpoint"
                )
            checkpoint_path = Path(str(checkpoint_value))
            if not checkpoint_path.is_absolute():
                checkpoint_path = checkpoint_root / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)

            threshold = stage_cfg.get("threshold")
            if threshold is None:
                raise ValueError(
                    f"Reward stage {configured_stage} is missing threshold"
                )
            self.models.append(
                ResnetRewModel(
                    checkpoint_path=str(checkpoint_path),
                    threshold=float(threshold),
                )
            )
            self.stage_names.append(
                str(stage_cfg.get("name", f"stage_{configured_stage}"))
            )

        self.num_stages = len(self.models)
        self.register_buffer(
            "stage_progress",
            torch.zeros(int(num_envs), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "last_active_score",
            torch.zeros(int(num_envs), dtype=torch.float32),
            persistent=False,
        )

    def reset_progress(self, env_idx=None):
        if env_idx is None:
            self.stage_progress.zero_()
            self.last_active_score.zero_()
            return
        mask = torch.zeros_like(self.stage_progress, dtype=torch.bool)
        mask[env_idx] = True
        self.stage_progress[mask] = 0
        self.last_active_score[mask] = 0.0

    @torch.no_grad()
    def predict_progress(self, frame_obs):
        """Return absorbing progress for every frame.

        Args:
            frame_obs: ``[num_envs, num_frames, 3, H, W]`` normalized to
                ``[-1, 1]``.
        Returns:
            Float tensor ``[num_envs, num_frames]`` with values in
            ``{0, ..., num_stages}``.
        """
        if frame_obs.ndim != 5 or frame_obs.shape[2] != 3:
            raise ValueError(
                "Expected reward frames [num_envs, num_frames, 3, H, W], "
                f"got {tuple(frame_obs.shape)}"
            )
        if frame_obs.shape[0] != self.stage_progress.shape[0]:
            raise ValueError(
                "Reward batch size does not match stage state: "
                f"frames={frame_obs.shape[0]}, states={self.stage_progress.shape[0]}"
            )

        num_envs, num_frames = frame_obs.shape[:2]
        progress_by_frame = torch.empty(
            num_envs,
            num_frames,
            dtype=torch.float32,
            device=frame_obs.device,
        )

        for frame_idx in range(num_frames):
            # Snapshotting is what enforces at most one transition per frame.
            active_stage = self.stage_progress.clone()
            for stage_idx, model in enumerate(self.models):
                sample_indices = torch.nonzero(
                    active_stage == stage_idx, as_tuple=False
                ).flatten()
                if sample_indices.numel() == 0:
                    continue

                scores = model.predict_score(
                    frame_obs[sample_indices, frame_idx]
                ).reshape(-1)
                self.last_active_score[sample_indices] = scores.to(torch.float32)
                triggered_indices = sample_indices[
                    scores >= float(model.threshold)
                ]
                if triggered_indices.numel() > 0:
                    self.stage_progress[triggered_indices] += 1

            progress_by_frame[:, frame_idx] = self.stage_progress.to(
                dtype=torch.float32
            )

        return progress_by_frame


class WanmultiviewStageRewardEnv(WanmultiviewEnv):
    """Wan multi-view environment with ordered single-frame 0--K rewards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.reward_model, OrderedStageResnetRewardRouter):
            raise TypeError(
                "wan_multiview_stage_reward_wm requires "
                "reward_model.type=OrderedStageResnetRewModel"
            )
        if not self.latch_success_reward:
            raise ValueError(
                "Ordered stage rewards require latch_success_reward=true"
            )
        if not self.use_rel_reward:
            raise ValueError(
                "Ordered stage rewards require use_rel_reward=true so the "
                "trajectory return is the number of first-time stage events"
            )

    def _load_reward_model(self):
        if self.cfg.reward_model.type != "OrderedStageResnetRewModel":
            return super()._load_reward_model()
        return OrderedStageResnetRewardRouter(
            checkpoint_root=self.cfg.reward_model.from_pretrained,
            stage_configs=self.cfg.reward_model.stages,
            num_envs=self.num_envs,
        )

    @torch.no_grad()
    def _infer_next_chunk_rewards(self, actions=None):
        del actions
        num_envs, channels, views, _, height, width = self.current_obs.shape
        if views != 1:
            raise ValueError(
                "Ordered stage RMs expect the existing vertically stacked "
                f"three-view image with singleton view axis, got views={views}"
            )

        frame_obs = self.current_obs.permute(0, 3, 1, 2, 4, 5)
        frame_obs = frame_obs[:, -self.chunk :]
        frame_obs = frame_obs.reshape(
            num_envs, self.chunk, channels, height, width
        ).to(self.device)
        return self.reward_model.predict_progress(frame_obs)

    def _apply_success_reward_latch(self, chunk_rewards):
        # The router has already produced an absorbing 0 -> ... -> K state.
        return chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        # Preserve the existing stop-on-success route, but only Stage K is final.
        return chunk_rewards.max(dim=1).values >= self.reward_model.num_stages

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx)
        reward_model = getattr(self, "reward_model", None)
        if isinstance(reward_model, OrderedStageResnetRewardRouter):
            reward_model.reset_progress(env_idx)

    def _record_metrics(self, step_reward, terminations, infos):
        infos = super()._record_metrics(step_reward, terminations, infos)
        stage_count = self.reward_model.stage_progress.clone()
        infos["reward_stage"] = stage_count
        infos["active_reward_model"] = torch.where(
            stage_count < self.reward_model.num_stages,
            stage_count + 1,
            torch.zeros_like(stage_count),
        )
        if "episode" in infos:
            infos["episode"]["stage_count"] = stage_count.to(torch.float32)
            for stage_idx in range(1, self.reward_model.num_stages + 1):
                infos["episode"][f"stage_{stage_idx}_reached"] = (
                    stage_count >= stage_idx
                )
        return infos
