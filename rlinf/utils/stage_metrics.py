"""Read-only stage diagnostics for fixed-reset embodied rollout batches.

All numeric fields are additive sufficient statistics, not rank-local means.
``reset_history_ids`` must be unioned across ranks (never summed). Success here
means a threshold event in the training rewards, not an independently verified
task outcome. A reset label does not identify stages visited later in a rollout.
"""

from __future__ import annotations

import numpy as np


def _mask_like(mask, rewards):
    if mask is None:
        return np.ones(rewards.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2:
        mask = mask[..., None]
    return np.broadcast_to(mask, rewards.shape)


def _metadata_matrix(value, time_steps, batch_size):
    value = np.asarray(value)
    if value.shape == (time_steps, batch_size, 1):
        value = value[..., 0]
    if value.shape != (time_steps, batch_size) or value.dtype.kind not in "iu":
        raise ValueError("Reset metadata must be integer [T, B] or [T, B, 1]")
    return value


def _stage_counters():
    return {
        "trajectory_count": 0,
        "return_sum": 0.0,
        "success_pred_count": 0,
        "episode_control_steps_sum": 0.0,
        "sampled_group_count": 0,
        "mixed_group_count": 0,
        "all_fail_group_count": 0,
        "all_success_group_count": 0,
        "filter_kept_group_count": 0,
        "variance_positive_group_count": 0,
        "eligible_group_count": 0,
        "pre_filter_chunk_count": 0,
        "post_filter_chunk_count": 0,
        "reset_history_ids": [],
    }


def collect_stage_metrics(
    *,
    rewards,
    reset_stage_ids=None,
    reset_history_ids=None,
    pre_filter_loss_mask=None,
    loss_mask=None,
    loss_mask_sum=None,
    dones=None,
    group_filter_keep=None,
    group_size=8,
    success_threshold=0.9,
    variance_epsilon=1e-12,
):
    """Return raw stage statistics without modifying any input array.

    Rewards are [T, B, H]; B already includes rollout epochs. Groups follow the
    actor's contiguous ``reshape(-1, group_size)`` convention. ``group_filter_keep``
    is the actual pre-broadcast filter decision, not a guess from the final mask.
    When filtering is disabled it is None, meaning all formed groups passed.

    Only trajectories with a known, constant stage/history over valid time are
    attributable. Auto-reset trajectories that change labels are excluded and
    counted in diagnostics. Group statistics additionally require every member
    to share the same reset stage and history. A positive-variance eligible group
    has potential GRPO signal; this does not assert a nonzero optimizer gradient.
    """
    if reset_stage_ids is None and reset_history_ids is None:
        return {}
    rewards = np.asarray(rewards)
    if rewards.ndim != 3 or group_size <= 0:
        raise ValueError("Expected [T, B, H] rewards and positive group_size")
    time_steps, batch_size, horizon = rewards.shape
    diagnostics = {
        "invalid_metadata_trajectory_count": 0,
        "empty_trajectory_count": 0,
        "incomplete_group_count": int(batch_size % group_size != 0),
        "inconsistent_group_count": 0,
    }
    result = {"schema_version": 1, "stages": {}, "diagnostics": diagnostics}
    if reset_stage_ids is None or reset_history_ids is None:
        diagnostics["invalid_metadata_trajectory_count"] = batch_size
        return result
    if time_steps == 0 or horizon == 0:
        diagnostics["empty_trajectory_count"] = batch_size
        return result

    stage_ids = _metadata_matrix(reset_stage_ids, time_steps, batch_size)
    history_ids = _metadata_matrix(reset_history_ids, time_steps, batch_size)
    pre_mask = _mask_like(pre_filter_loss_mask, rewards)
    post_mask = _mask_like(loss_mask, rewards) & pre_mask
    pre_chunks = pre_mask.any(axis=2)
    post_chunks = post_mask.any(axis=2)
    returns = np.where(pre_mask, rewards, 0).sum(axis=(0, 2), dtype=np.float64)
    successes = ((rewards >= success_threshold) & pre_mask).any(axis=(0, 2))

    if loss_mask_sum is not None:
        lengths = np.asarray(loss_mask_sum)
        if lengths.shape[:2] != (time_steps, batch_size):
            raise ValueError("loss_mask_sum must align with [T, B]")
        # This is the pre-filter control-step count retained by the actor,
        # including when the actual policy loss mask was collapsed to chunks.
        lengths = lengths.reshape(time_steps, batch_size, -1)[0, :, 0]
    elif dones is not None:
        dones = np.asarray(dones, dtype=bool)
        if dones.shape != (time_steps + 1, batch_size, horizon):
            raise ValueError("dones must be [T + 1, B, H]")
        flat_dones = dones.transpose(0, 2, 1).reshape(-1, batch_size)
        flat_dones = flat_dones[-(time_steps * horizon + 1) :]
        lengths = (flat_dones.cumsum(axis=0) == 0)[:-1].sum(axis=0)
    else:
        lengths = pre_mask.sum(axis=(0, 2))

    labels = [None] * batch_size
    histories = [None] * batch_size
    for index in range(batch_size):
        valid_time = pre_chunks[:, index]
        if not valid_time.any():
            diagnostics["empty_trajectory_count"] += 1
            continue
        stages = stage_ids[valid_time, index]
        resets = history_ids[valid_time, index]
        if (
            (stages < 0).any()
            or (resets < 0).any()
            or not (stages == stages[0]).all()
            or not (resets == resets[0]).all()
        ):
            diagnostics["invalid_metadata_trajectory_count"] += 1
            continue
        label = str(int(stages[0]))
        reset_id = int(resets[0])
        labels[index] = label
        histories[index] = reset_id
        counts = result["stages"].setdefault(label, _stage_counters())
        counts["trajectory_count"] += 1
        counts["return_sum"] += float(returns[index])
        counts["success_pred_count"] += int(successes[index])
        counts["episode_control_steps_sum"] += float(lengths[index])
        counts["pre_filter_chunk_count"] += int(pre_chunks[:, index].sum())
        counts["post_filter_chunk_count"] += int(post_chunks[:, index].sum())
        counts["reset_history_ids"].append(reset_id)

    group_count = batch_size // group_size
    if group_filter_keep is None:
        filter_keep = np.ones(group_count, dtype=bool)
    else:
        filter_keep = np.asarray(group_filter_keep, dtype=bool).reshape(-1)
        if filter_keep.size != group_count:
            raise ValueError("group_filter_keep must have one decision per group")
    for group in range(group_count):
        start = group * group_size
        stop = start + group_size
        group_labels = labels[start:stop]
        group_histories = histories[start:stop]
        if (
            group_labels[0] is None
            or any(label != group_labels[0] for label in group_labels)
            or any(reset != group_histories[0] for reset in group_histories)
        ):
            diagnostics["inconsistent_group_count"] += 1
            continue
        counts = result["stages"][group_labels[0]]
        success_count = int(successes[start:stop].sum())
        positive_variance = bool(returns[start:stop].var() > variance_epsilon)
        kept = bool(filter_keep[group])
        eligible = kept and positive_variance and bool(post_chunks[:, start:stop].any())
        counts["sampled_group_count"] += 1
        counts["mixed_group_count"] += int(0 < success_count < group_size)
        counts["all_fail_group_count"] += int(success_count == 0)
        counts["all_success_group_count"] += int(success_count == group_size)
        counts["filter_kept_group_count"] += int(kept)
        counts["variance_positive_group_count"] += int(positive_variance)
        counts["eligible_group_count"] += int(eligible)

    for counts in result["stages"].values():
        counts["reset_history_ids"] = sorted(set(counts["reset_history_ids"]))
    return result
