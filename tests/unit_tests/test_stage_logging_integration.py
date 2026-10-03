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

"""CPU integration checks for action-aligned reset logging.

The real trajectory containers and numerical actor functions are exercised.
Actor functions are loaded from their source AST to avoid importing model,
FSDP, Ray, and environment backends. Only the distributed scalar all-reduce is
replaced by a single-process no-op. No model or GPU is initialized.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    convert_trajectories_to_batch,
)
from rlinf.utils.stage_metric_aggregation import (
    STAGE_METRICS_KEY,
    aggregate_stage_metrics,
)
from rlinf.utils.stage_metrics import collect_stage_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 8
CHUNK_STEPS = 16
GROUP_SIZE = 8
ACTION_CHUNK = 16
ACTION_DIM = 2
STAGE_OFFSET = 2
METADATA_KEYS = {"reset_stage_ids", "reset_history_ids"}


def _load_functions(relative_path, qualified_names, namespace):
    """Execute selected existing functions without importing heavy backends."""
    path = REPO_ROOT / relative_path
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = []
    for qualified_name in qualified_names:
        body = source.body
        for part in qualified_name.split("."):
            node = next(item for item in body if getattr(item, "name", None) == part)
            body = getattr(node, "body", [])
        assert isinstance(node, ast.FunctionDef), qualified_name
        node.decorator_list = []
        functions.append(node)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


@pytest.fixture
def actor_functions(monkeypatch):
    """Load the production CPU data/advantage path with no FSDP instance."""
    namespace = {
        "torch": torch,
        "np": np,
        "DTensor": DTensor,
        "collect_stage_metrics": collect_stage_metrics,
        "Worker": SimpleNamespace(
            torch_platform=SimpleNamespace(current_device=lambda: torch.device("cpu"))
        ),
    }
    _load_functions(
        "rlinf/utils/metric_utils.py",
        ["compute_loss_mask", "compute_rollout_metrics"],
        namespace,
    )
    _load_functions(
        "rlinf/algorithms/utils.py",
        [
            "preprocess_embodied_advantages_inputs",
            "calculate_scores",
            "postprocess_embodied_advantages_outputs",
        ],
        namespace,
    )
    _load_functions(
        "rlinf/algorithms/advantages.py", ["compute_grpo_advantages"], namespace
    )

    def get_advantage(name):
        assert name == "grpo"
        return namespace["compute_grpo_advantages"]

    namespace["get_adv_and_returns"] = get_advantage
    _load_functions(
        "rlinf/algorithms/registry.py", ["calculate_adv_and_returns"], namespace
    )
    methods = [
        "_process_received_rollout_batch",
        "_collect_stage_rollout_metrics",
        "_rollout_tensor_to_numpy",
        "_save_rollout_diagnostics",
        "compute_advantages_and_returns",
    ]
    _load_functions(
        "rlinf/workers/actor/fsdp_actor_worker.py",
        ["process_nested_dict_for_adv", "process_nested_dict_for_train"]
        + [f"EmbodiedFSDPActor.{name}" for name in methods],
        namespace,
    )
    namespace["Actor"] = type(
        "CPUActorDataPath", (), {name: namespace[name] for name in methods}
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    return namespace


def _stage(worker, epoch, local_group):
    return 1 + (worker + epoch + local_group + STAGE_OFFSET) % 4


def _history(worker, epoch, local_group):
    # Globally unique reset identity across this synthetic update.
    return worker * 10000 + epoch * 100 + local_group


def _success_count(worker, epoch, local_group):
    # Includes all-fail, mixed, and all-success groups, not only easy groups.
    return (worker + 3 * epoch + local_group) % (GROUP_SIZE + 1)


def _make_rollout(worker, *, with_metadata=True, local_groups=1):
    """Follow EnvWorker's bootstrap/action/reward ordering over eight epochs."""
    batch_size = local_groups * GROUP_SIZE
    result = EmbodiedRolloutResult(max_episode_length=CHUNK_STEPS * ACTION_CHUNK)
    for epoch in range(EPOCHS):
        stage_ids = torch.tensor(
            [_stage(worker, epoch, group) for group in range(local_groups)],
            dtype=torch.long,
        ).repeat_interleave(GROUP_SIZE)[:, None]
        history_ids = torch.tensor(
            [_history(worker, epoch, group) for group in range(local_groups)],
            dtype=torch.long,
        ).repeat_interleave(GROUP_SIZE)[:, None]
        epoch_rewards = torch.zeros(CHUNK_STEPS, batch_size, ACTION_CHUNK)
        epoch_dones = torch.zeros(
            CHUNK_STEPS + 1, batch_size, ACTION_CHUNK, dtype=torch.bool
        )
        for group in range(local_groups):
            stage = _stage(worker, epoch, group)
            cap = (9, 16, 15, 16)[stage - 1]
            successes = _success_count(worker, epoch, group)
            for member in range(GROUP_SIZE):
                index = group * GROUP_SIZE + member
                succeeded = member < successes
                length = (
                    min(cap, 3 + (worker + epoch + group) % 5) if succeeded else cap
                )
                # The incoming done at action t describes the preceding chunk.
                epoch_dones[length, index, -1] = True
                epoch_dones[length + 1 :, index, :] = True
                if succeeded:
                    epoch_rewards[length - 1, index, -1] = 1.0

        for step in range(CHUNK_STEPS):
            identity = history_ids.float() + step / 32.0
            actions = identity.expand(batch_size, ACTION_CHUNK * ACTION_DIM).clone()
            metadata = (
                {"reset_stage_ids": stage_ids, "reset_history_ids": history_ids}
                if with_metadata
                else {}
            )
            result.append_step_result(
                ChunkStepResult(
                    actions=actions,
                    prev_logprobs=torch.zeros(batch_size, ACTION_CHUNK, ACTION_DIM),
                    prev_values=torch.zeros(batch_size, 1),
                    versions=torch.full((batch_size, 1), 7),
                    forward_inputs={
                        "action": actions.clone(),
                        "model_action": actions.clone() + 0.5,
                    },
                    rewards=None if step == 0 else epoch_rewards[step - 1],
                    dones=epoch_dones[step],
                    terminations=epoch_dones[step],
                    truncations=torch.zeros_like(epoch_dones[step]),
                    **metadata,
                )
            )
            result.append_transitions(
                {"state": identity.expand(batch_size, ACTION_DIM).clone()},
                {"state": identity.expand(batch_size, ACTION_DIM).clone() + 1.0},
            )

        # Terminal-only bootstrap contributes no action and no reset metadata.
        result.append_step_result(
            ChunkStepResult(
                rewards=epoch_rewards[-1],
                dones=epoch_dones[-1],
                terminations=epoch_dones[-1],
                truncations=torch.zeros_like(epoch_dones[-1]),
                prev_values=torch.zeros(batch_size, 1),
            )
        )
    return result


def _actor(actor_functions, trajectories, *, filter_rewards=True):
    actor = actor_functions["Actor"]()
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "rollout_epoch": EPOCHS,
                "group_size": GROUP_SIZE,
                "reward_type": "chunk_level",
                "adv_type": "grpo",
                "filter_rewards": filter_rewards,
                "rewards_lower_bound": 0.1,
                "rewards_upper_bound": 0.9,
                "gamma": 0.99,
            },
            "env": {
                "train": {
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "success_reward_threshold": 0.9,
                }
            },
            "runner": {"task_type": "embodied", "save_rollout_diagnostics": False},
        }
    )
    actor.logger = logging.getLogger(__name__)
    batch = convert_trajectories_to_batch(trajectories)
    actor.rollout_batch = actor._process_received_rollout_batch(batch)
    return actor


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    else:
        assert left == right


def test_rotating_stages_survive_shared_queue_and_actor_reshape(actor_functions):
    trajectories = [_make_rollout(worker).to_trajectory() for worker in range(8)]
    queue_order = [3, 6, 1, 7, 0, 2, 5, 4]
    actor_results = []
    for actor_rank, source_worker in enumerate(queue_order):
        actor = _actor(actor_functions, [trajectories[source_worker]])
        batch = actor.rollout_batch
        assert batch["reset_stage_ids"].shape == (CHUNK_STEPS, EPOCHS * GROUP_SIZE, 1)
        assert batch["dones"].shape == (
            CHUNK_STEPS + 1,
            EPOCHS * GROUP_SIZE,
            ACTION_CHUNK,
        )
        for epoch in range(EPOCHS):
            group = slice(epoch * GROUP_SIZE, (epoch + 1) * GROUP_SIZE)
            assert torch.all(
                batch["reset_stage_ids"][:, group] == _stage(source_worker, epoch, 0)
            )
            assert torch.all(
                batch["reset_history_ids"][:, group]
                == _history(source_worker, epoch, 0)
            )
            # Data identities verify metadata stayed aligned to the actual actions.
            assert torch.all(
                batch["actions"][0, group, 0] == _history(source_worker, epoch, 0)
            )
        assert source_worker != actor_rank
        assert actor._rollout_diagnostic_metrics == {}
        metrics = actor.compute_advantages_and_returns()
        assert STAGE_METRICS_KEY in metrics  # Independent of large NPZ saving.
        actor_results.append(metrics)

    pooled = aggregate_stage_metrics(actor_results)
    for stage in range(1, 5):
        prefix = f"stage/{stage}/"
        successes = sum(
            _success_count(worker, epoch, 0)
            for worker in range(8)
            for epoch in range(EPOCHS)
            if _stage(worker, epoch, 0) == stage
        )
        mixed = sum(
            0 < _success_count(worker, epoch, 0) < GROUP_SIZE
            for worker in range(8)
            for epoch in range(EPOCHS)
            if _stage(worker, epoch, 0) == stage
        )
        assert pooled[prefix + "trajectory_count"] == 128
        assert pooled[prefix + "sampled_group_count"] == 16
        assert pooled[prefix + "unique_reset_count"] == 16
        assert pooled[prefix + "success_pred_count"] == successes
        assert pooled[prefix + "success_rate"] == successes / 128
        assert pooled[prefix + "eligible_group_count"] == mixed
    assert pooled == aggregate_stage_metrics(list(reversed(actor_results)))


def test_split_remerge_and_multiple_sources_preserve_complete_groups(actor_functions):
    # Split along env batch, at a complete-group boundary, as the sender does.
    rollout = _make_rollout(3, local_groups=2)
    unsplit = convert_trajectories_to_batch([rollout.to_trajectory()])
    shards = rollout.to_splited_trajectories(2)
    remerged = convert_trajectories_to_batch(shards)
    _assert_equal(unsplit, remerged)

    # An actor may consume several sources; epoch transpose must preserve each
    # source's contiguous member groups even when the shared queue permutes them.
    other = _make_rollout(7).to_trajectory()
    actor = _actor(actor_functions, [other, shards[1], shards[0]])
    batch = actor.rollout_batch
    sources = [(7, 0), (3, 1), (3, 0)]
    for epoch in range(EPOCHS):
        for position, (worker, local_group) in enumerate(sources):
            start = (epoch * len(sources) + position) * GROUP_SIZE
            columns = slice(start, start + GROUP_SIZE)
            assert torch.all(
                batch["reset_stage_ids"][:, columns]
                == _stage(worker, epoch, local_group)
            )
            assert torch.all(
                batch["reset_history_ids"][:, columns]
                == _history(worker, epoch, local_group)
            )
    raw = actor._stage_rollout_metrics
    assert raw["diagnostics"]["inconsistent_group_count"] == 0
    assert sum(stage["sampled_group_count"] for stage in raw["stages"].values()) == 24


@pytest.mark.parametrize("filter_rewards", [False, True])
def test_logging_preserves_legacy_numbers_rng_and_model_inputs(
    actor_functions, filter_rewards
):
    rng_before = torch.random.get_rng_state().clone()
    legacy = _actor(
        actor_functions,
        [_make_rollout(5, with_metadata=False).to_trajectory()],
        filter_rewards=filter_rewards,
    )
    logged = _actor(
        actor_functions,
        [_make_rollout(5, with_metadata=True).to_trajectory()],
        filter_rewards=filter_rewards,
    )
    legacy_metrics = legacy.compute_advantages_and_returns()
    logged_metrics = logged.compute_advantages_and_returns()
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert legacy._stage_rollout_metrics == {}
    assert not METADATA_KEYS.intersection(legacy.rollout_batch)
    assert STAGE_METRICS_KEY not in legacy_metrics
    _assert_equal(
        legacy_metrics,
        {
            key: value
            for key, value in logged_metrics.items()
            if key != STAGE_METRICS_KEY
        },
    )
    _assert_equal(
        legacy.rollout_batch,
        {
            key: value
            for key, value in logged.rollout_batch.items()
            if key not in METADATA_KEYS
        },
    )
    # GRPO produces advantages and no value returns; metadata must not add either
    # an alternative reward/advantage path or an artificial value target.
    assert "advantages" in legacy.rollout_batch
    assert "returns" not in legacy.rollout_batch
    assert "returns" not in logged.rollout_batch

    count = CHUNK_STEPS * EPOCHS * GROUP_SIZE
    shuffle = torch.arange(count - 1, -1, -1)
    flatten = actor_functions["process_nested_dict_for_train"]
    legacy_train = flatten(legacy.rollout_batch, shuffle)
    logged_train = flatten(logged.rollout_batch, shuffle)
    for key in ("forward_inputs", "curr_obs", "next_obs"):
        assert not METADATA_KEYS.intersection(logged_train[key])
        _assert_equal(legacy_train[key], logged_train[key])
    assert torch.equal(rng_before, torch.random.get_rng_state())
