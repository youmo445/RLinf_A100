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

"""CPU-only checks for optional action-aligned reset logging metadata."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    Trajectory,
    convert_trajectories_to_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_metadata_method(relative_path: str, method_name: str):
    """Load a pure metadata method without importing GPU model/worker modules."""
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    method = next(
        node
        for class_node in tree.body
        if isinstance(class_node, ast.ClassDef)
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch, "Any": Any}
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)
    return namespace[method_name]


def test_world_metadata_uses_actual_history_and_snapshots_ids():
    get_metadata = load_metadata_method(
        "rlinf/envs/world_model/world_model_wan_multi_view_env.py",
        "get_reset_metadata",
    )
    env = SimpleNamespace(
        episode_reward_group_ids=[1, 4],
        current_reward_group_ids=torch.tensor([4]),
        current_reset_history_ids=torch.tensor([41]),
        reset_state_ids=torch.tensor([999]),
    )
    metadata = get_metadata(env)
    assert metadata["reset_history_ids"].tolist() == [[41]]
    assert metadata["reset_stage_ids"].tolist() == [[4]]
    env.current_reset_history_ids.fill_(42)
    env.current_reward_group_ids.fill_(1)
    assert metadata["reset_history_ids"].tolist() == [[41]]
    assert metadata["reset_stage_ids"].tolist() == [[4]]
    env.episode_reward_group_ids = None
    assert get_metadata(env) == {}


def test_worker_reads_wrapped_metadata_without_legacy_dynamic_attributes():
    read_metadata = load_metadata_method(
        "rlinf/workers/env/env_worker.py", "_get_reset_metadata"
    )

    class FakeEnvironment:
        def get_reset_metadata(self):
            return {"reset_stage_ids": torch.tensor([[3]])}

    wrapped = SimpleNamespace(env=SimpleNamespace(env=FakeEnvironment()))
    assert read_metadata(wrapped)["reset_stage_ids"].item() == 3
    assert read_metadata(SimpleNamespace()) == {}
    assert read_metadata(MagicMock()) == {}
    cycle = SimpleNamespace()
    cycle.env = cycle
    assert read_metadata(cycle) == {}


def test_action_start_metadata_is_cloned_and_terminal_only_step_is_ignored():
    stage = torch.tensor([1, 1])
    history = torch.tensor([10, 10])
    observation = EnvOutput(
        obs={"states": torch.zeros(2, 14)},
        reset_stage_ids=stage,
        reset_history_ids=history,
    )
    stage.fill_(4)
    history.fill_(40)
    assert "reset_stage_ids" not in observation.to_dict()
    assert "reset_history_ids" not in observation.to_dict()["obs"]
    result = EmbodiedRolloutResult()
    result.append_step_result(
        ChunkStepResult(
            actions=torch.zeros(2, 14),
            reset_stage_ids=observation.reset_stage_ids,
            reset_history_ids=observation.reset_history_ids,
        )
    )
    result.append_step_result(
        ChunkStepResult(
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            reset_stage_ids=torch.full((2, 1), 4),
            reset_history_ids=torch.full((2, 1), 40),
        )
    )
    trajectory = result.to_trajectory()
    assert trajectory.reset_stage_ids.shape == (1, 2, 1)
    assert trajectory.reset_stage_ids.flatten().tolist() == [1, 1]
    assert trajectory.reset_history_ids.flatten().tolist() == [10, 10]


def test_missing_step_and_shard_metadata_remains_aligned_as_unknown():
    result = EmbodiedRolloutResult()
    result.append_step_result(ChunkStepResult(actions=torch.zeros(2, 14)))
    result.append_step_result(
        ChunkStepResult(
            actions=torch.zeros(2, 14),
            reset_stage_ids=torch.tensor([2, 2]),
            reset_history_ids=torch.tensor([20, 20]),
        )
    )
    trajectory = result.to_trajectory()
    assert trajectory.reset_stage_ids[..., 0].tolist() == [[-1, -1], [2, 2]]
    legacy = Trajectory(actions=torch.zeros(2, 1, 14))
    batch = convert_trajectories_to_batch([legacy, trajectory])
    assert batch["reset_stage_ids"][..., 0].tolist() == [
        [-1, -1, -1],
        [-1, 2, 2],
    ]
    assert batch["reset_history_ids"][..., 0].tolist() == [
        [-1, -1, -1],
        [-1, 20, 20],
    ]
    legacy_batch = convert_trajectories_to_batch([legacy])
    assert "reset_stage_ids" not in legacy_batch
    assert "reset_history_ids" not in legacy_batch


def test_intervention_filter_keeps_reset_metadata_with_selected_actions():
    trajectory = Trajectory(
        actions=torch.arange(6).reshape(3, 2, 1),
        intervene_flags=torch.tensor([[[1], [0]], [[0], [0]], [[1], [0]]]).bool(),
        reset_stage_ids=torch.tensor([[[1], [2]], [[3], [4]], [[2], [1]]]),
        reset_history_ids=torch.tensor([[[10], [20]], [[30], [40]], [[21], [11]]]),
    )
    selected = trajectory.extract_intervene_traj()
    assert len(selected) == 1
    assert selected[0].actions.flatten().tolist() == [0, 4]
    assert selected[0].reset_stage_ids.flatten().tolist() == [1, 2]
    assert selected[0].reset_history_ids.flatten().tolist() == [10, 21]
