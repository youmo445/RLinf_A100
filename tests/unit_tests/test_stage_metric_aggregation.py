"""CPU tests for stage logging across workers with unequal sample counts."""

import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from rlinf.utils.stage_metric_aggregation import (
    STAGE_METRICS_KEY,
    aggregate_stage_metrics,
)


def payload(stage, **counts):
    return {STAGE_METRICS_KEY: {"schema_version": 1, "stages": {str(stage): counts}}}


def test_pooled_counts_not_mean_of_worker_means():
    results = [
        payload(1, trajectory_count=8, success_pred_count=8, return_sum=8),
        payload(1, trajectory_count=24, success_pred_count=0, return_sum=0),
    ]
    metrics = aggregate_stage_metrics(results)
    assert metrics["stage/1/success_rate"] == 0.25
    assert metrics["stage/1/return_mean"] == 0.25
    assert metrics["stage/1/trajectory_count"] == 32


def test_union_reset_ids_and_independent_stage_counts():
    results = [
        payload(1, trajectory_count=8, reset_history_ids=[10, 11, -1]),
        payload(1, trajectory_count=8, reset_history_ids=[11, 12]),
        payload(4, trajectory_count=8, reset_history_ids=[10]),
    ]
    metrics = aggregate_stage_metrics(results)
    assert metrics["stage/1/unique_reset_count"] == 3
    assert metrics["stage/4/unique_reset_count"] == 1
    assert metrics == aggregate_stage_metrics(list(reversed(results)))


def test_group_rates_use_all_sampled_groups_before_filter():
    metrics = aggregate_stage_metrics(
        [
            payload(2, sampled_group_count=2, eligible_group_count=2),
            payload(2, sampled_group_count=6, eligible_group_count=0),
        ]
    )
    assert metrics["stage/2/eligible_group_fraction"] == 0.25
    assert "stage/2/success_rate" not in metrics


def test_legacy_and_missing_stage_do_not_emit_fake_zero_rates():
    assert aggregate_stage_metrics(None) == {}
    assert aggregate_stage_metrics([None, {}, {"reward": 1.0}]) == {}
    metrics = aggregate_stage_metrics([payload(3, trajectory_count=0)])
    assert "stage/3/success_rate" not in metrics
    assert not any(key.startswith("stage/1/") for key in metrics)


def test_additive_diagnostics_and_unknown_schema():
    result = {
        STAGE_METRICS_KEY: {
            "schema_version": 1,
            "stages": {},
            "diagnostics": {"unknown_stage_trajectory_count": 8},
        }
    }
    metrics = aggregate_stage_metrics([result, result])
    assert metrics["stage/diagnostics/unknown_stage_trajectory_count"] == 16
    assert aggregate_stage_metrics([{STAGE_METRICS_KEY: {"schema_version": 2}}]) == {}


def _runner_method(name):
    source = Path(__file__).parents[2] / "rlinf/runners/embodied_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"defaultdict": defaultdict, "STAGE_METRICS_KEY": STAGE_METRICS_KEY}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


def test_runner_preserves_legacy_average_and_skips_raw_payload():
    aggregate = _runner_method("_aggregate_numeric_metrics")
    legacy = [{"reward": 0.5}, {"reward": 1.0}]
    with_stage = [dict(result, **payload(1, trajectory_count=8)) for result in legacy]
    assert aggregate(None, with_stage) == aggregate(None, legacy) == {"reward": 0.75}


def test_per_worker_logger_never_receives_nested_payload():
    calls = []
    runner = SimpleNamespace(
        enable_per_worker_metric_log=True,
        metric_logger=SimpleNamespace(log=lambda **kwargs: calls.append(kwargs)),
    )
    results = [dict(reward=0.5, **payload(1, trajectory_count=8)), payload(2)]
    _runner_method("_log_ranked_metrics")(runner, results, 7, "rollout", "actor")
    assert len(calls) == 1
    assert calls[0]["data"] == {"rollout/reward": 0.5}
