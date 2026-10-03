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

"""Pool stage diagnostic counts before computing distributed scalar metrics."""

import logging
from collections import defaultdict

STAGE_METRICS_KEY = "__stage_metrics__"
logger = logging.getLogger(__name__)

_SUM_FIELDS = (
    "trajectory_count",
    "return_sum",
    "success_pred_count",
    "episode_control_steps_sum",
    "sampled_group_count",
    "mixed_group_count",
    "all_fail_group_count",
    "all_success_group_count",
    "filter_kept_group_count",
    "variance_positive_group_count",
    "eligible_group_count",
    "pre_filter_chunk_count",
    "post_filter_chunk_count",
)


def aggregate_stage_metrics(metrics_list: list[dict] | None) -> dict[str, float]:
    """Compute stage means from pooled sums, and union reset IDs across workers.

    Args:
        metrics_list: Actor rollout results, each optionally containing the
            reserved stage payload. Worker rank is deliberately not a stage ID.

    Returns:
        Scalar TensorBoard-compatible metrics. Empty stages have no rate entries;
        no-stage legacy results yield an empty dictionary. Unique resets are
        counted within this update and assume a shared reset manifest.
    """
    totals = defaultdict(lambda: defaultdict(float))
    reset_ids = defaultdict(set)
    diagnostics = defaultdict(float)
    for result in metrics_list or []:
        payload = (result or {}).get(STAGE_METRICS_KEY)
        if not payload:
            continue
        if payload.get("schema_version") != 1:
            logger.warning("Skipping stage metrics with an unsupported schema version.")
            continue
        for name, count in payload.get("diagnostics", {}).items():
            diagnostics[name] += float(count)
        for stage, raw in payload.get("stages", {}).items():
            stage = str(int(stage))
            for name in _SUM_FIELDS:
                totals[stage][name] += float(raw.get(name, 0))
            reset_ids[stage].update(
                int(value) for value in raw.get("reset_history_ids", []) if value >= 0
            )

    metrics = {
        f"stage/diagnostics/{name}": value for name, value in diagnostics.items()
    }
    for stage in sorted(totals, key=int):
        raw = totals[stage]
        prefix = f"stage/{stage}/"
        for name in _SUM_FIELDS:
            if name not in {"return_sum", "episode_control_steps_sum"}:
                metrics[prefix + name] = raw[name]
        metrics[prefix + "unique_reset_count"] = float(len(reset_ids[stage]))
        trajectories = raw["trajectory_count"]
        groups = raw["sampled_group_count"]
        if trajectories > 0:
            metrics[prefix + "success_rate"] = raw["success_pred_count"] / trajectories
            metrics[prefix + "return_mean"] = raw["return_sum"] / trajectories
            metrics[prefix + "episode_length_mean"] = (
                raw["episode_control_steps_sum"] / trajectories
            )
        if groups > 0:
            for source, target in (
                ("mixed_group_count", "mixed_group_fraction"),
                ("all_fail_group_count", "all_fail_group_fraction"),
                ("all_success_group_count", "all_success_group_fraction"),
                ("filter_kept_group_count", "filter_keep_fraction"),
                ("variance_positive_group_count", "variance_positive_group_fraction"),
                ("eligible_group_count", "eligible_group_fraction"),
            ):
                metrics[prefix + target] = raw[source] / groups
    return metrics
