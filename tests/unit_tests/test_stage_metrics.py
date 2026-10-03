"""CPU-only tests for observational stage statistics (no model imports)."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

# Import the pure NumPy helper directly: rlinf.__init__ registers Torch/OmegaConf
# integrations that are unnecessary for these observational-math unit tests.
_spec = importlib.util.spec_from_file_location(
    "stage_metrics_under_test",
    Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "stage_metrics.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
collect_stage_metrics = _module.collect_stage_metrics


class StageMetricsTest(unittest.TestCase):
    def test_group_signal_filter_and_control_step_lengths(self):
        rewards = np.zeros((3, 12, 2), dtype=np.float32)
        rewards[0, [0, 4, 5, 6, 7], 1] = 1
        stages = np.broadcast_to(np.array([1] * 8 + [2] * 4)[None, :, None], (3, 12, 1))
        histories = np.broadcast_to(
            np.array([10] * 4 + [11] * 4 + [20] * 4)[None, :, None], (3, 12, 1)
        )
        chunks = np.array([2] * 4 + [3] * 4 + [1] * 4)
        pre = (np.arange(3)[:, None] < chunks[None, :])[..., None]
        post = pre.copy()
        post[:, 4:8] = False
        lengths = np.broadcast_to((chunks * 2)[None, :, None], (3, 12, 1))
        snapshot = [
            array.copy() for array in (rewards, stages, histories, pre, post, lengths)
        ]
        result = collect_stage_metrics(
            rewards=rewards,
            reset_stage_ids=stages,
            reset_history_ids=histories,
            pre_filter_loss_mask=pre,
            loss_mask=post,
            loss_mask_sum=lengths,
            group_filter_keep=np.array([True, False, True]),
            group_size=4,
        )
        first = result["stages"]["1"]
        self.assertEqual(first["trajectory_count"], 8)
        self.assertEqual(first["return_sum"], 5)
        self.assertEqual(first["success_pred_count"], 5)
        self.assertEqual(first["episode_control_steps_sum"], 40)
        self.assertEqual(first["sampled_group_count"], 2)
        self.assertEqual(first["mixed_group_count"], 1)
        self.assertEqual(first["all_success_group_count"], 1)
        self.assertEqual(first["filter_kept_group_count"], 1)
        self.assertEqual(first["variance_positive_group_count"], 1)
        self.assertEqual(first["eligible_group_count"], 1)
        self.assertEqual(first["pre_filter_chunk_count"], 20)
        self.assertEqual(first["post_filter_chunk_count"], 8)
        self.assertEqual(first["reset_history_ids"], [10, 11])
        second = result["stages"]["2"]
        self.assertEqual(second["all_fail_group_count"], 1)
        self.assertEqual(second["filter_kept_group_count"], 1)
        self.assertEqual(second["eligible_group_count"], 0)
        self.assertEqual(second["episode_control_steps_sum"], 8)
        for actual, expected in zip(
            (rewards, stages, histories, pre, post, lengths), snapshot
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_variance_is_not_binary_mixed_success(self):
        result = collect_stage_metrics(
            rewards=np.array([[[0], [0.1], [0.2], [0.3]]]),
            reset_stage_ids=np.ones((1, 4), dtype=int),
            reset_history_ids=np.full((1, 4), 7),
            group_size=4,
        )
        stage = result["stages"]["1"]
        self.assertEqual(stage["all_fail_group_count"], 1)
        self.assertEqual(stage["mixed_group_count"], 0)
        self.assertEqual(stage["variance_positive_group_count"], 1)
        self.assertEqual(stage["eligible_group_count"], 1)

    def test_filter_pass_and_nonempty_mask_are_separate(self):
        result = collect_stage_metrics(
            rewards=np.array([[[1.0], [0.0]]]),
            reset_stage_ids=np.ones((1, 2), dtype=int),
            reset_history_ids=np.ones((1, 2), dtype=int),
            pre_filter_loss_mask=np.ones((1, 2, 1), dtype=bool),
            loss_mask=np.zeros((1, 2, 1), dtype=bool),
            group_filter_keep=np.array([True]),
            group_size=2,
        )
        stage = result["stages"]["1"]
        self.assertEqual(stage["filter_kept_group_count"], 1)
        self.assertEqual(stage["eligible_group_count"], 0)
        self.assertEqual(stage["return_sum"], 1)

    def test_changed_metadata_and_inconsistent_groups_are_not_misattributed(self):
        result = collect_stage_metrics(
            rewards=np.zeros((2, 4, 1)),
            reset_stage_ids=np.array([[1, 1, 2, 2], [1, 1, 3, 2]]),
            reset_history_ids=np.array([[10, 11, 20, 20], [10, 11, 20, 20]]),
            group_size=2,
        )
        self.assertEqual(result["diagnostics"]["invalid_metadata_trajectory_count"], 1)
        self.assertEqual(result["diagnostics"]["inconsistent_group_count"], 2)
        self.assertEqual(result["stages"]["1"]["trajectory_count"], 2)
        self.assertEqual(result["stages"]["1"]["sampled_group_count"], 0)
        self.assertEqual(result["stages"]["2"]["trajectory_count"], 1)
        self.assertNotIn("3", result["stages"])

    def test_unknown_metadata_and_incomplete_group_are_reported(self):
        result = collect_stage_metrics(
            rewards=np.zeros((1, 3, 1)),
            reset_stage_ids=np.array([[1, -1, 2]]),
            reset_history_ids=np.array([[10, -1, 20]]),
            group_size=2,
        )
        self.assertEqual(result["diagnostics"]["invalid_metadata_trajectory_count"], 1)
        self.assertEqual(result["diagnostics"]["incomplete_group_count"], 1)
        self.assertEqual(result["diagnostics"]["inconsistent_group_count"], 1)
        self.assertEqual(result["stages"]["2"]["sampled_group_count"], 0)

    def test_control_length_fallback_and_padded_metadata(self):
        dones = np.zeros((3, 2, 2), dtype=bool)
        dones[1, :, -1] = True
        result = collect_stage_metrics(
            rewards=np.zeros((2, 2, 2)),
            reset_stage_ids=np.array([[1, 1], [-1, 2]]),
            reset_history_ids=np.array([[10, 10], [-1, 11]]),
            pre_filter_loss_mask=np.array([[[True], [True]], [[False], [False]]]),
            dones=dones,
            group_size=2,
        )
        self.assertEqual(result["stages"]["1"]["episode_control_steps_sum"], 4)
        self.assertEqual(result["stages"]["1"]["sampled_group_count"], 1)
        self.assertEqual(result["diagnostics"]["invalid_metadata_trajectory_count"], 0)

    def test_legacy_is_exact_noop_and_does_not_draw_randomness(self):
        before = np.random.get_state()
        self.assertEqual(collect_stage_metrics(rewards=np.zeros((1, 2, 1))), {})
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_partial_metadata_reports_missing_labels(self):
        result = collect_stage_metrics(
            rewards=np.zeros((1, 2, 1)),
            reset_stage_ids=np.ones((1, 2), dtype=int),
        )
        self.assertEqual(result["stages"], {})
        self.assertEqual(result["diagnostics"]["invalid_metadata_trajectory_count"], 2)


if __name__ == "__main__":
    unittest.main()
