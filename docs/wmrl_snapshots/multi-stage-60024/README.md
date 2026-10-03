# A100:60024 multi-stage WM-RL snapshot

This branch preserves the WM-RL source and experiment configurations from
`A100:60024:/data/zsq/RLinf_A100` as inspected on 2026-10-03 (Asia/Shanghai).
The source checkout was `main` at
`6ec98cffa53ae3498147081a239c91d8757b2eef`, with uncommitted experiment changes.
The snapshot is assembled and tested in an independent clone; the original
checkout's branch, index, tracked files, and untracked files are preserved.

## Implemented behavior

The Wan multi-view environment reads the reset-state manifest, maps each
episode to a reward group/stage, and routes frames to the corresponding stage
reward model. `reward_group_sampling: balanced` distributes complete GRPO
groups across stages, samples a reset within the selected stage, and repeats
that reset for every member of the group. The alternative
`reward_group_sampling: episode_uniform` remains available.

`task_specific_rollout` supplies stage/instruction horizons, success stopping,
and masking for completed environments. Numeric `reset_stage_ids` and
`reset_history_ids` travel through the embodied rollout data structures,
environment worker, and FSDP actor without becoming policy inputs. Stage
diagnostics are aggregated across workers by counts before computing ratios.
See [the detailed stage metric definitions](../../../examples/embodiment/README_stage_metrics_zh.md).

This snapshot also preserves the existing instruction/task reward routing,
RobotWin/WoVR/CtrlWorld environment registrations, OpenPI data adapters, and
their supporting source/configuration files. It is a working-tree snapshot,
not a refactor that removes those shared capabilities.

The standard GRPO implementation in `rlinf/algorithms/advantages.py`,
`rlinf/algorithms/losses.py`, and `rlinf/algorithms/utils.py` is unchanged from
the source HEAD. There is no new stage-weighted advantage or loss here. The
later `[23,20,11,10]` deterministic schedule developed in the separate 60022
checkout is not part of this 60024 snapshot.

## External dependencies and patches

The machine-readable inventory in [external-dependencies.json](external-dependencies.json)
records source paths, Git commits, tracked dirty states, and SHA-256 checksums
of the custom modules referenced by this snapshot. These are external code
dependencies, not Git submodules of RLinf.

| Dependency | Original path | Audited commit |
| --- | --- | --- |
| DiffSynth | `/data/zsq/diffsynth-studio-wujie-ppu` | `2d05394811e4d169658450d763f9b97babee5a1a` |
| OpenPI | `/data/zsq/openpi` | `c23745b5ad24e98f66967ea795a07b2588ed6c79` |
| RoboTwin | `/data/zsq/RoboTwin` | `0008ae6800df9f75fc8de7098bacb01735fd8fd2` |

The audited OpenPI working tree has local modifications in
`src/openpi/training/config.py` and `src/openpi/transforms.py`. They are preserved
in [patches/openpi-working-tree.patch](patches/openpi-working-tree.patch).
DiffSynth/RoboTwin tracked modifications or required untracked custom modules,
when present, are exported to the corresponding patch recorded in the inventory.
A missing patch means no such changes were required for that dependency.

Apply each patch to a **separate dependency checkout at its recorded commit**;
first run `git apply --check <patch>`, then `git apply <patch>`. Do not apply the
patches to a training checkout already containing these modifications. Match
the listed custom-file checksums after applying the patches. The runtime
environment is the original RLinf `.venv` (Python 3.11.14); the environment itself
and its installed packages are not copied into this branch.

## Configuration and runtime assets

The principal stage-reset configuration is
`examples/embodiment/config/wanmultiview_arxx5_newstructure_new_cam_0723_task_specific_grpo_pi05.yaml`.
Other WM-RL and compatibility configurations are preserved with their source
filenames. [source-manifest.json](source-manifest.json) lists every copied file,
its size, and SHA-256 checksum.

The configurations retain experiment-specific absolute paths and existing
checkpoint-loading/resume settings exactly as they were on 60024. In
particular, the principal stage configuration references the historical
`boundary_balanced_v2` RM directory and policy `global_step_250` checkpoint.
This snapshot does not silently replace them with newer RMs or a fresh run.
Review `runner.ckpt_path`, `runner.resume_dir`, initial-state/manifest paths,
WM/VAE/T5 paths, camera calibration/FK assets, and RM paths before launching a
new experiment on another machine.

No initial-state datasets, training trajectories, policy/WM/RM weights, logs,
videos, generated figures, backups, virtual environment, or credentials are
included. Copy or provision those runtime assets separately. `DIFFSYNTH_PATH`
can override the default external DiffSynth checkout in `run_embodiment.sh`.
Once the environment and assets are provisioned, the existing entrypoint remains:

```bash
bash examples/embodiment/run_embodiment.sh \
  wanmultiview_arxx5_newstructure_new_cam_0723_task_specific_grpo_pi05
```

This branch was created from the source HEAD rather than first merging remote
`main`: the two newer remote commits overlap the modified WM environment and
OpenPI action model. Integration with that newer baseline is a separate step.

## Validation

Snapshot validation checks Python syntax without bytecode writes, parses each
added YAML, verifies its Hydra defaults exist, checks shell syntax and staged
diff whitespace, and confirms the copied source checksums. The four CPU tests
can be run from the snapshot using the original virtual environment:

On 2026-10-03, all 27 selected Python files, 9 YAML configurations, and 2 shell
scripts passed those checks. The four stage test modules below passed all
24 tests (2.13 seconds). The three standard GRPO source files matched the
original HEAD byte for byte. The original checkout's `main`/HEAD, index,
1,461 tracked-file hashes, and 3,364 untracked-file hashes were unchanged after
the snapshot copy and test run.

The copied historical YAML files retain their original CRLF/line-end whitespace,
and the exported OpenPI patch retains its original context whitespace. Git's
staged whitespace check reports these preserved formatting warnings. They are
recorded in the audit rather than reformatting the source snapshot. The staged
Python/shell files and newly authored README/JSON files pass that check.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  /data/zsq/RLinf_A100/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit_tests/test_stage_metadata.py \
  tests/unit_tests/test_stage_metrics.py \
  tests/unit_tests/test_stage_metric_aggregation.py \
  tests/unit_tests/test_stage_logging_integration.py
```

The validation result and original-checkout preservation checks are retained
in the local audit record. No Ray cluster, GPU rollout, or training job is
started as part of snapshot validation.
