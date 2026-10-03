# A100:60023 WM-RL multi-task snapshot

This branch preserves the WM-RL source and experiment configurations from
`/data/zsq/RLinf_A100` on A100 SSH port 60023, audited on 2026-10-03.
It is an experimental source snapshot, not a portable one-command training release.

- Branch: `wmrl/multi-task-60023`.
- Original baseline: `6ec98cffa53ae3498147081a239c91d8757b2eef`.
- Publish target: `https://github.com/youmo445/RLinf_A100.git`.
- The active source checkout remains on its original `main`; publication uses an isolated clone.
- Do not interpret this branch as containing only multi-task features. It also preserves the separate full-start ordered-stage reward environment.

## Preserved behavior and entry points

The Wan multi-view environment supports instruction-routed reward models,
balanced task-group sampling, task-specific horizons, success stopping, and
inactive-environment masking. RobotWin gripper-tinted reward-model variants and
registered WoVR/CtrlWorld environment dependencies are included to avoid dangling imports.
The standard GRPO advantage and PPO loss implementations are unchanged from the baseline.
No new stage weighting, arrival-matched stage scheduler, or KIR behavior is introduced.

Relevant configurations under `examples/embodiment/config/` include:

- `wanmultiview_arxx5_newstructure_new_cam_task_specific_grpo_pi05.yaml`: multi-task, instruction-routed rewards.
- `wanmultiview_arxx5_newstructure_new_cam_packphone_0809_whole_task_grpo_pi05.yaml`: full-start phone packing, episode reward.
- `wanmultiview_arxx5_newstructure_new_cam_packphone_0809_whole_task_stage_reward_grpo_pi05.yaml`: full-start phone packing, ordered stage reward.
- Adjust-bottle, press-button, RobotWin dual-task, and WoVR/CtrlWorld comparison configurations are preserved alongside their source dependencies.

Launch patterns remain the original ones, for example:

```bash
bash examples/embodiment/run_embodiment.sh wanmultiview_arxx5_newstructure_new_cam_task_specific_grpo_pi05
bash examples/embodiment/run_packphone_fullstart_stage_reward_0to4.sh
```

These are documentation examples, not commands executed during publication.
Review all paths, resume settings, and available devices before launching.

## External dependency versions and patches

The machine-readable `dependency_versions.json` records repository URLs, exact
commits, working-tree changes, and SHA-256 hashes of relevant custom source files.
RLinf alone does not freeze the dependency stack.

| Dependency | Original checkout | Exact baseline | Local changes recorded |
| --- | --- | --- | --- |
| DiffSynth | `/data/zsq/diffsynth-studio-wujie-ppu` | `2d05394811e4d169658450d763f9b97babee5a1a` (`A100_0511`) | No tracked changes; custom Wan22 pipeline/model and ARXX5 dataset/FK modules are tracked in this commit and their hashes are recorded. |
| OpenPI | `/data/zsq/openpi` | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | `src/openpi/training/config.py`, `data_loader.py`, and `src/openpi/transforms.py`; see `patches/openpi-working-tree.patch`. |
| RoboTwin | `/data/zsq/RoboTwin` | `0008ae6800df9f75fc8de7098bacb01735fd8fd2` (`RLinf_support`) | `envs/camera/camera.py`; see `patches/robotwin-working-tree.patch`. |

To reproduce a dependency, create a separate checkout at its recorded commit.
For dependencies with a patch, run `git apply --check` before `git apply` using
the corresponding patch from this directory. Do not apply these patches blindly
to another version or to an active dirty training checkout.

The patch records include only tracked working-tree differences against the
listed dependency commits. They are not an archive of every untracked file,
installed package, or artifact in those external directories.

## Required assets and portability limitations

- Initial-state trajectory datasets and manifests, camera calibration JSONs,
  URDF/offset assets, policy normalization assets, world-model/RM/policy weights,
  and checkpoints must be prepared separately at the paths referenced by the YAMLs and source.
- Existing absolute `/data/zsq/...` paths and checkpoint resume settings are intentionally preserved. They may require path overrides; they are not changed to pretend that this snapshot is portable.
- No weights, rollout datasets, logs, videos, checkpoint archives, virtual environments, credentials, or unrelated LingBot SFT modifications are added.
- The fork's `main` was two commits ahead of the original baseline during the audit, touching the Wan environment and OpenPI action model. This branch is intentionally not rebased onto those commits; integration should be reviewed separately.

## Provenance and verification

`source_manifest.json` lists every copied source/configuration file, size, and
SHA-256, together with the excluded unrelated tracked modifications.
The source/configuration copies are byte-identical to the audited working tree;
the only additions are provenance documentation, dependency patch records, and
snapshot-consistency tests. See `validation_results.json` for checks actually run.
Static consistency checks do not establish end-to-end training reproducibility;
no GPU training or policy rollout is launched for this archival push.

Historical YAML files and exported patch context may contain original line-end
whitespace. Such staged Git formatting warnings are recorded in the validation
report rather than changing the byte-identical experiment snapshot. Python,
shell, newly authored documentation, and JSON receive a separate whitespace check.
