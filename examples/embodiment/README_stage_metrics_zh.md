# WM-RL 分阶段日志

适用当前同步 embodied FSDP actor 的 Wan 多视角 stage-reset 训练。
存在 reward-group manifest 的环境自动提供阶段与 reset 标识，原运行命令和 YAML 无需修改。
没有阶段元数据时保留原全局日志路径，不生成虚假的阶段成功率。

## TensorBoard 指标

在原日志目录新增 `stage/1/`、`stage/2/`、`stage/3/`、`stage/4/` 等命名空间。
实际编号取自 manifest，不依赖固定 rank 对应关系。全局 `env/`、`rollout/`、`train/` 指标继续保留。

| 指标后缀 | 定义 |
|---|---|
| `success_rate` | 过滤前有效轨迹中，训练 reward 至少一次达到 success threshold 的比例；当前配置为 WM 内局部预测成功率 |
| `return_mean` | 过滤前有效轨迹的 reward 总和的平均值，不使用折扣 |
| `episode_length_mean` | 过滤前有效控制步数的轨迹平均值，计至终止控制步（含该步），不计终止后的 padding |
| `trajectory_count` | 当前 update 可归属该阶段的轨迹数 |
| `sampled_group_count` | 阶段和 reset ID 一致的完整 GRPO 组数 |
| `mixed_group_fraction` | 同时含成功和失败成员的组数 / sampled group 数 |
| `all_success_group_fraction`、`all_fail_group_fraction` | 全成功、全失败组比例 |
| `filter_keep_fraction` | 实际 reward filter 保留组数 / sampled group 数 |
| `variance_positive_group_fraction` | 回报方差大于数值阈值的组比例；不等同于 reward filter |
| `eligible_group_count`、`eligible_group_fraction` | 回报有组内差异、filter 保留且最终 mask 非空的组数及比例 |
| `pre_filter_chunk_count`、`post_filter_chunk_count` | 过滤前后 mask 中有效的 action chunk 数，每 chunk 只计一次 |
| `unique_reset_count` | 当前 update 各 worker 实际使用 reset ID 的并集大小，非各 rank 数量之和 |

这些是训练采样与潜在 GRPO 信号的诊断，不代表真机成功率、完整任务阶段自然到达率，
也不保证最终梯度非零。chunk 数不等于实际 GPU 生成成本。
`unique_reset_count` 按 update 统计，默认各 worker 共用同一 reset manifest。
全局 gradient norm、学习率等指标没有被强行分配给各阶段。

## 汇总与原始元数据

各 actor 返回按阶段的原始计数和总和；runner 跨 worker 求和后计算比率。
不同 worker 轨迹数不相等时，不会出现“平均各 rank 成功率”的分母错误。
没有样本的阶段不输出成功率 0。

轨迹和开启保存的 rollout NPZ 新增 `reset_stage_ids`、`reset_history_ids`，
形状为 `[T, B, 1]`，和产生动作的观测对齐。`B` 在 actor 端包含 rollout epochs。
`reset_history_ids` 是实际选中的 dataset episode index，不是尚未使用的下一轮预采样 ID。
标识不进入 policy observation 或 forward inputs。

未知 ID 使用 -1；有效时段中 ID 变化的轨迹不强行归入某个阶段。
组内阶段或 reset 不一致时不计算该组的阶段归因，异常数量记录在 `stage/diagnostics/`。
当前方案针对固定 reset 的 rollout，未将含多次 auto-reset 的长轨迹拆成多个 episode。
分阶段标量记录独立于大体积 NPZ 保存开关，关闭 NPZ 不会关闭阶段统计。

## CPU 验证

在仓库目录使用现有 Python 环境运行，无需加载策略或世界模型：

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q \
  tests/unit_tests/test_stage_metadata.py \
  tests/unit_tests/test_stage_metrics.py \
  tests/unit_tests/test_stage_metric_aggregation.py \
  tests/unit_tests/test_stage_logging_integration.py
```

测试包含阶段轮换、轨迹拆分拼接、来源顺序打乱、过滤前后分母、reset ID 跨 rank 去重、
缺失元数据以及 legacy 数值和随机数状态不变的检查。不会启动 Ray 集群或 GPU 训练。
