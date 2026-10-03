# 0806 按按钮单任务 WM-RL

## 已准备内容

- 配置：`examples/embodiment/config/wanmultiview_arxx5_newstructure_new_cam_press_button_0806_grpo_pi05.yaml`
- initial states：`examples/embodiment/config/test_data/wanmultiview_arxx5_newcam_press_blue_button_6tasks_plus_0806`
- 数据构建脚本：`examples/embodiment/build_arxx5_press_button_initial_states_0806.py`
- 训练启动脚本：`examples/embodiment/run_press_button_0806_wmrl.sh`

合并数据只包含 `Press the blue button.`：

- 原 6-task initial states 中的按按钮轨迹：87 条
- `/data/zsq/lerobot_datasets/0806_pace_eef_lerobot_v21_10hz`：101 条
- 合计：188 条

长度统计：均值 70.22，中位数 61，P90 109，最短 24，最长 170。训练 horizon 使用 `P90 + 16` 后向上对齐一个 16 帧动作 chunk，即 128 帧；RM 判定成功后仍按现有逻辑单环境早停并 padding。

## 数据语义

每条 NPY 都保留：

- 三视角首帧：RGB、每视角 `224x224`，按头部/左腕/右腕纵向拼接为 `672x224`
- `action`：14D absolute EEF pose
- `state`：14D joint/gripper state
- instruction：`Press the blue button.`

当前 WM action map 训练协议使用 joint state，因此 YAML 保持：

```yaml
dataset_action_key: state
dataset_state_key: state
initial_joint_source: state
```

## 权重传完后启动

YAML 中的新 WM 和上一轮 VLA 路径有意设为 `null`，避免误加载旧权重。运行时通过环境变量传入：

```bash
cd /data/zsq/RLinf_A100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WM_CKPT=/data/zsq/path/to/new_button_wm.safetensors \
VLA_CKPT=/data/zsq/path/to/previous_6task_vla_torch \
bash examples/embodiment/run_press_button_0806_wmrl.sh
```

`VLA_CKPT` 必须是包含 `model.safetensors` 和对应 `norm_stats.json` 的 PyTorch checkpoint 目录。这种启动方式从上一轮 VLA 权重初始化，但新建优化器状态和训练 step。

如果传过来的是完整 RLinf checkpoint，并且希望连优化器、scheduler 和 global step 一起恢复，再额外设置：

```bash
RESUME_DIR=/data/zsq/path/to/checkpoints/global_step_N
```

只有 actor 权重时不要设置 `RESUME_DIR`。

## 重新生成 initial states

正常情况下不需要重复生成。确需重建时：

```bash
cd /data/zsq/RLinf_A100
/data/zsq/RLinf_A100/.venv/bin/python \
  examples/embodiment/build_arxx5_press_button_initial_states_0806.py \
  --overwrite
```
