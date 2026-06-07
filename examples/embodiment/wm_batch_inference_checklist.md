# Wan Multi-View WM Batch Inference Checklist

目标：把当前 world model 按 env 逐条 `self.pipe(**kwargs)` 的推理，改成可配置 micro-batch 推理，例如 `wm_infer_batch_size=1/2/4/8`。默认必须保持 `1`，保证旧行为完全不变。

## 0. 当前结论

当前不能只改 `/data/zsq/RLinf/rlinf/envs/world_model/world_model_wan_multi_view_env.py` 的 for 循环。DiffSynth Wan pipeline 现在基本是单样本推理路径：

- `WanVideoUnit_NoiseInitializer` 写死 noise batch 维为 `1`。
- `BasePipeline.preprocess_video()` 只支持 `list[PIL.Image] -> [1, C, T, H, W]`。
- `BasePipeline.vae_output_to_video()` 会把 batch 维 reduce 掉，无法返回多条视频。
- `WanVideoUnit_TI2V2ContextEmbedder`、`WanVideoUnit_Wan22CodexVACE144` 当前输入也是单条视频。

所以需要同时改 RLinf env 调度层和 DiffSynth pipeline batch 支持。

## 1. 配置新增项

涉及 yaml：

- `/data/zsq/RLinf/examples/embodiment/config/wanmultiview_robotwin_grpo_pi05.yaml`
- `/data/zsq/RLinf/examples/embodiment/config/wanmultiview_robotwin_ppo_pi05.yaml`
- 如需 ARXX5，也同步：
  - `/data/zsq/RLinf/examples/embodiment/config/wanmultiview_arxx5_grpo_pi05.yaml`
  - `/data/zsq/RLinf/examples/embodiment/config/wanmultiview_arxx5_ppo_pi05.yaml`

新增字段建议放在 `env.train` 和 `env.eval`：

```yaml
wm_infer_batch_size: 1
```

语义：

- `1`：当前逐 env 推理，旧行为。
- `>1`：一次把多个 env 合成 batch 送入 Wan pipeline。
- 推荐先试 `2`，再试 `4`，最后再考虑 `8`。

约束：

- `wm_infer_batch_size <= num_envs`
- 对 GRPO，建议先要求 `group_size % wm_infer_batch_size == 0`，避免跨 group 合 batch。
- 初期不要跨 group batch；只在同一个 group 内 micro-batch。

## 2. RLinf env 需要改

文件：

```text
/data/zsq/RLinf/rlinf/envs/world_model/world_model_wan_multi_view_env.py
```

### 2.1 初始化读取配置

在 `__init__` 中读取：

```python
self.wm_infer_batch_size = int(cfg.get("wm_infer_batch_size", 1))
```

检查：

- 必须 `>=1`
- 如果 `self.group_size` 存在，建议要求 `group_size % wm_infer_batch_size == 0`

### 2.2 `_infer_next_chunk_frames()` 重构

当前核心结构：

```python
for env_idx in range(num_envs):
    kwargs = {...}
    output = self.pipe(**kwargs)
```

目标结构：

```python
for batch_env_indices in infer_batches:
    batch_kwargs = collect_batch_kwargs(batch_env_indices)
    batch_outputs = self.pipe(**batch_kwargs, return_batch=True)
    scatter_outputs_back_to_envs(batch_env_indices, batch_outputs)
```

需要保持这些状态接口完全不变：

- `self.current_obs`: `[num_envs, 3, 1, T, H, W]`
- `self.image_queue`: 每个 env 独立更新
- `self.condition_action`: `[num_envs, condition_frame_length, 14]`
- `self.current_action14`: `[num_envs, 14]`
- `self.last_control_video`: `list[num_envs][chunk frames]`
- `self.last_control_video_chunks`: 结构不变

### 2.3 每个 env 仍要单独构造的信息

这些信息不能混：

- `input_video`
- `vace_video` 或 `action_map`
- `ray_map_o`
- `ray_map_d`
- `prompt`
- `seed`
- `reference_frame`
- `image_queue`
- `condition_action`
- `current_action14`

特别是 seed。当前：

```python
seed = int(self.cfg.get("seed", 0)) + int(self.elapsed_steps) + env_idx
```

batch 后应变成：

```python
seed = [base_seed + elapsed_steps + env_idx for env_idx in batch_env_indices]
```

不能整个 batch 共用一个 seed。

### 2.4 action chunk 多次 WM rollout

当前已支持：

```text
PI05 action_chunk = 16
WM chunk = 8
拆成 2 次自回归 WM rollout
```

batch 化后仍必须保持：

```text
第 1 个 8 动作 -> batch WM -> 更新 obs/queue/action
第 2 个 8 动作 -> batch WM -> 再更新 obs/queue/action
```

不能把 16 帧动作一次性送给 horizon=8 的 WM。

### 2.5 main / standalone rollout

如果 `/data/zsq/RLinf/rlinf/envs/world_model/world_model_wan_multi_view_env.py` 的 main 仍然单 env 可视化，必须保证：

```yaml
wm_infer_batch_size: 1
```

或单 env 时自动退化为 batch size 1。

## 3. DiffSynth utils 需要改

文件：

```text
/data/zsq/diffsynth-studio-wujie-ppu/diffsynth/utils/__init__.py
```

### 3.1 `BasePipeline.preprocess_video()`

当前：

```python
list[PIL.Image] -> [1, C, T, H, W]
```

需要支持：

```text
list[PIL.Image] -> [1, C, T, H, W]
list[list[PIL.Image]] -> [B, C, T, H, W]
torch.Tensor [B, C, T, H, W] -> 直接规范 dtype/device/range
```

要求：

- 老接口不变。
- 对 batch 输入检查每条视频帧数相同。
- 检查每帧尺寸一致。

### 3.2 `BasePipeline.vae_output_to_video()`

当前会 reduce batch：

```python
reduce(vae_output, "B C T H W -> T H W C", reduction="mean")
```

这会把 batch 平均掉，必须改。

建议：

```python
vae_output_to_video(..., return_batch=False)
```

行为：

- `B=1, return_batch=False -> list[PIL.Image]`
- `return_batch=True -> list[list[PIL.Image]]`

## 4. DiffSynth Wan pipeline 需要改

文件：

```text
/data/zsq/diffsynth-studio-wujie-ppu/diffsynth/pipelines/wan_video_new.py
```

### 4.1 `WanVideoPipeline.__call__()`

当前接口偏单样本：

```python
prompt: str
input_video: list[Image]
vace_video: list[Image]
ray_map_o: list[Image]
ray_map_d: list[Image]
seed: int
```

需要兼容：

```python
prompt: list[str]
input_video: list[list[Image]]
vace_video: list[list[Image]]
ray_map_o: list[list[Image]]
ray_map_d: list[list[Image]]
seed: list[int]
```

建议新增：

```python
return_batch: bool = False
```

行为：

- 旧单样本调用不变。
- batch 调用返回 `list[list[PIL.Image]]`。

### 4.2 `WanVideoUnit_NoiseInitializer`

当前硬编码：

```python
shape = (1, pipe.vae.model.z_dim, length, ...)
```

需要：

```python
B = infer_batch_size
shape = (B, pipe.vae.model.z_dim, length, ...)
```

seed 支持：

- `int`: 旧行为
- `list[int]`: 每个样本独立 generator，生成后 `torch.cat`

### 4.3 `WanVideoUnit_PromptEmbedder`

需要支持：

```text
prompt: str -> [1, L, D]
prompt: list[str] -> [B, L, D]
```

如果 `pipe.prompter.encode_prompt()` 不支持 list，就循环 encode 后 concat。

负 prompt 同理。

### 4.4 `WanVideoUnit_InputVideoEmbedder`

当前：

```python
input_video = pipe.preprocess_video(input_video)
input_latents = pipe.vae.encode(input_video, ...)
```

需要支持 batch context：

```text
input_video_batch -> [B, C, T, H, W]
input_latents -> [B, C_latent, T_latent, H_latent, W_latent]
latents[:, :, 0:context_slots] = input_latents[:, :, 0:context_slots]
```

### 4.5 `WanVideoUnit_ImageEmbedderFused`

如果当前 Wan2.2 TI2V path 使用 `fuse_vae_embedding_in_latents`，它也要支持 batch：

```python
image: [B, C, H, W]
z: [B, C_latent, 1, H_latent, W_latent]
latents[:, :, 0:1] = z
```

如果你的当前路径不走它，也要避免 batch 调用时误触发报错。

### 4.6 CFG merge

检查：

- `cfg_scale == 1.0`
- `cfg_scale != 1.0`
- `cfg_merge == False`
- `cfg_merge == True`

如果 `cfg_merge=True`，通常需要把 latents/context/vace_context 都扩到 `2B`。否则：

```python
noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
```

会出错或语义不对。

## 5. DiffSynth Wan2.2 Codex VACE 需要改

文件：

```text
/data/zsq/diffsynth-studio-wujie-ppu/diffsynth/pipelines/wan_video_new_wan22_codex.py
```

### 5.1 `WanVideoUnit_Wan22CodexVACE144._as_latents()`

当前逻辑：

```python
video = pipe.preprocess_video(value)
return pipe.vae.encode(video, ...)
```

需要支持：

```text
value = list[PIL.Image] -> [1, 48, T', H', W']
value = list[list[PIL.Image]] -> [B, 48, T', H', W']
value = tensor [B, C, T, H, W] -> [B, 48, T', H', W']
```

### 5.2 `WanVideoUnit_Wan22CodexVACE144.process()`

保持：

```python
vace_context = torch.cat((ray_o_latents, ray_d_latents, action_latents), dim=1)
```

batch 形状应为：

```text
ray_o_latents:   [B, 48, T', H', W']
ray_d_latents:   [B, 48, T', H', W']
action_latents:  [B, 48, T', H', W']
vace_context:    [B, 144, T', H', W']
```

需要检查三者的 batch、time、height、width 完全一致。

## 6. VACE model 需要确认/可能修改

文件：

```text
/data/zsq/diffsynth-studio-wujie-ppu/diffsynth/models/wan_video_vace_wan22_codex.py
```

当前注释偏单样本：

```python
# vace_context: [1, 144, 5, 30, 14]
c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
```

对 batch 来说，`for u in vace_context` 会按 batch 拆样本，最后再 cat。理论上可能还能得到 `[B, seq, dim]`，但必须确认：

- `c.shape[0] == B`
- 每个 VACE block 输出 `[B, seq, dim]`
- 返回的 `vace_hints` 每个元素也是 `[B, seq, dim]`
- `model_fn_wan_video()` 里加到 `x` 时 batch 对齐

如果能跑且 shape 对，就不用大改；否则要改成真正 batch tensor path：

```python
c = self.vace_patch_embedding(vace_context)
c = c.flatten(2).transpose(1, 2)
```

## 7. RLinf 自定义 TI2V2 context unit

文件：

```text
/data/zsq/RLinf/rlinf/envs/world_model/world_model_wan_multi_view_env.py
```

类：

```python
WanVideoUnit_TI2V2ContextEmbedder
```

当前只接受单条 `input_video`。

需要兼容：

```text
input_video: list[PIL.Image]
input_video: list[list[PIL.Image]]
```

输出：

```text
latents: [B, C, T', H', W']
first_frame_latents: [B, C, context_slots, H', W']
```

并保持首帧锚定逻辑：

```python
latents[:, :, 0:2] = context_latents[:, :, 0:2]
```

## 8. 输出后处理

在 `WanVideoPipeline.__call__()` decode 后：

```python
video = self.vae.decode(...)
video = self.vae_output_to_video(video)
```

batch 模式应返回：

```text
list[list[PIL.Image]]
```

然后 RLinf env 将每个样本 scatter 回：

```python
all_samples[env_idx] = pred_video
image_queue[env_idx] = ...
current_action14[env_idx] = ...
condition_action[env_idx] = ...
last_control_video[env_idx] = ...
```

## 9. RecordVideo / rollout worker 影响

如果 `self.current_obs` 结构不变，则这些理论上不用改：

- `/data/zsq/RLinf/rlinf/envs/wrappers/record_video.py`
- rollout worker
- reward model

但要验证：

- 视频保存数量不变
- 每个 env 的 reward shape 仍为 `[num_envs, chunk]`
- `infos["episode"]` 结构不变

## 10. standalone 脚本影响

可能受影响：

```text
/data/zsq/RLinf/examples/embodiment/run_pi05_wm_autoregressive_rollout.py
/data/zsq/RLinf/examples/embodiment/regenerate_wanmultiview_rollout_with_reward.py
```

要求：

- 默认 `wm_infer_batch_size=1` 时完全不变。
- 如果以后想单脚本多 env rollout，再接入 batch 输出。

## 11. 必测 case

### 11.1 单样本回归

```text
wm_infer_batch_size=1
num_envs=1
```

检查：

- main 能跑
- `run_pi05_wm_autoregressive_rollout.py` 能跑
- 输出视频尺寸不变

### 11.2 多 env micro-batch

```text
num_envs=4
group_size=4
wm_infer_batch_size=2
```

检查：

- `self.current_obs.shape == [4, 3, 1, T, H, W]`
- reward shape `[4, chunk]`
- 每个 env 的 prompt/seed/action 不混

### 11.3 GRPO group 内 batch

```text
group_size=8
wm_infer_batch_size=2 or 4
```

检查：

- 每个 group 初始状态仍共享
- group 内 seed 仍按 env_idx 不同
- GRPO advantage 计算不受影响

### 11.4 action chunk 多次 rollout

```text
PI05 action_chunk=16
WM chunk=8
wm_infer_batch_size=2
```

确认：

- 每次 PI05 输出仍拆成 2 次 WM rollout
- 第 2 次 WM rollout 使用第 1 次生成后的 image_queue/current_action14

### 11.5 CFG

测试：

- `cfg_scale=1.0`
- `cfg_scale!=1.0`
- 如有使用，测试 `cfg_merge=True`

### 11.6 显存

RobotWin 当前输入大约：

```text
单视角 256x320
三视角拼接 768x320
Wan2.2 5B + VACE
```

建议顺序：

```text
wm_infer_batch_size=1 -> 2 -> 4 -> 8
```

不要直接上 `8`。

## 12. 推荐实现顺序

1. 在 DiffSynth 增加 batch-safe helper，但保持旧单样本行为。
2. 改 `NoiseInitializer` 支持 `B` 和 `seed list`。
3. 改 `preprocess_video()` 和 `vae_output_to_video()`。
4. 改 `PromptEmbedder` 支持 prompt list。
5. 改 `TI2V2ContextEmbedder` 支持 batch context。
6. 改 `Wan22CodexVACE144` 支持 batch control/ray/action map。
7. 用独立小脚本直接测 `self.pipe(..., B=2, return_batch=True)`。
8. 再改 RLinf `_infer_next_chunk_frames()` 做 micro-batch 调度。
9. 跑 `wm_infer_batch_size=1` 回归。
10. 跑 `wm_infer_batch_size=2`。
11. 评估显存后再试 `4/8`。

## 13. 风险点

- batch 越大，VAE encode、VACE context、DiT attention 显存都会增加。
- 直接 `batch=group_size=8` 可能 OOM。
- 如果 `vae_output_to_video()` 误 reduce batch，会把多个 env 平均成一条视频，这是最危险的 silent bug。
- 如果 seed list 没处理好，group 内轨迹随机性会异常一致。
- 如果 prompt list 没处理好，不同 env 可能共用 prompt。
- 如果 action_chunk 拆分顺序出错，第二个 WM chunk 会用错 context。
