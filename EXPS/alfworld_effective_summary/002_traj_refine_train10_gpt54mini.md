# 002 GPT5.4-mini ALFWorld Traj-Refine 10-Task 可行性实验

日期：2026-06-29  
实验目录：`EXPS/alfworld_effective_summary/`

## 1. 目标

本实验验证一个最小的 `traj1 -> summary -> traj2` refine 流程能否在 ALFWorld train tasks 上跑通，并用用户指定的评价标准比较：

```text
traj1 success rate vs traj2 success rate
```

这次先不跑 Qwen 1.5B/7B 本地模型，因为当前 GPU 基本被占用；为了先验证流程可行性，使用 DataFactory-SML 中可用的闭源模型调用能力，模型为：

```text
azure::gpt-5.4-mini
```

## 2. 实现入口

新增主脚本：

```text
EXPS/alfworld_effective_summary/scripts/run_traj_refine_eval.py
```

一键运行脚本：

```text
EXPS/alfworld_effective_summary/run_traj_refine_train10_gpt54mini.sh
```

绝对路径：

```text
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/EXPS/alfworld_effective_summary/run_traj_refine_train10_gpt54mini.sh
```

脚本会：

1. 从 ALFWorld train split 采样 `NUM_TASKS=10` 个并行环境。
2. 用 GPT5.4-mini 作为 actor rollout `traj1`。
3. 根据 `traj1` 末尾 compact trace + outcome 生成短 summary。
4. 用 `retry_same_seed=True` reset 到同一批 task。
5. 把 summary 注入每一步 traj2 prompt，继续用 GPT5.4-mini rollout `traj2`。
6. 输出 `traj1_success_rate`、`traj2_success_rate`、same-task match rate、逐任务 trace。

## 3. 运行命令

v2 正式 probe 命令：

```bash
RUN_TAG=manual_v2_lowqps_20260629_060452 MAX_WORKERS=2 NUM_TASKS=10 MAX_STEPS=50 MODEL='azure::gpt-5.4-mini' bash EXPS/alfworld_effective_summary/run_traj_refine_train10_gpt54mini.sh
```

使用 `MAX_WORKERS=2` 是因为 `MAX_WORKERS=5` 会触发 QGenie 429 rate limit。

## 4. 输出路径

v2 输出目录：

```text
EXPS/alfworld_effective_summary/outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v2_lowqps_20260629_060452/
```

关键输出：

```text
config.json
metrics.json
report.md
records.jsonl
summary_prompts.jsonl
```

其中：

- `metrics.json`：核心成功率指标。
- `report.md`：逐任务表格。
- `records.jsonl`：完整 traj1/traj2 trace、summary、reward、actions。
- `summary_prompts.jsonl`：每个 task 的 summary prompt 和 summary 输出。

## 5. Summary Prompt 约束

用户要求 summary prompt 不要太长。因此这里没有使用原来的五行复杂模板，而是使用一个短模板。

当前硬编码在 `run_traj_refine_eval.py` 中：

```text
Write a short retry memory for the SAME ALFWorld task.
Use only the trace. Next try must use concrete command-style verbs: go to, open, take X from Y, move/put X to Y, use, cool, heat, clean, slice, examine.
Do not write search/find/grab/pick up/check/explore/other furniture. For two-object tasks, mention both objects and do not undo a placed object.

Task: ...
Outcome: ...
Trace: ...

Return exactly three short lines:
Experience: <reliable facts or route>
Next try: <concrete retry plan from start>
Avoid: <bad actions/places, or none>
```

实际 prompt 长度：

- summary prompt 平均长度约 `2311` 字符；最大约 `2679` 字符。
- summary 输出平均长度约 `372` 字符；最大约 `478` 字符。

这个长度主要来自 compact trace，而不是 instruction 本身。Trace 当前最多使用最后 `8` 步，每步 observation/feedback 被截断。

## 6. Same-task Reset 检查

这次没有使用旧的 `recipe/GraphGPO` env manager，而是使用 `agent_system.environments.env_manager.AlfWorldEnvironmentManager`。该版本已经支持：

```python
retry_same_seed = bool(kwargs.get('retry_same_seed', False))
text_obs, image_obs, infos = self.envs.reset(retry_same_seed=retry_same_seed)
```

底层 worker 在 reset 时会 reseed：

```python
worker.reset.remote(reseed=retry_same_seed)
```

本次 10 个 task 的检查结果：

```text
task_match_rate_after_retry_reset = 1.0
```

即 `traj1 task == traj2 task`，这说明本次闭源 probe 的 paired comparison 没有之前 SSCA full run 中的 task mismatch 问题。

## 7. v1 与 v2 结果

先跑了一个 v1 prompt，结果如下：

```text
traj1_success_rate = 0.5
traj2_success_rate = 0.4
task_match_rate_after_retry_reset = 1.0
improved_failed_to_success = 0
degraded_success_to_failed = 1
```

v1 问题：summary 虽然可读，但仍会写 `search/find/grab/pick up` 等高层词，并且 two-object task 容易只总结一个 object，导致一个成功样本退化。

随后改成 v2 prompt，强调：

- `Next try` 必须使用 concrete command-style verbs；
- 禁止 `search/find/grab/pick up/check/explore/other furniture`；
- two-object task 必须 mention both objects；
- 不要 undo 已经放好的 object。

v2 结果如下：

```text
traj1_success_rate = 0.6
traj2_success_rate = 0.6
task_match_rate_after_retry_reset = 1.0
improved_failed_to_success = 0
degraded_success_to_failed = 0
```

结论：v2 至少避免了 v1 的退化，完整流程跑通；但在这个 10-task 样本上还没有提升总体成功率。

## 8. v2 核心指标

来自：

```text
EXPS/alfworld_effective_summary/outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v2_lowqps_20260629_060452/metrics.json
```

```json
{
  "tasks": 10,
  "traj1_success_rate": 0.6,
  "traj2_success_rate": 0.6,
  "task_match_rate_after_retry_reset": 1.0,
  "improved_failed_to_success": 0,
  "degraded_success_to_failed": 0
}
```

## 9. 逐任务结果

| id | task | traj1 | traj2 | len1 | len2 | 观察 |
|---:|---|---:|---:|---:|---:|---|
| 0 | `find two newspaper and put them in sofa` | 0 | 0 | 50 | 50 | 两次都失败；summary 未找到第二份 newspaper 的可靠位置。 |
| 1 | `put some peppershaker on drawer` | 1 | 1 | 7 | 5 | traj2 更快，summary 提示先 open drawer。 |
| 2 | `examine the newspaper with the desklamp` | 1 | 1 | 11 | 4 | traj2 更快，summary 保留 newspaper/desklamp 路径。 |
| 3 | `put a candle in toilet` | 1 | 1 | 12 | 5 | traj2 更快，summary 直接定位 drawer 3/candle/toilet。 |
| 4 | `examine the laptop with the desklamp` | 0 | 0 | 50 | 50 | 两次都在 desk/dresser/desklamp 间循环，未触发成功动作。 |
| 5 | `put two candle in drawer` | 0 | 0 | 50 | 50 | two-object 任务仍难；模型被 drawer 里的 cloth/candle 状态干扰。 |
| 6 | `find two toiletpaper and put them in shelf` | 0 | 0 | 50 | 50 | summary 幻觉 toiletpaper 在 hanger，traj2 未改善。 |
| 7 | `find two remotecontrol and put them in armchair` | 1 | 1 | 17 | 10 | traj2 更快，summary 明确两个 remotecontrol 的位置。 |
| 8 | `put some toiletpaper on toiletpaperhanger` | 1 | 1 | 4 | 4 | 两次都直接成功。 |
| 9 | `find two keychain and put them in sofa` | 1 | 1 | 12 | 9 | traj2 更快，summary 保留两个 keychain 路径。 |

成功集合没有改变，但在多个已成功任务上，traj2 的路径明显变短。

## 10. 典型成功 refine 样例

### 10.1 `find two keychain and put them in sofa`

summary：

```text
Experience: One keychain was taken from drawer 2 and put on sofa 1; the second keychain was taken from armchair 1 and also moved to sofa 1.
Next try: go to drawer 2, take keychain 1 from drawer 2, go to sofa 1, move keychain 1 to sofa 1; then go to armchair 1, take keychain 2 from armchair 1, go to sofa 1, move keychain 2 to sofa 1.
Avoid: do not use drawer 1, and do not place either keychain anywhere except sofa 1.
```

traj1 action path：

```text
go to drawer 1 -> go to drawer 2 -> open drawer 2 -> take keychain 1 from drawer 2 -> go to drawer 1 -> go to sofa 1 -> move keychain 1 to sofa 1 -> examine sofa 1 -> go to armchair 1 -> take keychain 2 from armchair 1 -> go to sofa 1 -> move keychain 2 to sofa 1
```

traj2 action path：

```text
go to drawer 2 -> open drawer 2 -> take keychain 1 from drawer 2 -> go to sofa 1 -> move keychain 1 to sofa 1 -> go to armchair 1 -> take keychain 2 from armchair 1 -> go to sofa 1 -> move keychain 2 to sofa 1
```

这里 summary 的效果很明确：traj2 删除了 `drawer 1` 和 `examine sofa` 等冗余动作，步数从 12 降到 9。

### 10.2 `examine the newspaper with the desklamp`

summary：

```text
Experience: The newspaper 2 was on the sofa 1; the desklamp 1 was on the sidetable 1 and could be turned on with use desklamp 1.
Next try: go to sofa 1, take newspaper 2 from sofa 1, go to sidetable 1, use desklamp 1, examine newspaper 2.
Avoid: examining the sidetable instead of the newspaper, and do not move the newspaper after taking it.
```

结果：

```text
traj1: success, 11 steps
traj2: success, 4 steps
```

这说明 summary 作为 memory 可以提升已成功任务的执行效率。

## 11. 失败样例与问题

### 11.1 `find two toiletpaper and put them in shelf`

summary：

```text
Experience: Drawer 1, drawer 2, drawer 3, and toiletpaperhanger 1 were empty; shelf 1 had candle 1, soapbottle 1, and spraybottle 1.
Next try: go to toiletpaperhanger 1, take toiletpaper 1 from toiletpaperhanger 1, take toiletpaper 2 from toiletpaperhanger 1, go to shelf 1, put toiletpaper 1 to shelf 1, put toiletpaper 2 to shelf 1.
Avoid: wasting steps on empty drawers or revisiting shelf 1 before taking both toiletpaper items.
```

问题：summary 前半句说 `toiletpaperhanger 1` empty，后半句却建议从 `toiletpaperhanger 1` take toiletpaper，存在自相矛盾。这说明即便是 GPT5.4-mini，失败轨迹下也会根据 task prior 幻觉 object location。

### 11.2 `put two candle in drawer`

模型被 drawer 里的 `cloth 1` 干扰，traj2 仍反复 take/move cloth。说明 summary 需要更强地绑定 task object，避免 actor 被非目标物体吸引。

## 12. 当前结论

这个 probe 已经验证：

1. `traj1 -> summary -> same-task reset -> summary-conditioned traj2` 流程可以在 ALFWorld train tasks 上跑通。
2. 使用 `agent_system` env manager 后，same-task reset 在 10/10 task 上成立。
3. 短 summary prompt 可用，输出可读且通常能压缩成功路径。
4. 在 10 个 train task 上，v2 的总体成功率为：

```text
traj1 = 6/10 = 0.6
traj2 = 6/10 = 0.6
```

5. 当前没有总体 SR 提升，但也没有退化；在成功任务中，traj2 多数步数变短。
6. 失败任务没有被 summary rescued，主要瓶颈是失败轨迹中缺少真实 object location，summary 会产生 prior-based 幻觉。

## 13. 下一步建议

### 13.1 Oracle summary 上限实验

对失败轨迹，仅用 traj1 trace 可能没有足够信息。下一步可以让闭源 LLM 多看：

- full trajectory，不只最后 8 步；
- admissible action snapshots；
- final failure reason；
- 或者额外让模型判断“哪些目标 object 从未被观察到”。

但 prompt 仍应保持短 instruction，只增加 evidence，而不是增加复杂格式。

### 13.2 Qwen 1.5B/7B 本地 actor

当前 GPU 占用较高，所以这次没有跑本地 Qwen actor。后续可把 actor backend 换成 vLLM/local Qwen，summary 仍可先用闭源 teacher：

```text
Qwen traj1 -> closed LLM summary -> Qwen traj2
```

这样更贴近最终 RL 训练场景。

### 13.3 Summary 作为 teacher data

可以把本次 `summary_prompts.jsonl` 和 closed summary 输出转成 SFT 数据，先训练 1.5B summary action，再进入 RL。

### 13.4 评价指标扩展

除了 success rate，还应记录：

- `traj2_steps - traj1_steps`：已成功任务是否更高效；
- `failed_to_success`：失败任务是否被 summary rescue；
- `success_to_failed`：summary 是否导致退化；
- `summary_contradiction_rate`：summary 是否自相矛盾；
- `target_object_coverage`：two-object task 是否覆盖两个 object。

## 14. 当前决策

当前判断：**Keep as feasibility baseline**。

原因：流程已经跑通，same-task pairing 正常，短 prompt 可用；但当前 GPT5.4-mini summary 还不能把失败任务转成功。下一阶段应转向：

1. local Qwen actor + closed summary teacher；
2. 更强 evidence extraction；
3. 用 closed summary 做 SFT 数据；
4. 再接 RL 训练。
