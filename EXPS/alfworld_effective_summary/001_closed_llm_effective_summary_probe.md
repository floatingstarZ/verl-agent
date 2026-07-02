# 001 闭源 LLM 在 ALFWorld 有效 Summary 上的 Probe 实验

日期：2026-06-29  
目录：`EXPS/alfworld_effective_summary/`

## 1. 背景与动机

当前 SSCA / `traj1 + summary + traj2` 方案的核心问题是：`summary` 是否真的能成为一个对第二次尝试有用的中间 action。如果 summary 本身质量差，那么即使它参与 GRPO/VIMPO 训练，也很难稳定地产生正向优化信号。

此前在 1.5B policy 上观察到的问题包括：

1. **格式不稳定**：模型经常漏掉要求的 label，或者把 `Task:` 行直接省略。
2. **成功轨迹被误诊断为失败**：例如 traj1 已经完成任务，但 summary 仍写出大量 `failed to take`、`failed to move`。
3. **计划不可执行或过窄**：summary 往往只写 `take object from unknown`，没有保留完整路径。
4. **traj2 reward 污染**：目前还发现 ALFWorld retry reset 没有真正保证 same-task，因此旧实验中的 `reward2 - reward1` 不能直接解释为 summary 带来的改进。

因此，这个 probe 先不训练，只回答一个更基础的问题：**如果换成更强的闭源 LLM，用很简单的 prompt，它能不能生成有效的 ALFWorld retry summary？**

## 2. 实验问题

本实验关注三个问题：

1. 闭源 LLM 是否能稳定遵循简单 summary 格式？
2. 对成功轨迹，它是否能保留成功动作链，而不是误判为失败？
3. 对失败轨迹，它是否能提取有用负经验，帮助下一次少走弯路？

这个实验不直接证明 RL 训练有效；它只是检验 summary teacher / oracle summary 的可行性。

## 3. 数据来源

输入来自已有 SSCA full run 的最后 rollout：

```text
EXPS/analysis/024_ssca_retry_full/final_step_150_static_trace_bundle/full_trace_data.json
```

抽取逻辑：

1. 找到 `phase == summary` 的 row。
2. 从 summary prompt 中解析出：
   - `Task`
   - `Compact first attempt trace`
   - `Outcome feedback`
3. 保留原 1.5B policy 生成的 `old_policy_summary` 作为对比。
4. 抽取 8 个样本，尽量混合成功和失败轨迹。

生成的样本路径：

```text
EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/samples.jsonl
```

## 4. 闭源 LLM 调用方式

复用 `Projects/DataFactory_SML` 里的 LLM client：

```text
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/DataFactory_SML/SML/sml/llm_client.py
```

本次使用：

- model：`azure::gpt-5.5`
- provider：QGenie OpenAI-compatible endpoint
- key：环境变量 `QGENIE_API_KEY`
- output：纯文本三行 summary

一键运行脚本：

```bash
bash EXPS/alfworld_effective_summary/run_closed_llm_probe.sh
```

本次实际运行命令：

```bash
RUN_TAG=manual_20260629_052639 MAX_SAMPLES=8 bash EXPS/alfworld_effective_summary/run_closed_llm_probe.sh
```

主要输出：

```text
EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/closed_llm_summaries.jsonl
EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/eval_report.md
EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/eval_details.json
```

## 5. Prompt 设计

本实验刻意采用简单 prompt，避免过度格式工程。核心思想就是用户提出的：

> 请根据之前的尝试，总结出可靠的经验，帮助下一次从初始状态重新完成同一个 ALFWorld 任务。

当前模板路径：

```text
EXPS/alfworld_effective_summary/prompts/simple_experience_summary_v1.txt
```

模板要求模型只输出三行：

```text
Experience: <reliable facts or route from the previous attempt>
Next try: <one concise retry plan from the start>
Avoid: <unhelpful or invalid actions to avoid; write none if there is no evidence>
```

三行的物理含义：

- `Experience`：从 traj1 中提取的可靠事实、位置、成功路径或失败证据。
- `Next try`：traj2 从初始场景开始时可以参考的行动计划。
- `Avoid`：明确不要重复的失败动作、无效动作或无效搜索区域。

这个格式比之前五行强约束更简单，主要目的是让模型自然地产生有用经验，而不是为了满足复杂 parser。

## 6. 评估方法

当前评估是轻量规则评估 + 人工可读样例检查，不是最终 RL 指标。

规则指标包括：

| 指标 | 含义 |
|---|---|
| `heuristic_score` | 综合格式、任务覆盖、动作词、简洁性得到的粗略分数 |
| `requested 3 fields rate` | 是否包含 `Experience / Next try / Avoid` 三个字段 |
| `starts with requested fields` | 是否按指定三行字段开头 |
| `task-term overlap` | summary 是否覆盖 task 中的重要词 |
| `no wrapper/tag rate` | 是否没有 `<think>`、代码块、`<action>` 等污染 |
| `concise rate` | 是否控制在较短长度内 |

注意：`suspicious_new_name_count` 当前只是非常粗糙的词表指标，会把 `Experience`、`Next`、`Avoid`、`search` 等普通词误判为新 object，因此不作为主要结论。

## 7. 结果摘要

8 个样本上的聚合结果：

| Metric | Closed LLM | Old 1.5B policy summary |
|---|---:|---:|
| `heuristic_score mean` | 0.985 | 0.512 |
| `requested 3 fields rate` | 100.0% | 0.0% |
| `starts with requested fields` | 100.0% | 0.0% |
| `task-term overlap mean` | 0.927 | 1.000 |
| `no wrapper/tag rate` | 100.0% | 100.0% |
| `concise rate` | 100.0% | 62.5% |

主要观察：

1. 闭源 LLM 8/8 稳定遵循三行结构。
2. 闭源 LLM 的 summary 更短、更可读、更行动导向。
3. 对成功轨迹，闭源 LLM 通常能保留完整成功路线。
4. 对失败轨迹，闭源 LLM 能给出合理负经验和下一步搜索方向。
5. 旧 1.5B summary 虽然 task-term overlap 高，但经常只是复制 task/object，并不能形成可靠行动经验。

## 8. 代表样例分析

### 8.1 成功轨迹：`cool some pan and put it in stoveburner`

闭源 LLM 输出：

```text
Experience: pan 2 was on stoveburner 1; take pan 2 from stoveburner 1, cool pan 2 with fridge 1, then move pan 2 to stoveburner 1 succeeded.
Next try: go to stoveburner 1, take pan 2 from stoveburner 1, go to fridge 1, cool pan 2 with fridge 1, go to stoveburner 1, move pan 2 to stoveburner 1.
Avoid: cooling pan 2 with fridge 1 more than once.
```

这个 summary 的优点：

- 正确识别 `pan 2` 在 `stoveburner 1`。
- 保留了成功链：`take -> fridge cool -> stoveburner move`。
- `Next try` 可以直接作为 traj2 的 memory 使用。
- `Avoid` 不是泛泛而谈，而是指出不要重复 cooling。

旧 1.5B policy summary 的问题：

```text
Attempt1: tried=go to fridge 1, take pan 2, cool pan 2 with fridge 1, go to stoveburner 1, move pan 2 to stoveburner 1; bad=failed to take pan 2, failed to cool pan 2 with fridge 1, failed to move pan 2 to stoveburner 1; errors=failed to take pan 2, failed to cool pan 2 with fridge 1, failed to move pan 2 to stoveburner 1
```

虽然 traj1 是成功的，但旧 summary 却把关键步骤写成失败。这会误导 traj2。

### 8.2 成功轨迹：`heat some mug and put it in coffeemachine`

闭源 LLM 输出：

```text
Experience: mug 1 was visible on countertop 1; microwave 1 successfully heated mug 1; moving heated mug 1 to coffeemachine 1 completed the task.
Next try: go to countertop 1, take mug 1 from countertop 1, go to microwave 1, heat mug 1 with microwave 1, go to coffeemachine 1, move mug 1 to coffeemachine 1.
Avoid: none
```

优点：

- 抽出了物体位置：`mug 1` 在 `countertop 1`。
- 抽出了工具：`microwave 1` 用于 heat。
- 抽出了目标 receptacle：`coffeemachine 1`。
- 对成功轨迹没有强行编造 avoid。

### 8.3 失败轨迹：`heat some mug and put it in coffeemachine`

闭源 LLM 输出：

```text
Experience: countertop 1 had apple 1, bowl 1, bread 1, butterknife 3, butterknife 2, butterknife 1, and pot 1; coffeemachine 1 was empty; microwave 1 was open and empty; no mug was found or heated.
Next try: search cabinets, drawers, and countertops until you see a mug, take mug, go to microwave 1, heat mug with microwave 1, then go to coffeemachine 1 and put mug in coffeemachine 1.
Avoid: do not keep revisiting empty coffeemachine 1, stoveburner 1, or microwave 1 before finding and taking the mug.
```

优点：

- 正确总结失败原因：没有找到 mug。
- 明确指出无效区域：空的 `coffeemachine 1`、`microwave 1`。
- 给出下一次应扩大搜索范围的经验。

问题：

- `search cabinets, drawers, and countertops` 是高层自然语言，不是 ALFWorld admissible command。如果小模型在 traj2 中直接复制，可能产生 invalid action。
- 下一版 prompt 应要求 `Next try` 只能使用 `go to/open/take/put/move/examine/inventory/look/cool/heat/clean/use/slice` 等 ALFWorld 风格动作词，并避免 `search/check/explore`。

### 8.4 失败轨迹：`find two creditcard and put them in armchair`

闭源 LLM 输出：

```text
Experience: creditcard 1 was on sofa 1; moving creditcard 1 to armchair 1 or armchair 2 was valid, but the attempt then took it back from the armchair.
Next try: take creditcard 1 from sofa 1, move creditcard 1 to armchair 1, search remaining receptacles for another creditcard, then move it to armchair 1.
Avoid: do not take creditcard 1 from the armchair after placing it; do not finish with only one creditcard placed.
```

优点：

- 能理解 two-object task 的关键失败点：只放了一个，且又拿走了已放好的 card。
- `Avoid` 明确指出不要 undo 已完成的 subgoal。

问题：

- `search remaining receptacles` 仍然偏高层，需要进一步转为更可执行的 ALFWorld memory。

## 9. 与旧 summary prompt 的差异

旧 prompt 强调五行结构：

```text
Task / Known / Attempt1 / Plan / Rule
```

它的问题是：

1. 结构太硬，1.5B 容易漏 field。
2. `Known` 和 `Plan` 过于模板化，模型常常填 `unknown` 或复制错误 object。
3. reward bonus 鼓励格式，但不保证 summary 对 traj2 有用。
4. 对成功轨迹，模型也经常生成 `bad=failed ...`，说明它没有真正理解 outcome。

新 prompt 只保留三个直接服务 traj2 的字段：

```text
Experience / Next try / Avoid
```

这个设计更接近 memory：

- `Experience` 是状态和轨迹压缩；
- `Next try` 是可执行意图；
- `Avoid` 是负经验约束。

## 10. 当前结论

本 probe 的结论是：**闭源大模型可以有效生成 ALFWorld retry summary，且质量明显高于当前 1.5B policy summary。**

因此，闭源 LLM summary 适合作为下一阶段的：

1. **teacher summary**：为同一批 traj1 离线生成高质量 summary，用于 SFT 1.5B summary action。
2. **oracle-summary baseline**：固定使用闭源 summary，观察 traj2 是否比无 summary / 1.5B summary 更强。
3. **prompt upper bound**：判断 summary 机制本身是否有价值，避免把 1.5B summary 能力不足误判为算法无效。

## 11. 关键风险

### 11.1 Same-task retry reset 尚未修复

之前 final rollout 可视化中发现：summary 对应的 task 和 traj2 实际 task 经常不一致。例如 summary 是 `cool some pan and put it in stoveburner`，但 traj2 prompt 变成 `find two newspaper and put them in sofa`。

已定位的原因是：SSCA rollout 传入了：

```python
retry_kwargs = {"retry_same_seed": self._retry_same_seed()}
retry_obs, _ = envs.reset(kwargs=retry_kwargs)
```

但 ALFWorld environment manager 没有真正使用这个 kwargs，而是普通 reset。因此 traj2 不保证回到同一个 task。

这会污染所有 paired reward 指标：

```text
reward_delta = reward2 - reward1
```

在 same-task reset 修复前，不能把 reward delta 解释为 summary 的真实收益。

### 11.2 闭源 summary 可能过于自然语言

闭源 LLM 的 summary 可读性强，但偶尔会使用 `search`、`find`、`try` 等不是 ALFWorld action 的高层词。作为 memory 可以，但如果 actor 直接复制，可能导致 invalid action。

应在下一版 prompt 中加入：

```text
For Next try, avoid high-level words such as search, check, explore, find. Use concrete ALFWorld-style action intentions such as go to, open, take, put, move, examine, inventory, look, cool, heat, clean, use, slice.
```

## 12. 建议下一步实验

### 12.1 修复 same-task retry reset

这是最高优先级。需要让 traj2 确实从 traj1 的同一个 ALFWorld gamefile / seed / initial state 重新开始。

成功标准：

- summary input task 与 traj2 prompt task 100% 一致；
- `ssca_retry_prompt_match_rate` 接近 1；
- 可视化中每条 chain 都满足 `traj1 task == summary task == traj2 task`。

### 12.2 Closed-LLM oracle summary rollout

在 same-task retry 修复后，先不训练 summary policy，而是：

1. traj1 由当前 policy rollout；
2. summary 由闭源 LLM 离线/在线生成；
3. traj2 由 1.5B actor conditioned on closed LLM summary 执行；
4. 比较 `traj2 reward` 是否高于 `traj1 reward`。

如果 oracle summary 也不能提升 traj2，说明 memory 注入方式或 actor 使用 summary 的方式有问题。  
如果 oracle summary 能提升 traj2，说明 summary 机制本身有效，下一步才值得训练 1.5B summary action。

### 12.3 Distill 1.5B summary action

用闭源 summary 作为 teacher 构造 SFT 数据：

```text
input = traj1 compact trace + outcome feedback
label = closed LLM summary
```

目标是让 1.5B 在进入 RL 前就具备基本 summary 能力，减少纯 RL 从低质量 summary 起步的难度。

### 12.4 RL fine-tune with summary supervision

在 SFT 后，再恢复 `summary` 作为 policy action，并使用：

- traj1 标准 GRPO loss；
- summary + traj2 标准 GRPO loss；
- 可选 VIMPO / value-style improvement loss；
- summary format / groundedness auxiliary reward。

但这一步必须建立在 same-task retry 已修复的基础上。

## 13. 文件索引

实验目录：

```text
EXPS/alfworld_effective_summary/
```

关键文件：

```text
EXPS/alfworld_effective_summary/README.md
EXPS/alfworld_effective_summary/001_closed_llm_effective_summary_probe.md
EXPS/alfworld_effective_summary/closed_llm_probe_notes.md
EXPS/alfworld_effective_summary/prompts/simple_experience_summary_v1.txt
EXPS/alfworld_effective_summary/run_closed_llm_probe.sh
EXPS/alfworld_effective_summary/scripts/build_summary_samples.py
EXPS/alfworld_effective_summary/scripts/run_closed_llm_summary.py
EXPS/alfworld_effective_summary/scripts/evaluate_closed_llm_summary.py
```

本次输出：

```text
EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/samples.jsonl
EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/closed_llm_summaries.jsonl
EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/eval_report.md
EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/eval_details.json
```

## 14. 当前决策

当前判断：**Keep**。

原因：闭源 LLM 在小样本 probe 中已经证明可以生成稳定、简洁、可行动的 ALFWorld retry summary。它适合作为 teacher/oracle baseline，帮助判断 summary 机制本身是否有价值。

但在进入 full RL 前，必须先解决 same-task retry reset，否则 reward delta 和 value/improvement loss 都会被错误 task pairing 污染。
