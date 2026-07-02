# Closed LLM Effective Summary Probe Notes

时间：2026-06-29

## 目的

验证闭源大模型是否能在 ALFWorld `traj1 -> summary -> traj2` 设计中生成有效 summary。这里先不改训练主线，只从已有 final rollout 中抽取 `traj1` 后的 summary 输入，用 DataFactory-SML 的 `LLMClient` 调用 `azure::gpt-5.5` / QGenie provider 生成 summary。

## 运行命令

```bash
RUN_TAG=manual_20260629_052639 MAX_SAMPLES=8 bash EXPS/alfworld_effective_summary/run_closed_llm_probe.sh
```

## 输出路径

- 样本：`EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/samples.jsonl`
- 闭源 summary：`EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/closed_llm_summaries.jsonl`
- 评估报告：`EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/eval_report.md`
- 评估明细：`EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/eval_details.json`

## Prompt

当前 prompt 保持很简单，核心是：根据之前尝试总结可靠经验，帮助下一次从初始状态重试同一个任务。

模板路径：`EXPS/alfworld_effective_summary/prompts/simple_experience_summary_v1.txt`

## 初步结论

闭源模型可以有效总结，而且比当前 1.5B policy summary 明显更稳定：

- 闭源输出稳定遵循三行结构 `Experience / Next try / Avoid`，8/8 成功。
- 对成功轨迹，能保留真实成功路线，例如 `pan 2 -> fridge -> stoveburner`、`cd 1 -> sidetable 1`、`mug 1 -> microwave -> coffeemachine`。
- 对失败轨迹，能指出有用负经验，例如没有找到 mug 时不要反复访问空的 coffeemachine/microwave，creditcard 任务不要把已经放入 armchair 的 card 又拿走。
- 相比旧 policy summary，闭源 summary 更短、更行动导向，也更少出现“成功轨迹却总结成失败”的错误。

## 规则评估摘要

来自 `EXPS/alfworld_effective_summary/outputs/closed_llm_probe_manual_20260629_052639/eval_report.md`：

| Metric | Closed LLM | Old policy summary |
|---|---:|---:|
| heuristic_score mean | 0.985 | 0.512 |
| requested 3 fields rate | 100.0% | 0.0% |
| starts with requested fields | 100.0% | 0.0% |
| task-term overlap mean | 0.927 | 1.000 |
| no wrapper/tag rate | 100.0% | 100.0% |
| concise rate | 100.0% | 62.5% |

注意：当前 `suspicious_new_name_count` 只是粗糙词表规则，会把 `Experience/Next/Avoid/search` 等普通词误判为新 object；不应作为主要结论。

## 代表样例

成功样例 `cool some pan and put it in stoveburner`：

```text
Experience: pan 2 was on stoveburner 1; take pan 2 from stoveburner 1, cool pan 2 with fridge 1, then move pan 2 to stoveburner 1 succeeded.
Next try: go to stoveburner 1, take pan 2 from stoveburner 1, go to fridge 1, cool pan 2 with fridge 1, go to stoveburner 1, move pan 2 to stoveburner 1.
Avoid: cooling pan 2 with fridge 1 more than once.
```

失败样例 `heat some mug and put it in coffeemachine`：

```text
Experience: countertop 1 had apple 1, bowl 1, bread 1, butterknife 3, butterknife 2, butterknife 1, and pot 1; coffeemachine 1 was empty; microwave 1 was open and empty; no mug was found or heated.
Next try: search cabinets, drawers, and countertops until you see a mug, take mug, go to microwave 1, heat mug with microwave 1, then go to coffeemachine 1 and put mug in coffeemachine 1.
Avoid: do not keep revisiting empty coffeemachine 1, stoveburner 1, or microwave 1 before finding and taking the mug.
```

## 需要修正/注意的点

1. 闭源模型偶尔会在 `Next try` 中写 `search/find` 这种高层词。作为 memory 没问题，但如果小模型 actor 直接复制成 action，会变成 invalid action。下一版 prompt 应补一句：`Next try` 使用 ALFWorld command vocabulary，不要写 `search/check/explore`。
2. 闭源模型 summary 只能证明“总结能力”强；要证明训练有效，还必须修复 ALFWorld retry reset，使 `traj2` 真正回到同一个 task。之前已经发现 `retry_same_seed=True` 传入后在 `AlfWorldEnvironmentManager.reset` / `AlfworldEnvs.reset` 中没有生效。
3. 建议下一步做两个对照：
   - closed-LLM summary fixed teacher + 1.5B actor retry rollout；
   - closed-LLM summary distillation dataset，用 1.5B 学 summary action。

## 当前判断

Keep 作为下一阶段方案：闭源 LLM 可以生成有效 summary，足以作为 teacher 或 oracle-summary baseline；但正式训练前应优先修复 same-task retry reset，否则 reward delta 会继续被不同 task 污染。
