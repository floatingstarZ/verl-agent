# EXPS 日志与文档约定

本目录用于保存 StepPPO 研究中的实验规划、方法说明、诊断记录和编译后的 PDF。

## 文档布局

- Markdown 文件可以作为历史工作笔记保留。
- LaTeX 源文件统一放在 `EXPS/tex/`。
- 编译后的 PDF 放在 `EXPS/` 根目录，并保留相同的数字前缀。
- 数字前缀是稳定文档编号，例如 `001`、`002`、`003`。
- 除 `003_shared_value_head_step_ppo_method` 这类方法/算法正式介绍外，其他文档默认以中文为主。
- 聊天归档放在 `EXPS/Chat/`；采用 append-only 原文对话记录，PDF 放在该目录根部，TeX 源文件放在 `EXPS/Chat/tex/`。

## 日志格式

训练日志是 `logs/` 下的 plain text console log。每条 metric line 通常符合如下形式：

```text
step:<global_step> - key_1:value_1 - key_2:value_2 - ...
```

日志可能带有 Ray worker 的 ANSI color prefix。解析前应先移除 ANSI escape sequence。

## 训练行与验证行区分

如果一行包含如下字段，则视为 validation row：

- `timing_s/testing`
- `val/success_rate`
- `val/webshop_task_score (not success_rate)`

如果一行包含 `timing_s/step`，但不包含 validation/testing 字段，则视为 non-validation training row。

## 常用 Metric Key

Timing metrics：

- `timing_s/step`
- `timing_s/gen`
- `timing_s/old_log_prob`
- `timing_s/ref`
- `timing_s/update_actor`
- `timing_s/adv`
- `timing_s/testing`

Resource metrics：

- `perf/throughput`
- `perf/max_memory_allocated_gb`
- `perf/max_memory_reserved_gb`
- `perf/cpu_memory_used_gb`

Policy 与行为指标：

- `episode/valid_action_ratio`
- `response_length/mean`
- `response_length/clip_ratio`
- `global_seqlen/mean`
- `actor/pg_loss`
- `actor/kl_loss`
- `actor/grad_norm`

StepPPO 特有指标：

- `actor/value_loss`
- `actor/value_clipfrac`
- `actor/value_pred_mean`
- `actor/value_return_mean`
- `critic/advantages/max`
- `critic/advantages/min`

Validation 指标：

- `val/success_rate`
- `val/webshop_task_score (not success_rate)`
- `val/webshop_text_score`

## 重要日志

GiGPO 已完成基线日志：

- `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260530_114442.log`，seed 2026
- `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_004613.log`，seed 2077
- `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_123134.log`，seed 2501

StepPPO-v1 诊断日志：

- `logs/webshop_step_ppo_qwen25_15b_4gpu_paper_align_simple_20260604_102927.log`

## 解析规则

- 使用 `step:(\d+)\s*-\s*(.*)` 提取 global step 和 metric payload。
- metric value 用 literal key name 加 `:<float>` 匹配。
- runtime 分析必须区分 validation row 和 non-validation training row。
- 比较 wall-clock 时，必须确认 `trainer.test_freq`、validation batch size、rollout group size 和 GPU 数一致。
- 对 StepPPO，value metrics 需要和 policy metrics 分开报告，因为 value head 同时引入优化信号和额外计算开销。

## 可视化输出

可视化脚本在 `VISULIZATION/` 下。当前相关输出包括：

- `VISULIZATION/figures/fig_gigpo_vs_step_ppo_val_metrics.png`
- `VISULIZATION/figures/fig_step_ppo_instability_diagnostics.png`
- `VISULIZATION/figures/fig_step_ppo_time_resource_diagnostics.png`
- `VISULIZATION/data/step_ppo_time_resource_summary.txt`
