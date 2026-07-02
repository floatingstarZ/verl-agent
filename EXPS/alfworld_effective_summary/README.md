# ALFWorld Effective Summary Experiments

这个目录专门用于验证：在 ALFWorld 的 `traj1 -> summary -> traj2` 设计中，summary 是否能成为对第二次尝试真正有用的经验记忆。

当前先做一个低成本 probe：

1. 从已有 SSCA final rollout 中抽取 `traj1` 后的 summary 输入样本。
2. 用 DataFactory-SML 的闭源 LLM client 调用 `azure::gpt-5.5` / QGenie provider 生成更强的 summary。
3. 用规则指标和人工可读样例检查它是否比原 1.5B policy 的 summary 更稳定、可执行、少幻觉。
4. 如果闭源 LLM summary 有明显质量优势，再把它作为 teacher / offline distillation / fixed-summary retry baseline 接入正式 ALFWorld rollout。

## 核心假设

原来的 prompt 对 1.5B policy 太格式化，模型经常缺 label、复制任务但不能形成稳定行动经验。这里先用更简单的经验总结 prompt：

> 请根据之前的尝试，总结出可靠的经验，帮助下一次从初始状态重新完成同一个 ALFWorld 任务。

## 主要文件

- `prompts/simple_experience_summary_v1.txt`：闭源 LLM summary prompt 模板。
- `scripts/build_summary_samples.py`：从 `full_trace_data.json` 抽样 summary 输入。
- `scripts/run_closed_llm_summary.py`：复用 DataFactory-SML `LLMClient` 调用闭源模型。
- `scripts/evaluate_closed_llm_summary.py`：对闭源 summary 和原 policy summary 做规则评估。
- `run_closed_llm_probe.sh`：一键构造样本、调用闭源模型、生成评估报告。

## 一键命令

```bash
bash EXPS/alfworld_effective_summary/run_closed_llm_probe.sh
```

常用覆盖：

```bash
MAX_SAMPLES=16 MODEL='azure::gpt-5.5' bash EXPS/alfworld_effective_summary/run_closed_llm_probe.sh
```

## 注意

- 默认使用环境变量 `QGENIE_API_KEY`；如果要走 Azure，可提供 `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_API_VERSION`、`AZURE_OPENAI_DEPLOYMENT` 和 Azure `BASE_URL`。
- 本目录只保存 prompt、脚本和小规模 probe 输出，不保存 API key。
- 当前 probe 只评估 summary 本身，不代表 traj2 已经在同一 ALFWorld task 上正确重置；之前发现 `retry_same_seed` 在 ALFWorld manager 中未生效，需要单独修复。 

## Traj-Refine 10-Task GPT5.4-mini Probe

一键运行：

```bash
bash EXPS/alfworld_effective_summary/run_traj_refine_train10_gpt54mini.sh
```

本次 v2 输出：

```text
EXPS/alfworld_effective_summary/outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v2_lowqps_20260629_060452/
```

核心结果：`traj1_success_rate=0.6`，`traj2_success_rate=0.6`，`task_match_rate_after_retry_reset=1.0`。详见 `002_traj_refine_train10_gpt54mini.md`。

## Summary Context Optimization

最新优化记录见 `003_summary_context_optimization.md`。当前一键 traj-refine 脚本默认使用 `SUMMARY_CONTEXT_MODE=adaptive`：失败轨迹用 structured 全局证据，成功轨迹用 final-placement / terminal-reward 证据，避免把成功轨迹中的探索循环写入 summary。

关键结论：已完成的最佳 10-task run 是 v3 structured，`traj1_success_rate=0.6`、`traj2_success_rate=0.8`；但 structured 对成功轨迹有退化风险，所以当前默认切到 safer adaptive。
