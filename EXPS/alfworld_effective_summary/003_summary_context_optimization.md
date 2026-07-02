# 003 Summary Context 组织优化记录

日期：2026-06-29  
目标：优化 ALFWorld `traj1 -> summary -> traj2` 中的 summary 生成上下文，使 summary 更像“可执行经验记忆”，并让 `traj2_success_rate` 相比 `traj1_success_rate` 更稳定提升。

## 结论

1. **只给最近若干步 raw trace 不够**：v2 的 raw-tail prompt 太容易只看到局部失败，遗漏未探索候选位置，`traj2` 没有提升。
2. **失败轨迹最需要结构化全局证据**：v3 的 structured evidence 把“访问过/打开过/目标相关事件/未访问初始候选/循环动作”显式列出，在 10 个 train task 上把 `traj2_success_rate` 从 `0.6` 提到 `0.8`，救回 2 个失败任务。
3. **成功轨迹不能也无脑给全局结构化证据**：成功轨迹里经常包含大量探索和循环；如果 summary 把这些也写进 `Next try`，会让第二次尝试变长，甚至把已放好的物体拿出来。
4. **当前默认改成 `adaptive`**：失败轨迹仍用 structured evidence；成功轨迹改用 `final placement / terminal reward` 视角，只保留最终有效放置、终止奖励动作和明确的 undo/loop 禁忌。
5. **traj2 使用 memory 时也需要约束**：已把 `append_retry_memory` 改成“route hints, not current state”，要求只选当前 admissible action；如果 memory 中的 take/move 当前不可执行，先导航/打开，避免模型直接执行不可用动作。

## 当前实现改动

### Summary prompt 约束

`SUMMARY_TEMPLATE` 现在额外要求：

- `Next try` 必须是可执行路线，而不是抽象建议。
- 每个 `take X from Y` 前需要包含 `go to Y`；如果 `Y` 是容器，需要 `open Y`。
- 每个 `move/put X to Y` 前需要包含 `go to Y`。
- examine-with-desklamp 任务应采用 `use desklamp -> go to object -> examine object`，除非证据明确说明需要搬动物体。
- 成功轨迹优先总结最终有效放置/终止奖励动作，忽略早期失败循环。
- 两物体任务必须明确两个物体；未知时写 `second object unknown`。
- 不允许把已放到最终目标的物体再拿出来。

### Context mode

当前支持四种 context mode：

| mode | 作用 | 适用性 |
| --- | --- | --- |
| `raw_tail` | 只给最后 N 步原始 trace | 最短，但经常遗漏全局信息，不推荐 |
| `structured` | 从完整 traj1 中抽取 visited/opened/unvisited/target/manipulation/loop evidence | 救回失败能力最强，但成功轨迹可能被早期探索干扰 |
| `hybrid` | structured + 最近 raw trace | 信息最多，prompt 更长，未作为默认 |
| `adaptive` | 失败用 structured，成功用 final-placement/terminal-reward evidence | 当前推荐默认，更稳地避免成功轨迹退化 |

### 成功轨迹的 final-placement evidence

新增 `build_success_plan_evidence` 后，成功轨迹不再用“完整 successful action path”作为核心上下文，而是解析：

- `Parsed intent`：目标物体、最终 receptacle、工具物体。
- `Completed final placements`：最终仍在目标 receptacle 中的物体，以及它们第一次被发现的位置。
- `Terminal reward event`：获得 reward / done 的终止动作。
- `Critical target/tool operations`：只保留目标物体和工具相关的 take/move/use/examine。
- `Undo actions that should NOT be copied`：例如从 sofa 中把已经放好的 keychain 拿出来。
- `Likely loops to avoid`：重复三次以上的动作。

这个设计是为了解决 v4 中 `find two keychain and put them in sofa` 的退化：旧 adaptive 从成功轨迹中保留了大量早期循环，summary 误写成“再从 sofa 取 keychain 再放回 sofa”，导致 traj2 卡死。

### Traj2 memory 注入

`append_retry_memory` 现在明确告诉 actor：

- retry memory 是路线提示，不是当前状态。
- 仍然只能选择当前 admissible action。
- 如果 memory 中某个动作当前不可执行，要先导航/打开。
- 不要从最终目标中拿回已经放好的物体。

这解决的是另一类问题：summary 可能写出全局计划，但 actor 每一步面对的是局部 admissible actions；如果没有说明，模型会把计划中的后续动作当成当前动作。

## 实验结果对比

闭源模型：`azure::gpt-5.4-mini`  
任务：ALFWorld train split 前 10 个 task，`seed=2026`，`max_steps=50`，`max_workers=2`。  
注意：GPT5.4-mini/QGenie 侧没有完全确定性，同样 seed 的 `traj1` 会有小幅波动，所以对比重点是同一次 run 内 `traj2 - traj1` 的方向，以及具体 trace 的失败原因。

| 版本 | context/prompt | 输出路径 | traj1 SR | traj2 SR | improve | degrade | 观察 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| v2 | raw-tail + 短 prompt | `outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v2_lowqps_20260629_060452/` | 0.6 | 0.6 | 0 | 0 | 不退化，但失败轨迹救不回来 |
| v3 | structured evidence | `outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v3_structured_20260629_061950/` | 0.6 | 0.8 | 2 | 0 | 最好 completed 10-task 结果，救回 newspaper 和 laptop |
| v4 | old adaptive：成功轨迹压缩为 useful path | `outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v4_adaptive_20260629_063300/` | 0.6 | 0.6 | 1 | 1 | keychain 成功轨迹被早期循环污染，发生退化 |
| v5 | adaptive + final-placement success evidence | `outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v5_successplan_20260629_064719/` | 0.6 | 0.6 | 0 | 0 | 去掉退化，但这次没有救回失败 |
| v6 | structured + route prompt fix | `outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v6_structured_promptfix_20260629_065826/` | 0.5 | 0.6 | 2 | 1 | 救回 laptop/remotecontrol，但成功 keychain 因计划缺少 go-to target 退化 |
| v7 | adaptive + route-hint actor memory | `outputs/traj_refine_train10_azure__gpt-5.4-mini_seed2026_manual_v7_adaptive_routehint_20260629_071012/` | 未完成 | 未完成 | - | - | 本地 Ray actor 在 traj1 step 40 中断，不是 summary 代码错误 |
| v7-smoke | adaptive + route-hint，2 task/20 step | `outputs/traj_refine_train2_azure__gpt-5.4-mini_seed2026_smoke_v7_adaptive_routehint_20260629_071456/` | 0.5 | 0.5 | 0 | 0 | smoke 跑通，task reset match=1.0，无退化 |

## 关键样例分析

### v3 成功救回：newspaper two-object task

任务：`find two newspaper and put them in sofa`。  
traj1 失败在 50 步；structured evidence 明确给出：

- sofa 已访问但没有 newspaper。
- 已访问抽屉/桌子没有 newspaper。
- `diningtable 1`、`sidetable 1/2`、`garbagecan 1` 等初始候选未访问。
- 第二份 newspaper unknown。

summary 因此引导 traj2 去未访问候选位置搜索，最终 traj2 在 23 步成功。这个样例说明，失败轨迹需要“全局搜索覆盖率/未访问候选”的 context，而不是最后 8 步 raw trace。

### v3/v6 成功救回：laptop with desklamp

任务：`examine the laptop with the desklamp`。  
失败轨迹经常在 desk/dresser/desklamp 附近循环。structured evidence 能抽出：

- desklamp 在 `dresser 1`，可以 `use desklamp 1`。
- desk 没有 laptop。
- `bed 1` 是未访问候选。

traj2 在对应 run 中转向 bed，找到 laptop 后成功。这个样例说明，summary 不一定要复述长轨迹，关键是把“已证伪位置”和“还没查的位置”组织清楚。

### v4 退化：keychain two-object task

任务：`find two keychain and put them in sofa`。  
traj1 虽然最终成功，但过程中多次把 keychain 从 sofa 拿出来再放回去。old adaptive 把这段循环纳入 `Shortest useful successful path candidate`，summary 错写：

- “another keychain was later on sofa”
- “go to sofa 1 and take keychain 1 from sofa 1 if needed”

这直接违背两物体放置任务的物理意义：已经放到 final target 的物体不应再拿走。因此 v7 将成功轨迹改成 final-placement evidence，并在 prompt 和 memory 注入处都加入“不要从最终目标拿回已放好的物体”。

## 当前推荐配置

一键脚本默认已改为：

```bash
bash EXPS/alfworld_effective_summary/run_traj_refine_train10_gpt54mini.sh
```

等价关键参数：

```bash
SUMMARY_CONTEXT_MODE=adaptive \
MAX_WORKERS=2 \
NUM_TASKS=10 \
MAX_STEPS=50 \
MODEL='azure::gpt-5.4-mini' \
bash EXPS/alfworld_effective_summary/run_traj_refine_train10_gpt54mini.sh
```

如果只追求复现目前最好的 completed 10-task 数字，可以显式用旧思路的 structured context：

```bash
SUMMARY_CONTEXT_MODE=structured \
MAX_WORKERS=2 \
NUM_TASKS=10 \
MAX_STEPS=50 \
MODEL='azure::gpt-5.4-mini' \
bash EXPS/alfworld_effective_summary/run_traj_refine_train10_gpt54mini.sh
```

但 structured 对成功轨迹的退化风险更高；如果后续要接到训练/teacher 数据，建议先用 `adaptive`。

## 后续建议

1. 在干净 RUNAI 环境重跑 v7 `adaptive` 10-task，因为本地 Ray 已出现大量 defunct actor，当前 10-task v7 被环境中断。
2. 把 success/failure 分开统计：失败轨迹看 rescue rate，成功轨迹看 degradation rate 和 traj2 step inflation。
3. 对 examine-with-desklamp 单独做模板化 summary：`use lamp -> go object -> examine object`，避免“把 laptop 搬到 desk”这类无用动作。
4. 如果后续训练 1.5B policy，应把 summary action 的监督信号绑定到 `traj2` 的提升，而不是只让 summary 模仿文本；否则 summary 可能格式正确但对动作无用。
