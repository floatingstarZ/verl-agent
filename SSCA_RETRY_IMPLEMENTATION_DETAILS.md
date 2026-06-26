# SSCA Retry-GRPO 实现细节说明

本文档说明当前仓库里第一版 SSCA（Self-Summary Credit Assignment）在 ALFWorld + GRPO 框架下是如何实现的，以及 summary 为什么会作为可训练 action 参与优化。文末附了从已有 smoke trace 抽取的真实 rollout 样例，方便人工 review。

> 当前实现目标链路：`traj1 + summary prompt -> summary action -> reset -> ALFWorld system/user prompt + summary -> traj2 actions`。
>
> 其中 `summary action` 是 actor 生成的一段 response；它和后续 `traj2` actions 共享 retry episode reward，从而让 summary 的目标变成提升后续 policy 的质量。

## 1. 代码入口与文件地图

核心文件：

- `recipe/SSCA/main_ssca.py`: SSCA 训练入口，基本沿用 PPO/GRPO trainer，只把 rollout collector 换成 `SSCA2TrajCollector`。
- `recipe/SSCA/rollout_loop.py`: SSCA 的主要实现；定义 `traj1 -> summary -> reset -> traj2`。
- `agent_system/environments/env_manager.py`: 给 ALFWorld reset 增加 `retry_same_seed` 入口。
- `agent_system/environments/env_package/alfworld/envs.py`: 在 worker 中保存初始 seed，并支持 retry reset 时 reseed。
- `scripts/run_ssca_retry_grpo_alfworld_smoke.sh`: 一键 smoke/full 脚本，仍然走标准 GRPO trainer 配置。

关键入口位置：

- `recipe/SSCA/main_ssca.py:170`: import `SSCA2TrajCollector`。
- `recipe/SSCA/main_ssca.py:171`: 实例化 `traj_collector` 并交给 `RayPPOTrainer`。
- `recipe/SSCA/rollout_loop.py:29`: `SSCA2TrajCollector` 文档化当前链路。
- `recipe/SSCA/rollout_loop.py:333`: `vanilla_multi_turn_loop` 是每个 batch rollout 的主流程。

## 2. 整体数据流

对一个 train batch，当前实现会把每个 prompt/group 展开成两类 GRPO 训练组：

1. `traj1` 组：
   - 输入：原始 ALFWorld prompt。
   - 动作：actor 在环境中执行第一轮完整/截断 episode。
   - reward：第一轮 episode reward，即 `reward1`。
   - GRPO uid 后缀：`:traj1`。

2. `retry` 组：
   - 第一个 row 是 `summary`：输入是 `traj1` 的完整历史 + outcome feedback + summary instruction；输出是 self-summary。
   - 后续 rows 是 `traj2`：环境 reset 到初始状态，然后 actor 只看当前 ALFWorld prompt/observation + summary，不看 traj1 明细。
   - reward：第二轮 retry episode reward，即 `reward2`。
   - GRPO uid 后缀：`:retry`。

因此一个样本在训练数据里近似是：

```text
uid=:traj1
  row 1: obs_1 -> action_1, episode_reward = reward1
  row 2: obs_2 -> action_2, episode_reward = reward1
  ...

uid=:retry
  row 0: full_traj1 + feedback + summary_prompt -> summary, episode_reward = reward2
  row 1: reset_obs_1 + summary -> retry_action_1, episode_reward = reward2
  row 2: retry_obs_2 + retry_history + summary -> retry_action_2, episode_reward = reward2
  ...
```

## 3. `traj1` 如何生成

在 `recipe/SSCA/rollout_loop.py:342`，collector 首先调用 `envs.reset(...)` 得到第一轮 ALFWorld 初始 observation。

随后：

- `recipe/SSCA/rollout_loop.py:347`: 创建 `uid_first`，用于 GRPO 的第一轮 group。
- `recipe/SSCA/rollout_loop.py:349`: 创建 `traj_uid_first`，用于标记每条真实 trajectory。
- `recipe/SSCA/rollout_loop.py:353`: 调用 `_run_env_episode(... phase="traj1", attempt=1)`。

`_run_env_episode` 内部每一步做以下事情：

1. 用当前 observation 调 `preprocess_batch`，把 ALFWorld 文本包装成 chat prompt。
2. 调 `actor_rollout_wg.generate_sequences(...)` 让 actor 生成 response。
3. 解码 response，从 `<action>...</action>` 中投影出环境动作。
4. 调 `envs.step(text_actions)` 执行动作。
5. 记录 row metadata：`uid`, `traj_uid`, `ssca_phase`, `ssca_attempt`, `ssca_step_idx`, `active_masks`, `rewards`, `is_action_valid` 等。
6. 把当前 observation/action/reward/done/valid 写入 `histories`，供 summary prompt 使用。

对应代码：

- 生成 actor response: `recipe/SSCA/rollout_loop.py:74`
- episode loop: `recipe/SSCA/rollout_loop.py:263`
- step env: `recipe/SSCA/rollout_loop.py:280`
- 记录历史: `recipe/SSCA/rollout_loop.py:304`

## 4. Summary action 如何生成

第一轮完成后，`recipe/SSCA/rollout_loop.py:365` 构造 summary observation。

summary prompt 包含三块：

1. `[Original task prompt]`: 原始 ALFWorld 任务 prompt。
2. `[First trajectory]`: 第一轮 trajectory 的 observation/action/reward/done/valid 历史。
3. `[Outcome feedback]`: 第一轮结果反馈，包括 reward、是否 won、episode length、invalid action count、final observation 等。

生成 summary 的代码路径：

- `recipe/SSCA/rollout_loop.py:188`: `_build_summary_obs(...)` 构造 summary prompt。
- `recipe/SSCA/rollout_loop.py:156`: `_feedback_text(...)` 生成 failure/success feedback 文本。
- `recipe/SSCA/rollout_loop.py:373`: `_generate_from_obs(...)` 让 actor 生成 summary。
- `recipe/SSCA/rollout_loop.py:374`: `_decode_responses(...)` 得到 summary 文本。

注意：summary 不是外部规则生成的标签，也不是 SFT target。它是当前 policy 采样出的 response，因此在 PPO/GRPO 更新时有自己的 logprob、old_logprob、ref_logprob、response mask 和 policy loss。

当前 instruction 在 `recipe/SSCA/rollout_loop.py:14` 明确告诉模型：

```text
first trajectory + summary prompt -> self-summary -> ALFWorld task prompt + self-summary -> retry actions
```

这使 summary 的语义与用户希望的链路对齐：summary 的目标不是复述轨迹，而是提升后续 retry policy 的质量。

## 5. `traj2` 如何只看到 `sysprompt + summary + 当前环境状态`

summary 生成后，collector reset 环境：

- `recipe/SSCA/rollout_loop.py:391`: `retry_kwargs = {"retry_same_seed": self._retry_same_seed()}`
- `recipe/SSCA/rollout_loop.py:392`: `envs.reset(kwargs=retry_kwargs)`
- `recipe/SSCA/rollout_loop.py:393`: `_prepend_summary_to_obs(...)` 把 summary 拼到 retry observation 后面。

`_prepend_summary_to_obs` 的 contract 在 `recipe/SSCA/rollout_loop.py:221`：

```text
current ALFWorld observation / prompt

[Self-summary action generated from the first attempt]
{summary}

Retry context contract: use the current ALFWorld observation plus this self-summary only.
The detailed first trajectory is not available; choose exactly one admissible action.
```

也就是说：

- summary row 的 state 是 `full traj1 + feedback + summary prompt`。
- traj2 step-0 的 state 是 `reset 后的 ALFWorld 初始 prompt + summary`。
- traj2 step-k 的 state 是 `retry 内部历史 + 当前 observation + summary`。
- traj2 不会看到 traj1 的 detailed history；traj1 只通过 summary 这个 action 压缩传递。

实际进入模型时，`agent_system/multi_turn_rollout/rollout_loop.py:90` 会把 observation 文本放成一个 user message，再由 tokenizer 的 chat template 包装，因此实际模型输入是：

```text
system: You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user: <ALFWorld prompt/current retry observation + summary block>
assistant: <to be generated>
```

这就是你说的 `sysprompt + sum -> action1 -> ...`。其中 `sysprompt` 由 chat template 注入，`sum` 被放在 user observation 文本内，并在每个 retry step 重复提供。

## 6. Summary 为什么会受到 GRPO 监督

关键点是：summary row 被放进 retry 轨迹组的第一个 row。

在 `recipe/SSCA/rollout_loop.py:376`，summary batch 被标记为：

- `uid = uid_retry`
- `traj_uid = traj_uid_retry`
- `ssca_phase = "summary"`
- `ssca_attempt = 2`
- `ssca_step_idx = -1`

在 `recipe/SSCA/rollout_loop.py:424`，最终 retry 训练序列是：

```python
retry_items = [summary_items[idx]] + retry_lists[idx]
```

然后在 `recipe/SSCA/rollout_loop.py:428`，这整个 retry trajectory 的 `episode_reward` 被设为 `reward2`。

接下来走原框架的标准 reward/advantage 路径：

1. `agent_system/multi_turn_rollout/rollout_loop.py:268` 会把 `episode_rewards[bs]` 写到每个 active row 的 `data['episode_rewards']`，包括 summary row。
2. `agent_system/reward_manager/episode.py:72` 读取每个 row 的 `episode_rewards`。
3. `agent_system/reward_manager/episode.py:79` 把该 scalar reward 放到该 row response 的最后一个有效 token 上。
4. `verl/trainer/ppo/ray_trainer.py:304` 调标准 `compute_grpo_outcome_advantage(...)`。
5. `verl/trainer/ppo/core_algos.py:144` 以每个 row 的 token-level reward sum 作为 GRPO score。
6. `verl/trainer/ppo/core_algos.py:155` 按 `uid` 聚合 group，`uid_retry` 下包含 summary row 和 retry action rows。
7. `verl/trainer/ppo/core_algos.py:167` 给每个 response token 分配 group-normalized advantage。

所以 summary 的 response tokens 会和 retry actions 一样进入 PPO policy loss；它的优化目标就是 retry episode reward，而不是第一轮 reward。

一个重要细节：summary row 的即时 `rewards` 字段是 `0`，因为它不是环境 step；但它的 `episode_rewards` 会在 gather 阶段被设为 `reward2`。训练用的是 `episode_rewards -> token_level_scores/token_level_rewards`，不是 summary row 的即时 raw reward。

## 7. Group、trajectory 与 credit 的字段语义

当前 trace/训练数据中主要字段：

- `uid`: GRPO group id。当前会生成两套：`...:traj1` 和 `...:retry`。
- `traj_uid`: 真实 trajectory id。summary row 和同一次 retry 的 traj2 rows 共享一个 `traj_uid`。
- `ssca_phase`: `traj1` / `summary` / `traj2`。
- `ssca_attempt`: 第一轮是 `1`，summary 和 retry 是 `2`。
- `ssca_step_idx`: summary 是 `-1`，环境 action step 从 `0` 开始。
- `ssca_summary_is_policy_action`: 最新代码中 summary row 为 `True`，其他 row 为 `False`。
- `ssca_reward_source`: 最新代码中 `traj1` 为 `first_attempt_reward`，`summary/traj2` 为 `retry_attempt_reward`。
- `ssca_context_contract`: 最新代码中标记每类 row 的上下文约束。
- `ssca_linked_first_traj_uid`: 最新代码中 retry/summary 会记录对应的 first trajectory，方便 review 配对。

已有 smoke trace 是在新增这些 trace 字段前跑出的，所以旧 trace 中没有 `ssca_summary_is_policy_action` / `ssca_context_contract` / `ssca_linked_first_traj_uid`。下一次运行脚本会自动带上这些字段。

## 8. Retry reset 如何保证是同一个任务

为了让 `traj2` 从同一个 ALFWorld 初始状态开始，我加了 retry reseed 通路：

- `agent_system/environments/env_package/alfworld/envs.py:67`: worker 保存初始 `seed_value`。
- `agent_system/environments/env_package/alfworld/envs.py:78`: `reset(reseed=False)` 支持在 reset 前重新 seed。
- `agent_system/environments/env_package/alfworld/envs.py:153`: `AlfworldEnvs.reset(retry_same_seed=False)` 把参数传给每个 worker。
- `agent_system/environments/env_manager.py:138`: `AlfWorldEnvironmentManager.reset(kwargs)` 读取 `retry_same_seed`。
- `recipe/SSCA/rollout_loop.py:391`: SSCA retry 默认传 `retry_same_seed=True`。

这保证 retry 默认回到同一个 task/game 初始状态。脚本中可用 `SSCA_RETRY_SAME_SEED=False` 关闭。

## 9. 一键脚本与输出位置

脚本绝对路径：

```bash
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_ssca_retry_grpo_alfworld_smoke.sh
```

smoke 运行命令：

```bash
MAX_ENV_STEPS=4 TOTAL_EPOCHS=1 TEST_FREQ=-1 SAVE_FREQ=-1   bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_ssca_retry_grpo_alfworld_smoke.sh
```

dry-run 检查命令：

```bash
DRY_RUN=1 RUN_TAG=check   bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_ssca_retry_grpo_alfworld_smoke.sh
```

输出：

- log: `logs/<experiment_name>.log`
- rollout generations: `EXPS/analysis/022_ssca_retry_smoke/<experiment_name>/rollout_generations/`
- RL trace: `EXPS/analysis/022_ssca_retry_smoke/<experiment_name>/rl_trace/`

脚本默认开启：

- `trainer.rollout_data_dir=$TRACE_DIR/rollout_generations`
- `+trainer.rl_trace_dir=$TRACE_DIR/rl_trace`
- `+trainer.rl_trace_interval=1`
- `+trainer.rl_trace_include_text=True`
- `+trainer.rl_trace_save_dataproto=True`

## 10. 已有 smoke trace 的真实 rollout 样例

样例来源：

```text
EXPS/analysis/022_ssca_retry_smoke/ssca_retry_grpo_qwen25_15b_alfworld_smoke_seed2026_ssca_retry_smoke_20260626_133319
```

注意：这个 smoke 用的是很短的 `MAX_ENV_STEPS=2`，所以这些样例主要用于检查链路和字段，不用于判断算法效果。所有示例里的 episode reward 都是 `0.0`，这是短 smoke 的预期现象。另外，trace 经过 batch padding/divisor 对齐后会有重复 row；下面已经按 `(phase, step, response, traj_uid)` 去重。

#### Example 3: `put some peppershaker on drawer`

- Trace key: `index=3`, first uid `01c8a270-b11f-49bd-af70-5e7f05ccfcd4:traj1`, retry uid `dbf8995d-e7e4-4f2f-9808-a1db97614585:retry`
- Traj links: first `traj_uid=01350a2c-2f99-45bb-b29f-6b5dd4d0e385`, retry `traj_uid=8cf70064-7588-43d3-9c53-304a43dc3b4a`
- First attempt reward: `0.0`
- Retry reward: `0.0`

First attempt rows:
- `traj1 step=0` valid=`True` raw_reward=`0.0` action: `go to cabinet 1`
- `traj1 step=1` valid=`True` raw_reward=`0.0` action: `go to countertop 1`

Summary action row:
- `summary step=-1` valid=`True` raw_reward=`0` episode_reward=`0.0`

Summary output excerpt:

```text
### Self-Summary for Retry

**State Belief:**
- Task: Place peppershaker on drawer.
- Key Objects: Peppershaker, drawers.
- Key Receptacles: Counter, drawers (specifically cabinet 2).
- Constraints: Need a drawer to put the peppershaker in.

**Failure Causes:**
- **Action Selection:** Tried going to cabinets before looking at countertop.
- **Drawer Selection:** Mistakenly selected countertop first instead of drawers.

**Policy Improvement:**
- Structure the action directly after observing the environment.
- Ensure to check drawers after countertop views.
- Prefer moving objects to drawers.

**Retry Plan:**
1. **Start with countertop inspection:** Check the countertop to find edible items like peppers.
2. **Use inventory view:** Quickly check for any cabinets and drawers open to see what is inside.
3. **Select drawer first:** Go directly to a cabinet or drawer and try placing the peppershaker inside using an inventory view if done correctly.
4. **Handle with caution:** Avoid taking item
...
```

Retry step-0 prompt excerpt showing summary visibility:

```text
[Self-summary from the first attempt]
### Self-Summary for Retry

**State Belief:**
- Task: Place peppershaker on drawer.
- Key Objects: Peppershaker, drawers.
- Key Receptacles: Counter, drawers (specifically cabinet 2).
- Constraints: Need a drawer to put the peppershaker in.

**Failure Causes:**
- **Action Selection:** Tried going to cabinets before looking at countertop.
- **Drawer Selection:** Mistakenly selected countertop first instead of drawers.

**Policy Improvement:**
- Structure the action directly after observing the environment.
- Ensure to check drawers after countertop views.
- Prefer moving objects to drawers.

**Retry Plan:**
1. **Start with countertop inspection:** Check the countertop to find edible items like peppers.
2. **Use inventory view:** Quickly check for any cabinets and drawers open to see what is inside.
3. **Select drawer first:** Go directly to a cabinet
```

Retry action rows:
- `traj2 step=0` valid=`True` raw_reward=`0.0` action: `inventory`
- `traj2 step=1` valid=`True` raw_reward=`0.0` action: `go to countertop 1`

#### Example 5: `examine the newspaper with the desklamp`

- Trace key: `index=5`, first uid `d56b77c5-48cd-4f17-91dd-ec3210ceff0a:traj1`, retry uid `8d68d34b-fc99-4ca2-9df9-9a2091aed666:retry`
- Traj links: first `traj_uid=cb7d7958-56f5-4bdf-8095-a3f4df6b6ab3`, retry `traj_uid=1e1aec74-8dfb-4015-8b0d-c7e851edef69`
- First attempt reward: `0.0`
- Retry reward: `0.0`

First attempt rows:
- `traj1 step=0` valid=`True` raw_reward=`0.0` action: `go to armchair 1`
- `traj1 step=1` valid=`True` raw_reward=`0.0` action: `<think> 1. Positioned at armchair 1. 2. Observations: Facing armchair 1 (right arm) - see: a pillow 1. </think> <Action> examine pillow 1 </Action>...`

Summary action row:
- `summary step=-1` valid=`True` raw_reward=`0` episode_reward=`0.0`

Summary output excerpt:

```text
### Self-Summary for Retry

_I am retrying the task of examining the newspaper with a desklamp in the ALFRED environment, starting from the initial state described in the prompt._

**State Belief:**
- **Objects:** armchair 1, diningtable 1, drawer 1, drawer 2, drawer 3, drawer 4, garbagecan 1, shelf 1, sidetable 1, sidetable 2, sidetable 3, sofa 1
- **Receptacles:** None
- **Constraints:** Task is to examine the newspaper with a desklamp on armchair 1

**Failure / Success Causes:**
- **Task:** Examining the newspaper using the desk lamp
- **Initial Mistake:**
  - **Action:** Went to armchair 1 instead of examining the newspaper at the dining table or any other potential locations with a newspaper.
- **Decision Making:** Initially, the action "go to armchair 1" was chosen, which was outside the admissible actions given in the prompt.

- **Improvement Directions:**
  - The next step should examine the dining table or any potential newspaper locations using the desk lamp as it is the prim
...
```

Retry step-0 prompt excerpt showing summary visibility:

```text
[Self-summary from the first attempt]
### Self-Summary for Retry

_I am retrying the task of examining the newspaper with a desklamp in the ALFRED environment, starting from the initial state described in the prompt._

**State Belief:**
- **Objects:** armchair 1, diningtable 1, drawer 1, drawer 2, drawer 3, drawer 4, garbagecan 1, shelf 1, sidetable 1, sidetable 2, sidetable 3, sofa 1
- **Receptacles:** None
- **Constraints:** Task is to examine the newspaper with a desklamp on armchair 1

**Failure / Success Causes:**
- **Task:** Examining the newspaper using the desk lamp
- **Initial Mistake:**
  - **Action:** Went to armchair 1 instead of examining the newspaper at the dining table or any other potential locations with a newspaper.
- **Decision Making:** Initially, the action "go to armchair 1" was chosen, which was outside the admissible actions given in the prompt.

- **Improvement
```

Retry action rows:
- `traj2 step=0` valid=`True` raw_reward=`0.0` action: `go to diningtable 1`
- `traj2 step=1` valid=`False` raw_reward=`0.0` action: `go to diningtable 1`

#### Example 7: `put a candle in toilet`

- Trace key: `index=7`, first uid `ec0790b3-d0b0-4ea4-979e-e493aaa20964:traj1`, retry uid `81ccc9d4-2be1-42c6-ab78-cc5d8a1f3a61:retry`
- Traj links: first `traj_uid=18463bdd-5e0d-4f93-94d0-ead0df38d06f`, retry `traj_uid=a673f858-c337-4f54-9c35-2945117e73c6`
- First attempt reward: `0.0`
- Retry reward: `0.0`

First attempt rows:
- `traj1 step=0` valid=`True` raw_reward=`0.0` action: `go to drawer 1`
- `traj1 step=1` valid=`True` raw_reward=`0.0` action: `open drawer 1`

Summary action row:
- `summary step=-1` valid=`True` raw_reward=`0` episode_reward=`0.0`

Summary output excerpt:

```text
**Self-Summary for Retry:**
---

**Step-by-Step Analysis of the First Trajectory:**

**1. Initial Location and Observation:**
   - Agent starts in the room without a particular task or goal in mind.
   - Actions taken to move around the room without a clear objective.

**2. Failed Actions and Decisions:**
   - Attempted to go to the drawer 1 directly, without a clear plan.
   - Opened the drawer but found only a cloth, which is unrelated to the task and has no information that would help the agent focus on the next step.

**3. Better Next Move:**
   - Recognizing that opening drawer 1 was an incorrect and illogical decision.
   - Choosing to stay put and mentally preparing to rethink the path forward.

**4. Updated Action Process:**
   - Changed reasoning to inspect the environment carefully before attempting any action.
   - Started by exploring other accessible areas without immediately committing to a specific action.

**5. New Trajectory:**
   - Investigated countertop 1, drawer 2,
...
```

Retry step-0 prompt excerpt showing summary visibility:

```text
[Self-summary from the first attempt]
**Self-Summary for Retry:**
---

**Step-by-Step Analysis of the First Trajectory:**

**1. Initial Location and Observation:**
   - Agent starts in the room without a particular task or goal in mind.
   - Actions taken to move around the room without a clear objective.

**2. Failed Actions and Decisions:**
   - Attempted to go to the drawer 1 directly, without a clear plan.
   - Opened the drawer but found only a cloth, which is unrelated to the task and has no information that would help the agent focus on the next step.

**3. Better Next Move:**
   - Recognizing that opening drawer 1 was an incorrect and illogical decision.
   - Choosing to stay put and mentally preparing to rethink the path forward.

**4. Updated Action Process:**
   - Changed reasoning to inspect the environment carefully before attempting any action.
   - Started by exploring oth
```

Retry action rows:
- `traj2 step=0` valid=`True` raw_reward=`0.0` action: `go to countertop 1`
- `traj2 step=1` valid=`False` raw_reward=`0.0` action: `take candle 3 from countertop 1`

## 11. 当前实现与可能的下一步

当前版本刻意保持为“第一版、最小侵入”：

- 不改原始 GRPO trainer 主路径。
- 不改原始 GRPO 脚本，而是使用 `recipe/SSCA` 和 `scripts/run_ssca_retry_grpo_alfworld_smoke.sh`。
- summary 作为 actor response 进入同一 PPO/GRPO loss。
- summary 的 reward 来源是 retry outcome reward。

已知限制：

1. 短 smoke 几乎全是 0 reward，所以只能验证能否跑通和数据结构是否对齐，不能说明 summary 已经学到东西。
2. 当前 feedback 只使用 traj1 结束后的 outcome、invalid count、final observation 等，没有更细粒度的中途 reward attribution。
3. 当前 summary 与 traj2 共享 retry reward；如果后续想强调“summary 是否提升 retry”，可以考虑新版本使用 `reward2 - reward1` 或 success delta 作为 summary/retry credit，但那就不再是完全标准 GRPO outcome path。
4. GRPO group 当前沿用 repo 的 multi-turn row-level 处理方式：同一 `uid` 下的 summary row 与 action rows 都参与 group normalization。这个和现有 multi-turn GRPO/GiGPO 数据流一致，但如果要做“trajectory-level only”的 baseline，需要额外控制 `compute_mean_std_cross_steps` 或单独实现 advantage 路径。
5. 旧 smoke trace 没有新增的 `ssca_context_contract` 等字段；需要新跑一次脚本才能看到这些更清楚的 review 字段。

## 12. Summary 质量审计与 Prompt 改造建议

对最新 smoke trace 的逐条人工检查显示：当前 summary prompt 能让模型写出结构化“反思”，但不能稳定产出忠实、可执行、能约束 retry policy 的 summary。典型问题包括：

- summary 编造未在 trajectory 中出现的 object / receptacle，例如把不存在或未观察到的物体写进 state belief。
- summary 把有效动作误判为失败原因，或把 invalid / malformed action 的根因描述错。
- retry plan 太泛，例如“更仔细探索”“检查可能位置”，没有给出 ALFWorld 风格的具体行动顺序。
- 部分 summary 诱导 traj2 重复同一动作，例如连续 `go to countertop 1` 或连续 `go to diningtable 1`。
- traj2 仍会输出 `[Action] ...`、markdown、缺少 `<think>/<action>` 等 malformed response，说明 summary prompt 没有明确约束 retry 输出格式。

因此下一版不建议继续把 full trajectory 原样塞进 summary prompt。当前 full traj dump 中包含大量重复的 system/task instruction、history wrapper、admissible action list 和长文本截断碎片，这些信息对 summary 反而是噪声。更合理的做法是仿照 ALFWorld 自己给 agent 的 observation 方式，给 summary 一个压缩后的环境状态轨迹，而不是完整 prompt 轨迹。

### 12.1 将 full trajectory 改成 compact obs trace

建议新增 `summary_context_mode=compact_obs_trace`，把 summary prompt 中的 `[First trajectory]` 从完整 `obs["text"]` 改成以下结构：

```text
[Task]
find two newspaper and put them in sofa.

[Initial scene]
You are in the middle of a room. Looking quickly around you, you see ...

[Compact first attempt]
Step 1
Obs: <raw environment observation / anchor_obs only>
Action output: <raw model response, clipped or omitted unless malformed>
Projected action: inventory
Valid: true
Env feedback: You are not carrying anything.

Step 2
Obs: You are not carrying anything.
Projected action: look
Valid: false
Malformed reason: contains Chinese / missing action tag / not one exact admissible action
Env feedback: You are in the middle of a room. Looking quickly around you, you see nothing.

[Outcome]
Reward: 0.0
Won: false
Invalid action count: 3
Final observation: ...
```

核心原则：

- 只保留 ALFWorld 原始 observation / `anchor_obs`，不要重复完整 prompt 模板。
- task 和 initial scene 只出现一次。
- 每步只保留 `Obs -> projected action -> valid -> env feedback`。
- 默认不保存每步完整 admissible actions；只在 invalid 时保存当步 admissible action excerpt，用于解释为什么 invalid。
- raw model response 默认截断；只有 malformed 时保留，用于让 summary 识别格式错误。
- 对长 episode 只保留最近 `k` 步、失败附近步骤、首次看到目标物的步骤、invalid 步和 final observation。

这样 summary 看到的是“环境变化轨迹”，不是“prompt 日志”。这和 ALFWorld agent 每一步收到的 observation 形式更一致，也能降低 hallucination 和重复行动。

### 12.2 Summary prompt 应改成 evidence-grounded retry plan

当前 instruction 太像开放式反思，容易产生漂亮但不可执行的文本。建议改成更硬的 schema：

```text
You are writing an evidence-grounded retry memory for the same ALFWorld task.
Use only facts supported by the compact first attempt.
Do not invent objects, receptacles, locations, or actions.

Return exactly these fields:

Task:
Observed facts:
- target object observed: yes/no/unknown; evidence step:
- target receptacle observed: yes/no/unknown; evidence step:
- checked locations:
- empty or unhelpful locations:
- inventory state:

Failure diagnosis:
- invalid or malformed actions:
- repeated or wasted actions:
- missing necessary subgoal:

Retry plan:
- search priority:
- first action intention:
- next action intention if target not found:
- action/output format rule:

Rules:
1. If the target object was not observed, write "target not observed yet".
2. Mention only object/receptacle names appearing in observations, actions, or the original task.
3. Give concrete ALFWorld-style action intentions, not generic advice.
4. Keep under 160 words.
```

这里的目标不是让 summary 文学化，而是让它成为一个可被 retry policy 消化的短期工作记忆。

### 12.3 加 summary quality gate

在 reward 还很稀疏、短 smoke 大多 `0` reward 时，坏 summary 和好 summary 没有足够强的 outcome 区分。如果直接把所有 summary 都放进 `uid_retry` 训练，模型可能学到 verbose reflection，而不是有效 memory。建议至少记录并可选过滤：

- `faithfulness_pass`: summary 中的 object / receptacle / action 是否都能在 task、observations、actions 或 final observation 中找到证据。
- `actionability_pass`: 是否包含具体 search priority 或 action intention。
- `format_warning_pass`: 是否明确提醒 retry 输出必须使用 `<think>...</think><action>...</action>` 且只选一个 admissible action。
- `retry_follow_rate`: traj2 前几步是否遵循 summary 中的 search priority。
- `invalid_delta`: `invalid_count(traj2) - invalid_count(traj1)`，若变差则降低 summary credit。

第一版可以只打日志，不过滤训练；下一版可以加入 gate：

```text
summary_train_weight =
  1.0 if faithfulness_pass and actionability_pass
  0.3 if only actionability_pass
  0.0 if hallucination or retry invalid rate worsens significantly
```

### 12.4 推荐 ablation

为了确认“简化 trajectory”是否真的更好，建议下一批实验至少比较：

- `SSCA-FullTraj`: 当前版本，summary 看完整 prompt/history dump。
- `SSCA-CompactObsTrace`: summary 只看 compact observation/action/feedback trace。
- `SSCA-CompactObsTrace-NoRawResponse`: 不给 raw model response，只给 projected action 和 valid/malformed reason。
- `SSCA-RandomSummary`: 用随机或 shuffle summary 控制“只是增加上下文长度”的收益。
- `SSCA-NoSummary`: reset 后普通 retry，不提供 summary。

主要指标：

```text
summary_faithfulness_fail_rate
summary_actionability_pass_rate
traj2_invalid_rate
retry_follow_rate
reward2 - reward1
success2 - success1
```

如果 `CompactObsTrace` 明显降低 hallucination / invalid rate，即使短期 reward 还没涨，也说明 prompt 方向更健康。

## 13. Review 建议

人工 review 时建议看三件事：

1. summary prompt 是否真的包含 traj1 细节和 outcome feedback。
2. traj2 prompt 是否只包含当前 ALFWorld observation/history + summary，不泄露 traj1 明细。
3. summary row 的 `uid/traj_uid/episode_rewards` 是否和后续 traj2 rows 对齐。

下一次新 trace 中可以直接 grep：

```bash
python3 - <<'PY'
import json, pathlib, collections
rows = pathlib.Path('EXPS/analysis/022_ssca_retry_smoke/<experiment>/rl_trace/step_000001/rows.jsonl')
cs = collections.Counter()
for line in rows.open():
    nt = json.loads(line)['non_tensors']
    cs[(nt.get('ssca_phase'), nt.get('ssca_context_contract'), nt.get('ssca_reward_source'))] += 1
print(cs)
PY
```
