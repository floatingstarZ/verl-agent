# StepPPO 实验重规划与版本定义

本文档记录当前 StepPPO 实验的重新规划。核心变化是：GiGPO 已经作为 step-aware baseline 跑过，接下来需要补跑 GRPO baseline，并将 StepPPO 明确拆成两个版本：`v1_step_norm` 和 `v0_base`。

## 1. 实验目标

当前问题不再只是验证 StepPPO 是否优于 GRPO，而是要区分三类方法：

| 方法 | 是否显式利用 step 信息 | 实验作用 |
|---|---:|---|
| GRPO | 否 | non-step baseline |
| GiGPO | 是 | 已有 step-aware baseline，已跑过 |
| StepPPO-v1_step_norm | 是 | 当前版本，step advantage 做归一化 |
| StepPPO-v0_base | 是 | 去掉 step advantage 归一化的基础版本 |

其中 GiGPO 已经跑过；本轮需要补齐：

```text
StepPPO-v1_step_norm seed=2026
StepPPO-v0_base seed=2026
GRPO seed=2026
```

## 2. 共同定义

每个 agent episode 包含多个 step。对第 `t` 个 step/action，StepPPO 先得到两个标量 advantage：

- episode-level advantage：`A_t^{epi}`
- step-level critic advantage：`A_t^{step}`

其中 episode advantage 来自 GRPO-style episode outcome normalization，step advantage 来自沿同一条 `traj_uid` 的 TD/GAE。

Step-level TD residual 为：

```tex
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

Step-level GAE 为：

```tex
A_t^{step}
= \delta_t + \gamma \lambda A_{t+1}^{step}
```

最终 action-level advantage 会 broadcast 到该 action 的所有有效 response tokens：

```tex
A_{t,l}^{token} = A_t^{final} \cdot m_{t,l}
```

其中 `m_{t,l}` 是 response/loss mask。

## 3. v1_step_norm：当前版本

`v1_step_norm` 是当前默认 StepPPO 实现。它会对 episode advantage 和 step advantage 都做 batch-level whitening，然后再混合。

Episode advantage 归一化：

```tex
\widehat{A}_t^{epi}
= \operatorname{Whiten}_{\mathcal{B}_{step}}(A_t^{epi})
```

Step advantage 归一化：

```tex
\widehat{A}_t^{step}
= \operatorname{Whiten}_{\mathcal{B}_{step}}(A_t^{step})
```

其中：

```tex
\operatorname{Whiten}_{\mathcal{B}}(x_i)
=
\frac{x_i - \mu_{\mathcal{B}}}{\sigma_{\mathcal{B}} + \epsilon}
```

最终混合：

```tex
A_t^{final,v1}
=
\frac{
\widehat{A}_t^{epi} + w \widehat{A}_t^{step}
}{1+w}
```

对应配置：

```yaml
algorithm:
  step_ppo:
    normalize_episode_advantage: True
    normalize_step_advantage: True
    step_advantage_w: 1.0
```

该版本的实验名使用：

```text
step_ppo_v1_step_norm_qwen2.5_1.5b_4gpu_paper_align_full_e250_seed2026
```

## 4. v0_base：不归一化 step advantage 的版本

`v0_base` 保留 StepPPO 的 step-level critic/GAE 计算，但不再对 `A_t^{step}` 做 batch-level whitening。它直接将 raw GAE step advantage 与归一化后的 episode advantage 混合。

Episode advantage 仍然使用：

```tex
\widehat{A}_t^{epi}
= \operatorname{Whiten}_{\mathcal{B}_{step}}(A_t^{epi})
```

Step advantage 不再归一化：

```tex
\widetilde{A}_t^{step} = A_t^{step}
```

最终混合：

```tex
A_t^{final,v0}
=
\frac{
\widehat{A}_t^{epi} + w A_t^{step}
}{1+w}
```

对应配置：

```yaml
algorithm:
  step_ppo:
    normalize_episode_advantage: True
    normalize_step_advantage: False
    step_advantage_w: 1.0
```

该版本的实验名使用：

```text
step_ppo_v0_base_qwen2.5_1.5b_4gpu_paper_align_full_e250_seed2026
```

## 5. v0_base 与 v1_step_norm 的区别

两个版本的区别只在 policy advantage 中 step component 的尺度处理：

| 项 | v1_step_norm | v0_base |
|---|---|---|
| episode advantage | batch-level whitening | batch-level whitening |
| step advantage | batch-level whitening | 不做 whitening |
| value head | 启用 | 启用 |
| step TD/GAE | 启用 | 启用 |
| final mix | `A_epi_hat + w A_step_hat` | `A_epi_hat + w A_step_raw` |

需要注意：`v0_base` 不是 GRPO。它仍然利用了 step reward、`traj_uid + step_idx`、value head 和 TD/GAE。它只是去掉了 step advantage normalization。

## 6. GRPO baseline

GRPO 作为 non-step baseline，policy advantage 只来自 episode-level outcome relative advantage：

```tex
A_t^{GRPO} = A_t^{epi}
```

它不使用 step reward、step TD/GAE、value head 或 `anchor_obs` step grouping。因此它可以回答：如果不显式利用 step 信息，episode-level relative optimization 能达到什么水平。

GRPO 实验名使用：

```text
grpo_qwen2.5_1.5b_4gpu_paper_align_full_e250_seed2026
```

## 7. 本轮批量运行顺序

本轮只使用一个 seed：

```text
seed = 2026
```

运行顺序固定为：

```text
1. StepPPO-v1_step_norm seed=2026
2. StepPPO-v0_base seed=2026
3. GRPO baseline seed=2026
```

对应脚本：

```bash
bash exps/run_webshop_step_ppo_v1_v0_grpo_seed2026.sh
```

## 8. 代码实现方式

`v0_base` 不需要新增核心算法分支，因为当前 StepPPO 已经支持配置项：

```yaml
algorithm.step_ppo.normalize_step_advantage
```

代码路径在：

```text
verl/trainer/ppo/step_ppo_algos.py::compute_step_ppo_advantage_return
```

关键逻辑为：

```python
if normalize_step_advantage:
    step_advantages = _whiten_1d(step_advantages, epsilon=epsilon)

final_step_advantages = (
    episode_advantages + step_advantage_w * step_advantages
) / (1.0 + step_advantage_w)
```

因此：

- `v1_step_norm`：`normalize_step_advantage=True`
- `v0_base`：`normalize_step_advantage=False`

## 9. 后续扩展

如果 seed=2026 的结果正常，再扩展到多 seed：

```text
2026, 2077, 2501
```

届时建议对以下方法都跑相同 seeds：

```text
GRPO
GiGPO
StepPPO-v1_step_norm
StepPPO-v0_base
```
