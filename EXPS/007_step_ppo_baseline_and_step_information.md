# StepPPO Baseline 与 Step 信息使用说明

本文档记录当前关于 StepPPO baseline 选择的共识，尤其是：为什么 GiGPO 应该作为主要 baseline，以及如果希望对比“不利用 step 信息”的方法，应该如何设计实验。

## 1. 核心判断

如果研究问题是：

```text
StepPPO 的 learned step-level critic correction 是否优于已有的 step-aware 方法？
```

那么主 baseline 应该是 **GiGPO**，而不是 GRPO。

原因是：

- GRPO 只使用 episode-level outcome reward 的组内相对优势。
- GiGPO 已经显式利用 step 信息，包括 `traj_uid`、step discounted return、`anchor_obs` step grouping。
- StepPPO 也利用 step 信息，只是把 GiGPO 的非参数 step relative advantage 换成了 learned value head 加 TD/GAE correction。

因此，若要公平比较“利用 step 信息的方法”，更合理的主对比是：

```text
GiGPO: 非参数 step-aware baseline
StepPPO: learned critic step-aware method
```

而 GRPO 更适合作为“不显式利用 step 信息”的基础 baseline。

## 2. 方法之间的关系

当前应把三个方法放在同一个实验谱系里理解：

| 方法 | 是否显式利用 step 信息 | 作用 |
|---|---:|---|
| GRPO | 否 | non-step baseline |
| GiGPO | 是 | non-parametric step-aware baseline |
| StepPPO | 是 | learned-critic step-aware method |

其中：

- **GRPO**：每个 episode 只有一个 outcome advantage，episode 内所有 step/action 共享这个 advantage。
- **GiGPO**：利用 trajectory 内的 step reward/discounted return，并通过 `anchor_obs` 构造 step-level comparison group。
- **StepPPO**：利用 `traj_uid + step_idx` 沿 trajectory 做 TD/GAE，并用 actor backbone 上的 shared value head 预测 step value。

## 3. 什么叫“不利用 step 信息”

这里的“不利用 step 信息”应定义为：

```text
policy advantage 不使用 step reward、step index、step return、TD error、GAE、anchor_obs grouping 等 step-level signal。
```

在这个定义下，GRPO 是最干净的 non-step baseline。

GRPO 的形式可以写成：

```tex
A_t = A^{epi}
```

也就是说，虽然训练样本中仍然包含多个 action/step，但是每个 step/action 的 policy advantage 都只来自 episode-level GRPO advantage。

## 4. 为什么 classic PPO 不一定是干净的 non-step baseline

直觉上，PPO 似乎可以作为 no-step baseline。但在 agentic multi-step task 里，如果 PPO 使用 value function 和 TD/GAE，那么它本质上仍然利用了 step sequence：

```tex
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

只要使用 `r_t`、`V(s_t)`、`V(s_{t+1})` 或 step-level return，它就已经进行了 temporal credit assignment。因此，如果我们严格区分“是否利用 step 信息”，classic PPO/GAE 并不是一个干净的 no-step baseline。

相比之下，GRPO 更适合作为不利用 step 信息的 baseline，因为它只依赖 episode-level outcome relative advantage。

## 5. 推荐实验矩阵

建议当前 StepPPO 实验至少包含以下几组：

| 实验 | 作用 |
|---|---|
| GRPO | non-step baseline，衡量不利用 step 信息时的表现 |
| GiGPO | step-aware baseline，衡量已有非参数 step 信息利用方法的表现 |
| StepPPO | 主方法，衡量 learned value head + TD/GAE step correction 的表现 |
| StepPPO-no-step-adv | 框架内 ablation，验证 step advantage 是否真正贡献收益 |

更具体地说：

```text
主 baseline: GiGPO
non-step baseline: GRPO
ablation baseline: StepPPO with step_advantage_w=0
```

## 6. StepPPO-no-step-adv Ablation

最容易实现的 no-step ablation 是设置：

```yaml
algorithm:
  step_ppo:
    step_advantage_w: 0.0
```

此时 StepPPO 的 final advantage 退化为：

```tex
A_t^{final}
=
\frac{\widehat{A}_t^{epi} + 0 \cdot \widehat{A}_t^{step}}{1+0}
=
\widehat{A}_t^{epi}
```

这个 ablation 可以回答：

```text
step-level advantage correction 对 policy update 是否有贡献？
```

不过需要注意：如果 value head 仍然启用并训练，那么该实验仍然保留了 value head 的计算和优化开销。因此它不是最纯粹的 GRPO baseline，而是 StepPPO 框架内部的 ablation。

## 7. 更严格的 no-value Ablation

如果想进一步排除 shared value head 本身对训练动态的影响，可以设计一个更严格的 ablation：

```text
StepPPO-no-value
```

它应满足：

- 不启用 shared value head。
- 不计算 step value。
- 不计算 step TD/GAE。
- policy advantage 完全使用 episode-level GRPO advantage。

这个版本本质上会非常接近 GRPO，但仍可用于排查 StepPPO 代码路径是否引入了额外行为差异。

当前实现中最容易先做的是 `StepPPO-no-step-adv`，即只设置 `step_advantage_w=0.0`。`StepPPO-no-value` 需要额外代码路径或单独入口。

## 8. 推荐汇报方式

实验报告中建议明确分三类比较：

### 8.1 Non-step Baseline

```text
GRPO
```

用于回答：不利用 step 信息时，episode-level outcome optimization 的上限如何。

### 8.2 Step-aware Baseline

```text
GiGPO
```

用于回答：已有非参数 step-aware 方法能达到什么水平。

### 8.3 Learned Step-aware Method

```text
StepPPO
```

用于回答：用 learned value head 做 TD/GAE step correction 是否带来增益。

### 8.4 Ablation

```text
StepPPO-no-step-adv
```

用于回答：StepPPO 的收益是否来自 step advantage，而不是训练脚本、batch 构造、value head 附加路径等其他因素。

## 9. 当前结论

当前更合理的 baseline 设置应为：

```text
主 baseline: GiGPO
辅助 baseline: GRPO
框架内 ablation: StepPPO-no-step-adv
```

其中：

- 如果 StepPPO 优于 GRPO，但不优于 GiGPO，只能说明 learned step correction 优于 no-step baseline，但没有超过已有 step-aware 方法。
- 如果 StepPPO 优于 GiGPO，才更能说明 learned value-based step credit assignment 有额外价值。
- 如果 StepPPO 与 `StepPPO-no-step-adv` 差异很小，则需要怀疑 step correction 的实际贡献不足，或者当前 value learning/normalization 设计还不够有效。
