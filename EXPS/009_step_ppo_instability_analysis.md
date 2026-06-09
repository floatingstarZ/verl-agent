# StepPPO-v1 不稳定性与效果偏弱分析

本文档记录当前 StepPPO-v1 (`v1_step_norm`) 在 WebShop 上表现不稳定、效果弱于 GiGPO 的初步诊断。分析基于现有日志和可视化结果：

- GiGPO 三个 seed：`2026 / 2077 / 2501`，均完整跑到 step 250。
- StepPPO-v1 seed `2026`：当前日志跑到 step 185，是 partial run。

相关图：

- `VISULIZATION/figures/fig_gigpo_vs_step_ppo_val_metrics.png`
- `VISULIZATION/figures/fig_step_ppo_instability_diagnostics.png`

## 1. 主要现象

StepPPO-v1 相比 GiGPO 明显更不稳定，且当前效果较差。与 GiGPO seed2026 对齐到 step185 比较，StepPPO-v1 平均落后：

```text
task_score:   -0.153
success_rate: -0.162
text_score:   -1.95
```

当前最后一个 validation 点为：

```text
StepPPO-v1 step185:
task_score=0.703
success_rate=0.492
text_score=3.866
```

而 GiGPO seed2026 在相同 step 附近已经显著更高：

```text
GiGPO seed2026 step185:
task_score≈0.856
success_rate≈0.684
text_score≈6.698
```

从曲线看，StepPPO-v1 的 validation task score 和 success rate 有明显回撤，例如 early stage 出现过较大的 task score drop，后续虽然恢复，但整体仍落后 GiGPO。

## 2. 当前 StepPPO-v1 的算法形式

当前 `v1_step_norm` 的 final advantage 是：

```tex
A_t^{final,v1}
=
\frac{
\widehat{A}_t^{epi} + w \widehat{A}_t^{step}
}{1+w}
```

其中：

```tex
\widehat{A}_t^{epi}
= \operatorname{Whiten}_{\mathcal{B}_{step}}(A_t^{epi})
```

```tex
\widehat{A}_t^{step}
= \operatorname{Whiten}_{\mathcal{B}_{step}}(A_t^{step})
```

step advantage 来自 shared value head 上的 TD/GAE：

```tex
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

```tex
A_t^{step}
= \delta_t + \gamma \lambda A_{t+1}^{step}
```

value head 当前挂在 actor backbone 上，并且默认配置为：

```yaml
actor_rollout_ref:
  actor:
    value_head:
      enable: True
      detach_value_backbone: False
      value_loss_coef: 0.1
```

这意味着 value loss 会回传到 actor backbone。

## 3. 诊断证据

### 3.1 valid action ratio 明显偏低

StepPPO-v1 的 valid action ratio 明显低于 GiGPO：

```text
StepPPO valid_action_ratio mean: 0.803
GiGPO   valid_action_ratio mean: 0.980
```

GiGPO 很快接近 `1.0`，而 StepPPO-v1 长时间在 `0.75-0.9` 附近震荡。对于 WebShop 这类 action-format-sensitive 任务，valid action ratio 低会导致：

- 环境反馈更噪。
- 有效探索效率更低。
- reward/advantage 估计更不稳定。
- policy 更容易被 invalid action penalty 牵制。

### 3.2 response clipping 更严重

StepPPO-v1 的 response clipping 明显更高：

```text
StepPPO response_length/clip_ratio mean: 0.0259
GiGPO   response_length/clip_ratio mean: 0.0043

StepPPO last clip_ratio: 0.056
GiGPO   last clip_ratio: 0.002
```

这说明 StepPPO-v1 更容易生成过长 response，并被 `max_response_length` 截断。在 agentic action 任务里，截断通常意味着：

- action 格式不完整。
- tool/action syntax 更容易损坏。
- reward 与 valid action 指标都会变差。

这也和 valid action ratio 偏低相互印证。

### 3.3 value loss 长期较大且波动

StepPPO-v1 中 value loss 长期在较高水平波动：

```text
actor/value_loss mean: 3.61
actor/value_loss max:  8.07
```

分阶段看，value loss 在中后期仍然较高，例如 step101-150 阶段均值接近 `4.95`。这说明 value head 对 step-level return/GAE target 的拟合并不稳定。

考虑到 WebShop 的 reward 高方差、稀疏、且 trajectory 长度不固定，step-level return 本身是一个难学的目标。critic 不准时，GAE correction 可能不是降噪，而是给 policy 引入错误 credit assignment。

### 3.4 actor grad norm 明显更高，且后期不下降

对比 GiGPO：

```text
StepPPO grad_norm mean: 10.13
GiGPO   grad_norm mean:  5.78
```

分阶段趋势更明显：

```text
StepPPO:
step 1-50:    grad_norm mean≈5.23
step 51-100:  grad_norm mean≈9.08
step 101-150: grad_norm mean≈13.62
step 151-200: grad_norm mean≈13.47

GiGPO:
step 1-50:    grad_norm mean≈7.93
step 51-100:  grad_norm mean≈8.42
step 101-150: grad_norm mean≈5.60
step 151-200: grad_norm mean≈3.67
step 201-250: grad_norm mean≈3.28
```

GiGPO 后期梯度逐渐下降，说明 policy 逐渐稳定；StepPPO-v1 后期梯度反而维持高位，说明优化仍然震荡。

这和 shared value head 的训练压力有关：value loss 回传到 actor backbone，可能持续改变 policy 表征。

### 3.5 policy gradient signal 可能被压弱

StepPPO-v1 里 episode advantage 和 step advantage 都先 whitening，再做平均：

```tex
A_t^{final,v1}
=
\frac{
\widehat{A}_t^{epi} + w \widehat{A}_t^{step}
}{1+w}
```

这可能压低最终 policy advantage 的尺度。日志中也能看到：

```text
StepPPO actor/pg_loss std: 0.052
GiGPO   actor/pg_loss std: 0.242
```

即 StepPPO-v1 的 policy gradient loss 波动更小，可能意味着 policy update 的有效信号变弱。

如果 policy gradient signal 变弱，而 value loss 仍然较大并回传 backbone，就会形成一个不利组合：

```text
policy 优化信号较弱
value regression 信号较强
shared backbone 被 value loss 持续牵引
最终生成行为变差
```

## 4. 初步原因排序

当前更可能的原因按优先级排序如下。

### 原因 1：value loss 干扰 actor backbone

这是目前最值得怀疑的原因。当前：

```yaml
detach_value_backbone: False
value_loss_coef: 0.1
```

value loss 会回传到 actor backbone。由于 value target 本身高方差且难学，actor 表征可能被 critic 目标牵引，导致 policy 生成质量下降。valid action ratio 低和 response clipping 高都支持这个判断。

### 原因 2：v1_step_norm 的 advantage 混合压弱了 policy signal

`v1_step_norm` 对 `A_epi` 和 `A_step` 分别 whitening 后再除以 `1+w`。这可能导致 final advantage 尺度偏小，从而 policy update 变弱。

如果 policy update 变弱，但 value loss 继续强力更新 shared backbone，训练会偏离原本的 GRPO/GiGPO 稳定轨道。

### 原因 3：step critic 早期不准，GAE correction 反而引入噪声

StepPPO 的 step correction 依赖 value head：

```tex
A_t^{step} = \operatorname{GAE}(r_t, V(s_t), V(s_{t+1}))
```

早期 value head 不准时，TD error 可能给出错误 credit assignment。尤其 WebShop 的 reward 稀疏且 action 质量强依赖格式，critic 学习目标更困难。

### 原因 4：当前 step advantage normalization 粒度可能不理想

当前 step advantage 是 batch-level whitening：

```tex
\widehat{A}^{step}_i
=
\frac{A^{step}_i - \mu_{\mathcal{B}}}{\sigma_{\mathcal{B}} + \epsilon}
```

它不是按 task uid、trajectory、anchor state group 做 normalize。这可能破坏 step-level credit 的相对语义，也可能和 episode-level GRPO advantage 的 group normalization 不匹配。

## 5. 下一轮实验建议

不要立即大规模多 seed。建议先做小规模单 seed 诊断，明确问题来源。

### 5.1 v0_base：不归一化 step advantage

当前已经规划的 `v0_base`：

```yaml
algorithm:
  step_ppo:
    normalize_step_advantage: False
```

对应公式：

```tex
A_t^{final,v0}
=
\frac{
\widehat{A}_t^{epi} + w A_t^{step}
}{1+w}
```

这个实验回答：

```text
v1 差是不是因为 step advantage norm 后 policy signal 太弱？
```

如果 v0 明显更好，说明当前 `step_norm` 设计可能不合适。

### 5.2 detach value backbone

建议新增一个版本：

```yaml
actor_rollout_ref:
  actor:
    value_head:
      detach_value_backbone: True
```

目标是让 value loss 只训练 value head，不更新 actor backbone。这个实验回答：

```text
StepPPO 不稳定是否主要来自 value loss 对 actor backbone 的干扰？
```

如果 detach 后 valid action ratio 提升、response clipping 下降、grad norm 下降，那么主要问题就是 shared backbone 被 critic 干扰。

### 5.3 降低 value_loss_coef

建议尝试：

```yaml
actor_rollout_ref:
  actor:
    value_head:
      value_loss_coef: 0.02
```

或：

```yaml
value_loss_coef: 0.01
```

目标是降低 value regression 对 actor update 的影响。

### 5.4 final advantage 混合后再 normalize

可以考虑把当前：

```tex
A_t^{final}
=
\frac{
\widehat{A}_t^{epi} + w \widehat{A}_t^{step}
}{1+w}
```

改为：

```tex
A_t^{final}
=
\operatorname{Whiten}_{\mathcal{B}_{step}}
\left(
\widehat{A}_t^{epi} + w \widehat{A}_t^{step}
\right)
```

这样可以保证最终进入 policy loss 的 advantage 仍有稳定尺度。

### 5.5 补充内部日志

当前还缺少一些关键指标。建议记录：

```text
episode_advantage mean/std/min/max
step_advantage raw mean/std/min/max
step_advantage normalized mean/std/min/max
final_advantage mean/std/min/max
corr(episode_advantage, step_advantage)
value_error = value_pred - step_return
value_error mean/std/min/max
```

这些指标可以直接判断 step correction 是在帮助 episode signal，还是和 episode advantage 冲突。

## 6. 推荐下一步优先级

建议优先跑三个单 seed 诊断版本：

```text
1. v1_step_norm 当前版本
2. v0_base，不 norm step advantage
3. v1_detach_value_backbone，value loss 不回传 actor backbone
```

判断逻辑：

- 如果 `v0_base` 明显好于 `v1_step_norm`，说明 step advantage normalization 或 final mix 尺度处理有问题。
- 如果 `detach_value_backbone=True` 明显更稳定，说明 shared backbone 被 value loss 干扰是主要问题。
- 如果两者都不改善，需要进一步怀疑 step-level critic target 本身太噪，或者当前 step reward/return 定义不适合 WebShop。

## 7. 当前结论

当前 StepPPO-v1 表现差，不应简单归因于“step 信息没用”。更合理的解释是：

```text
当前 shared-backbone critic 的训练方式破坏了 actor 的生成稳定性，
同时 v1_step_norm 的 advantage 混合可能削弱了 policy signal。
```

这导致：

```text
valid action ratio 低
response clipping 高
value loss 波动大
grad norm 后期不下降
validation 曲线回撤明显
最终效果落后 GiGPO
```

因此下一轮实验应优先验证：

```text
1. 是否应 detach value backbone
2. 是否应降低 value_loss_coef
3. 是否应取消 step advantage norm 或混合后再 norm
```
