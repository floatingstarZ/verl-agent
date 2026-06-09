# 006 StepPPO Advantage Normalization 说明

本文档记录 StepPPO 当前 prototype 的 advantage normalization 语义。核心点是：episode-level GRPO advantage 和 step-level critic/GAE advantage 属于不同来源，当前 v1 版本分别归一化后再混合；v0 版本则不对 step advantage 做归一化。

## 1. 符号

一个 agentic trajectory 包含多个环境 step。step `t` 中，agent 接收当前 context/state，并生成一个 action，例如文本动作或 tool call。StepPPO 对每个 action 分配一个 scalar advantage，然后为了复用现有 PPO loss，把这个 scalar 广播到 action 的有效 response tokens 上。

记：

```tex
\tau_i = (s_{i,0}, a_{i,0}, r_{i,0}, ..., s_{i,T_i-1}, a_{i,T_i-1}, r_{i,T_i-1})
```

其中 `i` 是 trajectory index，`t` 是 environment step index。`uid` 表示同一个初始任务的 rollout group，`traj_uid` 表示一条具体 trajectory，`step_idx` 表示 trajectory 内的 step 顺序。

## 2. Episode Advantage

对 group `g` 内 trajectory `i`，episode return 为：

```tex
R_i^{epi} = \sum_t r_{i,t}
```

GRPO-style episode advantage 为：

```tex
A_i^{epi} = \frac{R_i^{epi} - \mu_g}{\sigma_g + \epsilon}
```

这个 scalar 会广播到 trajectory `i` 的每一个 step：

```tex
A_{i,t}^{epi} = A_i^{epi}
```

实现中使用 `traj_uid` 做 episode-level 去重，避免同一 trajectory 的多个 step 被当作多个独立 episode return 计入 group mean/std。

## 3. Step Advantage

step advantage 来自 shared value head 的 GAE。对每个 `traj_uid`，按 `step_idx` 排序后计算：

```tex
\delta_{i,t}=r_{i,t}+\gamma(1-d_{i,t})V^{old}_{i,t+1}-V^{old}_{i,t}
```

```tex
A_{i,t}^{step}=\delta_{i,t}+\gamma\lambda(1-d_{i,t})A_{i,t+1}^{step}
```

value target 为：

```tex
R_{i,t}^{step}=A_{i,t}^{step}+V^{old}_{i,t}
```

这些量都在 environment step 维度上计算，不在 token 维度上计算。

## 4. v1 当前归一化

当前 v1 版本在 valid step batch 上分别 whitening 两个信号：

```tex
\widehat A^{epi} = Whiten_{B_step}(A^{epi})
```

```tex
\widehat A^{step} = Whiten_{B_step}(A^{step})
```

最终 advantage 为：

```tex
A_t^{final,v1}=\frac{\widehat A_t^{epi}+w\widehat A_t^{step}}{1+w}
```

这个设计让两个信号 scale 可比，但也可能削弱原始 episode-level policy signal，并把 noisy critic ordering 引入 policy update。

## 5. v0 Base 归一化

v0 保留 episode normalization，但不对 step GAE advantage 做 whitening：

```tex
A_t^{final,v0}=\frac{\widehat A_t^{epi}+wA_t^{step}}{1+w}
```

它用于定位 batch-level step whitening 是否是 instability 来源。

## 6. 与 GiGPO 的 `traj_uid` 关系

使用 `traj_uid` 不代表 StepPPO 等价于 GiGPO。二者都利用 trajectory identity，但用途不同：

- StepPPO 用 `traj_uid` 做 temporal GAE 和 episode return 去重。
- GiGPO 用 trajectory identity 与 anchor observation 构造 comparable step groups。

因此，StepPPO 和 GiGPO 都使用 step 信息，但 credit assignment 机制不同。

## 7. 后续可选版本

后续可以测试：

- 按 `uid` 做 step normalization。
- trajectory-wise centering。
- 按 anchor observation group 做 normalization。
- 先 mixing 再 whitening final advantage。
- value target normalization 与 policy advantage normalization 分离。
