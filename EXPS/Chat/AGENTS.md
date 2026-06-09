# Chat 归档约定

本目录用于保存用户与 Codex 关于实验、算法和代码实现的聊天原文记录。

## 核心原则

- 使用 append-only 模式记录对话。
- 不重写、压缩或摘要已经记录过的对话条目。
- 新对话只追加到当前 raw chat log 的末尾，或在文件过长时新建下一个编号文件。
- 记录用户与 Codex 的原文内容；公式可以整理成 LaTeX 熟悉的展示格式。
- 不逐字记录超长工具输出、日志 dump 或命令输出；这些只记录文件路径、命令和关键结果。

## 存放方式

- PDF 放在 `EXPS/Chat/` 根目录。
- LaTeX 源文件放在 `EXPS/Chat/tex/`。
- 文件名使用三位编号加主题，例如 `002_raw_append_only_chat_log.pdf`。
- `001_step_ppo_chat_log` 是历史摘要式归档；从 `002_raw_append_only_chat_log` 开始使用 append-only 原文格式。

## 条目格式

每轮对话按时间顺序追加：

```text
[Turn N | User]
<用户原文>

[Turn N | Codex]
<Codex 原文或最终回复原文>
```

如果回复中包含公式，可以写成：

```tex
\[
A_t^{final}=\frac{A_t^{epi}+wA_t^{step}}{1+w}
\]
```

## 默认动作

后续每次关于 StepPPO/GiGPO/GRPO 的重要讨论结束后，都应 append 到当前 Chat TeX，并重新编译对应 PDF。若当前文件过长，则新建下一个编号文件继续 append-only 记录。
