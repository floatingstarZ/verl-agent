# 2026-05-28 A100/H100 Flash-Attn Compatibility Note

## 结论（短答）

- 目前 `verl-agent` 环境中的 `flash-attn==2.8.3` 已重新源码编译，编译参数包含 `sm_80` 和 `sm_90`。
- 这意味着从 **二进制目标架构** 上，已同时覆盖 A100（sm_80）和 H100（sm_90）。
- **A100 已实机验证通过**（`flash_attn_func` 最小 CUDA 调用成功）。
- **H100 在当前会话未实机复测**，但从编译产物与历史文档看，按同样环境应可运行。

## 本次已执行的修复与验证

1. 在 `conda env: verl-agent` 中补齐编译工具链：
   - `cuda-nvcc=12.8`（conda 安装）
2. 重装 `flash-attn`：
   - `flash-attn==2.8.3`
   - `FLASH_ATTN_CUDA_ARCHS='80;90'`
   - `CUDA_HOME=$CONDA_PREFIX`
3. 验收结果：
   - `import torch, flash_attn, flash_attn_2_cuda` 成功
   - `flash_attn_func` 在 A100 上成功，输出：
     - `torch.Size([2, 128, 8, 64]) torch.bfloat16 cuda:0`

## 仍需注意（影响“跨机都顺利运行”）

1. 启动脚本参数仍建议与 `AGENTS` 文档保持一致，尤其：
   - `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
   - `actor_rollout_ref.rollout.enforce_eager=True`
   - 线程/JVM 限制（避免 WebShop worker 线程爆炸）
2. 训练脚本 `examples/gigpo_trainer/run_webshop.sh` 当前仍有与建议不一致项，建议后续同步。
3. 跨机器运行前，务必在目标机执行一次最小算子验收（见下）。

## 目标机器最小验收（A100/H100 通用）

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate verl-agent

python - <<'PY'
import torch
from flash_attn import flash_attn_func
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0))
q = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
print('ok', out.shape, out.dtype, out.device)
PY
```

期望看到：
- 无 `no kernel image is available for execution on the device`
- 输出包含 `ok ... cuda:0`

## 若再次出现报错

- 典型错误：
  - `CUDA error: no kernel image is available for execution on the device`
  - `undefined symbol ... flash_attn_2_cuda`
- 优先检查：
  1. 是否在正确环境（`conda activate verl-agent`）
  2. `torch` CUDA 版本与编译工具链是否一致（建议 cu128 + nvcc 12.8）
  3. `flash-attn` 是否被其他路径旧包污染（`~/.local`）
  4. 重装时是否包含 `FLASH_ATTN_CUDA_ARCHS='80;90'`
