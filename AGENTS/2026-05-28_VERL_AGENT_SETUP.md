# 2026-05-28 VERL_AGENT_SETUP（持续更新）

## 1. 结论

已在 4 卡 H100 上稳定拉起 GiGPO(WebShop) 训练，且未再出现
`Paged KV cache block size must be divisible by 256`。

成功日志样例：

- `logs/webshop_gigpo_qwen25_15b_4gpu_flash_eager_20260528_120119.log`
- `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_20260528_151338.log`

## 1.1 关键版本（当前环境）

- Python: `3.11.11`
- PyTorch: `2.9.1+cu128`
- CUDA(torch): `12.8`
- flash-attn: `2.8.3`
- setuptools: `80.9.0`

## 2. 成功配置（与论文主干对齐，最小必要工程修复）

- 算法与核心超参保持 `examples/gigpo_trainer/run_webshop.sh` 默认设置：
  - `algorithm.adv_estimator=gigpo`
  - `train_batch_size=16`
  - `val_batch_size=128`
  - `group_size=8`
  - `test_freq=5`
  - `total_epochs=150`
- 4 卡训练参数：
  - `trainer.n_gpus_per_node=4`
  - `trainer.nnodes=1`
- 稳定性修复项（不改算法逻辑）：
  - `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
  - `actor_rollout_ref.rollout.enforce_eager=True`
  - `ray_init.num_cpus=32`
  - `env.resources_per_worker.num_cpus=0.1`
  - 线程限制：`OMP/MKL/OPENBLAS/NUMEXPR/VECLIB/BLIS/RAYON=1`
  - `MALLOC_CONF=background_thread:false,narenas:1,dirty_decay_ms:0,muzzy_decay_ms:0`
  - `TOKENIZERS_PARALLELISM=false`
  - `WANDB_MODE=offline`
- 工程修复：
  - 启动 wrapper 中加入 `shift || true`，避免 `ENGINE` 参数吞掉后续 hydra overrides。
  - `examples/gigpo_trainer/run_webshop.sh` 中不再硬编码 `ulimit -u 65536`，改为尽量使用系统 hard limit，并加入 `JAVA_TOOL_OPTIONS` 控制 JVM 线程规模，避免 WebShop worker 触发 `Cannot create worker GC thread`。

## 3. 成功启动命令

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate verl-agent
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent

mkdir -p logs /tmp/verl-agent-run

awk 'NR==3{print "shift || true"} {print}' examples/gigpo_trainer/run_webshop.sh \
  | sed \
      -e 's/export VLLM_ATTENTION_BACKEND=XFORMERS/export VLLM_ATTENTION_BACKEND=FLASH_ATTN/' \
      -e 's/actor_rollout_ref.rollout.enforce_eager=False/actor_rollout_ref.rollout.enforce_eager=True/' \
  > /tmp/verl-agent-run/run_webshop_4gpu_flash_eager.sh
chmod +x /tmp/verl-agent-run/run_webshop_4gpu_flash_eager.sh

ulimit -n 65536 || true
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_CONF=background_thread:false,narenas:1,dirty_decay_ms:0,muzzy_decay_ms:0

bash /tmp/verl-agent-run/run_webshop_4gpu_flash_eager.sh vllm \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.experiment_name=gigpo_qwen2.5_1.5b_4gpu_flash_eager \
  ray_init.num_cpus=32 \
  env.resources_per_worker.num_cpus=0.1 \
  2>&1 | tee logs/webshop_gigpo_qwen25_15b_4gpu_flash_eager_$(date +%Y%m%d_%H%M%S).log
```

## 4. 日志位置

- 训练日志：`logs/webshop_gigpo_qwen25_15b_4gpu_flash_eager_*.log`
- 训练日志（论文对齐新命名）：`logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_*.log`

## 4.1 2026-05-28 新增验证（2GPU 稳定性烟测）

- 验证任务：`trainer.n_gpus_per_node=2`，其余算法超参与默认保持一致。
- 验证结果：
  - 可稳定进入训练阶段并产出 `step:1` 指标。
  - 未再出现 JVM `Cannot create worker GC thread` 崩溃。
- 额外稳定性结论：
  - `ray_init.num_cpus=256` 在本机可正常运行（资源充足时可提高调度余量）。
  - 当前环境下直接 `nohup ... > log 2>&1 &` 可能出现“进程退出且日志空文件”的假象；
    建议使用前台会话 + `tee` 持续落盘，或先前台确认进入 `main_ppo` 后再后台托管。

## 4.2 运行注意事项（保留成功经验）

- 训练前建议执行：
  - `ray stop --force || true`
  - 检查 GPU 空闲后再启动，避免旧会话资源残留影响新任务。
- 启动脚本建议保留以下工程修复：
  - `shift || true`（避免 `ENGINE` 吞参数）
  - `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
  - `actor_rollout_ref.rollout.enforce_eager=True`
  - 线程限制与 `MALLOC_CONF` 设置
  - 使用 `tee` 实时落盘日志（避免个别环境下 `nohup > file` 空日志误判）

## 5. Flash-Attention 专项修复（H100）

### 5.1 根因

- 报错：`torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`
- 实际原因：环境中旧版 `flash_attn_2_cuda` 仅包含 `sm_80` cubin，在 H100(`sm_90`) 上训练侧 flash-attn 内核不可执行。

### 5.2 修复动作

在 `verl-agent` conda 环境中重装源码编译版 flash-attn（显式编译 `sm_90`）：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate verl-agent

python -m pip install --no-cache-dir -U pip wheel ninja
python -m pip uninstall -y flash-attn flash_attn

export MAX_JOBS=16
export CUDA_HOME=/usr/local/cuda
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTN_CUDA_ARCHS=90

python -m pip install --no-cache-dir --no-build-isolation -v flash-attn==2.8.3
```

说明：
- `setuptools==79.0.1` 是历史阶段为兼容构建链的临时措施；当前环境已可在 `setuptools 80.9.0` 下工作，无需强制回退。

### 5.3 验收

1) 版本和导入：

```bash
python - <<'PY'
import torch, flash_attn, flash_attn_2_cuda
print(torch.__version__, torch.version.cuda)
print(flash_attn.__version__)
print(flash_attn_2_cuda.__file__)
PY
```

2) 二进制架构检查（必须看到 `sm_90`）：

```bash
so=/opt/conda/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so
cuobjdump --list-elf "$so" | rg 'sm_90' | head
```

注：当前环境 Python 为 3.11，实际 `.so` 路径应位于 `.../python3.11/site-packages/`。

3) 最小 CUDA 调用测试：

```bash
python - <<'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
print(out.shape, out.dtype, out.device)
PY
```
