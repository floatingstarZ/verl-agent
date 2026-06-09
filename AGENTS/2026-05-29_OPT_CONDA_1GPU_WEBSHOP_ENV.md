# 2026-05-29 /opt/conda 环境与 1GPU WebShop Smoke Run 记录

## 1. 结论

当前 `verl-agent` 统一使用 `/opt/conda` 环境运行，不再依赖其他 Conda 安装路径。

已完成一次 1GPU WebShop GiGPO smoke run 验证：

- 启动脚本：`exps/run_webshop_1gpu_paper_align_simple.sh`
- Python：`/opt/conda/bin/python`
- GPU：`NVIDIA A100-SXM4-80GB`
- 运行状态：已进入训练阶段，并连续完成 `step:1` 到 `step:7`
- 中止原因：人工停止，避免继续占用 GPU；停止前未见 `Traceback`、CUDA OOM 或 vLLM runtime error
- 最新日志：`logs/webshop_gigpo_qwen25_15b_1gpu_paper_align_simple_20260529_222730.log`

这说明当前 `/opt/conda` 环境下，WebShop 环境、Ray、vLLM、FSDP actor、rollout、reward、old logprob、ref logprob、advantage 和 actor update 训练链路均可跑通。

## 2. 项目路径

- 主项目：`/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent`
- 参考环境项目：`/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl_mcn/code_agent`
- WebShop 1GPU 脚本：`/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/exps/run_webshop_1gpu_paper_align_simple.sh`
- 4GPU 已跑通过的脚本：`/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/exps/run_webshop_4gpu_paper_align.sh`

注意：`run_webshop_4gpu_paper_align.sh` 是已跑过的基准脚本，不应为了环境路径清理而改动其历史成功配置。

## 3. 环境基线

用以下命令确认当前环境：

```bash
/opt/conda/bin/python - <<'PY'
import sys
import torch
mods = ['ray', 'vllm', 'transformers', 'flash_attn', 'wandb']
print('python', sys.version.split()[0])
print('executable', sys.executable)
print('torch', torch.__version__)
print('torch_cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu_count', torch.cuda.device_count())
    print('gpu0', torch.cuda.get_device_name(0))
for name in mods:
    mod = __import__(name)
    print(name, getattr(mod, '__version__', 'unknown'))
PY
```

本次实际输出：

```text
python 3.11.11
executable /opt/conda/bin/python
torch 2.9.1+cu128
torch_cuda 12.8
cuda_available True
gpu_count 1
gpu0 NVIDIA A100-SXM4-80GB
ray 2.53.0
vllm 0.15.0
transformers 4.57.1
flash_attn 2.8.3
wandb 0.25.1
```

## 4. 必须统一使用的环境入口

推荐所有脚本直接使用绝对路径：

```bash
export PATH="/opt/conda/bin:$PATH"
export PYTHONNOUSERSITE=1
/opt/conda/bin/python -m verl.trainer.main_ppo ...
```

运行前建议确认：

```bash
which python
python -c 'import sys; print(sys.executable)'
/opt/conda/bin/python -c 'import sys; print(sys.executable)'
```

预期关键点：

- `sys.executable` 应为 `/opt/conda/bin/python`
- 避免用户 site-packages 污染，保留 `PYTHONNOUSERSITE=1`
- 不要在同一次实验中混用多个 Conda 根目录或多个 Python 解释器

## 5. 启动前清理和资源设置

每次启动前建议先停止旧 Ray：

```bash
/opt/conda/bin/ray stop --force || true
```

脚本中保留的资源与稳定性设置：

```bash
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
```

这些设置的目的：

- 限制 BLAS/OpenMP/tokenizer 等库的线程数，避免 WebShop/Ray worker 数量较多时线程爆炸
- `WANDB_MODE=offline` 避免训练启动卡在外部登录或同步
- `HYDRA_FULL_ERROR=1` 便于定位 Hydra 配置错误
- `MALLOC_CONF` 降低长时间 Ray worker 运行中的内存碎片影响

## 6. 1GPU Smoke 脚本说明

脚本位置：

```bash
exps/run_webshop_1gpu_paper_align_simple.sh
```

运行方式：

```bash
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent
bash exps/run_webshop_1gpu_paper_align_simple.sh
```

脚本设计目标：

- 参考 `exps/run_webshop_4gpu_paper_align.sh` 的核心配置
- 用简单 wrapper 复用 `examples/gigpo_trainer/run_webshop.sh`
- 默认使用 `/opt/conda/bin/python`
- 单卡只做 smoke run，不追求论文规模吞吐
- 先验证训练链路能跑通，再逐步放大 batch、rollout 和验证规模

当前 1GPU 覆盖项：

```bash
data.train_batch_size=2
data.val_batch_size=4
actor_rollout_ref.actor.ppo_mini_batch_size=2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
env.rollout.n=2
trainer.n_gpus_per_node=1
trainer.nnodes=1
actor_rollout_ref.rollout.tensor_model_parallel_size=1
trainer.val_before_train=False
trainer.total_epochs=1
trainer.test_freq=-1
ray_init.num_cpus=16
env.resources_per_worker.num_cpus=0.1
```

关键原因：

- 4GPU 默认 `train_data_size=16`、`val_data_size=128`、`group_size=8` 会在 1GPU 上拉起大量 WebShop worker，启动和验证都很慢
- `val_before_train=False` 用于先验证训练链路，不让完整 validation 阶段消耗大量时间
- `trainer.total_epochs=1` 将 smoke run 缩短到 8 个训练 step
- `test_freq=-1` 关闭中途验证，避免 smoke run 主要时间被评估占用

## 7. awk 和 sed 在脚本中的作用

当前脚本用 `awk` 和 `sed` 动态生成一个临时版 `run_webshop.sh`，不直接改原始训练脚本：

```bash
bash <(
  awk 'NR==3{print "shift || true"} {print}' examples/gigpo_trainer/run_webshop.sh \
  | sed \
      -e '/examples\.data_preprocess\.prepare/,+3d' \
      -e 's|^python3 -m |/opt/conda/bin/python -m |' \
      -e 's/export VLLM_ATTENTION_BACKEND=XFORMERS/export VLLM_ATTENTION_BACKEND=FLASH_ATTN/' \
      -e 's/actor_rollout_ref.rollout.enforce_eager=False/actor_rollout_ref.rollout.enforce_eager=True/'
) vllm ...
```

具体作用：

- `awk 'NR==3{print "shift || true"} {print}'`：在原脚本第 3 行插入 `shift || true`，让第一个位置参数 `vllm` 被当作 `ENGINE` 消费后，后续 Hydra overrides 不会被吞掉
- `sed '/examples\.data_preprocess\.prepare/,+3d'`：删除当前环境不存在的 `examples.data_preprocess.prepare` 数据预处理调用，避免启动时报 `ModuleNotFoundError`
- `sed 's|^python3 -m |/opt/conda/bin/python -m |'`：强制 trainer 使用 `/opt/conda/bin/python`
- `sed 's/export VLLM_ATTENTION_BACKEND=XFORMERS/export VLLM_ATTENTION_BACKEND=FLASH_ATTN/'`：切换到当前 A100/H100 环境验证过的 attention backend
- `sed 's/actor_rollout_ref.rollout.enforce_eager=False/actor_rollout_ref.rollout.enforce_eager=True/'`：规避 vLLM graph/编译路径上的稳定性问题，优先保证 smoke run 可跑通

## 8. vLLM 0.15 兼容修复

当前 `/opt/conda` 中的 vLLM 是 `0.15.0`。该版本没有旧路径：

```text
vllm.lora.models
```

实际可用路径是：

```text
vllm.lora.lora_model
```

因此 `verl/utils/vllm_utils.py` 中已做兼容导入：

```python
try:
    from vllm.lora.models import LoRAModel
except ImportError:
    from vllm.lora.lora_model import LoRAModel
```

验证命令：

```bash
/opt/conda/bin/python - <<'PY'
from verl.utils.vllm_utils import LoRAModel
print(LoRAModel)
PY
```

预期可正常打印 `vllm.lora.lora_model.LoRAModel`。

## 9. 本次 1GPU 运行观察

运行命令：

```bash
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent
bash exps/run_webshop_1gpu_paper_align_simple.sh
```

最新日志：

```text
logs/webshop_gigpo_qwen25_15b_1gpu_paper_align_simple_20260529_222730.log
```

训练启动后关键日志：

```text
Total steps: 8, num_warmup_steps: 0
Training Progress: 0/8
step:1 ... training/global_step:1.000 ... timing_s/step:90.196 ... perf/max_memory_allocated_gb:73.689 ... perf/max_memory_reserved_gb:89.240
step:2 ... training/global_step:2.000 ... timing_s/step:89.430 ... perf/max_memory_allocated_gb:73.690 ... perf/max_memory_reserved_gb:89.240
step:3 ... training/global_step:3.000 ... timing_s/step:82.841 ... perf/max_memory_allocated_gb:73.690 ... perf/max_memory_reserved_gb:89.240
step:4 ... training/global_step:4.000 ... timing_s/step:77.940 ... perf/max_memory_allocated_gb:73.690 ... perf/max_memory_reserved_gb:89.240
step:5 ... training/global_step:5.000 ... timing_s/step:80.580 ... perf/max_memory_allocated_gb:73.690 ... perf/max_memory_reserved_gb:89.240
step:6 ... training/global_step:6.000 ... timing_s/step:95.595 ... perf/max_memory_allocated_gb:73.690 ... perf/max_memory_reserved_gb:89.240
step:7 ... training/global_step:7.000 ... timing_s/step:86.135 ... perf/max_memory_allocated_gb:73.690 ... perf/max_memory_reserved_gb:89.240
```

说明：

- `perf/max_memory_*` 是训练日志报告值，用于判断单卡配置是否逼近上限
- 当前 1GPU smoke 配置显存压力仍然很高，不建议直接在 40GB 卡上使用同一配置
- 1GPU smoke run 每步约 78 到 96 秒，主要时间在 rollout generation
- `episode/success_rate` 在 smoke run 初期为 0 是正常现象，这次验证目标是链路可运行，不是效果收敛

## 10. 常见问题和处理

### 10.1 `ModuleNotFoundError: No module named 'examples.data_preprocess'`

原因：当前项目中没有该模块，但 `examples/gigpo_trainer/run_webshop.sh` 会调用它。

处理：1GPU wrapper 用 `sed` 删除这段调用；不要为 smoke run 新造空模块掩盖问题。

### 10.2 `ModuleNotFoundError: No module named 'vllm.lora.models'`

原因：vLLM 0.15 中 LoRA model 路径变化。

处理：保留 `verl/utils/vllm_utils.py` 中的 fallback import。

### 10.3 1GPU 启动后大量 WebShop worker，长时间无训练 step

原因：继承 4GPU 配置时，`train_batch_size=16`、`env.rollout.n=8`、`val_batch_size=128` 会导致 worker 数量和验证任务过大。

处理：用当前 1GPU smoke 覆盖项缩小规模，并设置：

```bash
trainer.val_before_train=False
trainer.total_epochs=1
trainer.test_freq=-1
```

### 10.4 Ray 资源残留或启动异常

处理：每次重跑前执行：

```bash
/opt/conda/bin/ray stop --force || true
```

必要时检查：

```bash
ps -eo pid,ppid,stat,pcpu,pmem,cmd | rg 'ray|TaskRunner|WorkerDict|WebshopWorker'
```

## 11. 放大实验建议

当前 `run_webshop_1gpu_paper_align_simple.sh` 是 smoke 脚本，不是最终效果实验脚本。

建议放大顺序：

1. 保持 `trainer.val_before_train=False`，先把 `trainer.total_epochs` 从 `1` 增大到更长训练。
2. 如果显存稳定，再尝试增加 `data.train_batch_size` 或 `env.rollout.n`，不要同时增加。
3. 需要评估时再恢复 `trainer.test_freq`，并谨慎设置 `data.val_batch_size`。
4. 如果要贴近论文配置，优先使用已跑通过的 4GPU paper-align 脚本，而不是强行在 1GPU 上恢复完整 batch 和 validation。

## 12. 当前可复现最小命令

```bash
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent
/opt/conda/bin/ray stop --force || true
bash exps/run_webshop_1gpu_paper_align_simple.sh
```

观察训练 step：

```bash
latest=$(ls -1t logs/webshop_gigpo_qwen25_15b_1gpu_paper_align_simple_*.log | head -1)
rg -n 'Total steps:|Training Progress:|step:[0-9]+|Traceback|Error executing job|CUDA out of memory|ModuleNotFoundError' "$latest"
```

停止实验：

```bash
pkill -f 'run_webshop_1gpu_paper_align_simple.sh' || true
/opt/conda/bin/ray stop --force || true
```
