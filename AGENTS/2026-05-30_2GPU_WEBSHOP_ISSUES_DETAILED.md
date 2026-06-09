# 2026-05-30 2GPU WebShop 问题详细记录（H100）

## 1. 背景与目标

- 目标：基于 `exps/run_webshop_1gpu_paper_align_simple.sh` 产出并验证 2GPU 版本脚本，检查环境可用性。
- 机器 GPU：`NVIDIA H100 80GB HBM3`（sm_90）。
- 主运行脚本：`exps/run_webshop_2gpu_paper_align_simple.sh`。

---

## 2. 关键改动（本次会话内）

### 2.1 新增脚本

- `exps/run_webshop_2gpu_paper_align_simple.sh`

主要内容：
- 固定 `/opt/conda/bin/python`。
- 2GPU 参数：
  - `trainer.n_gpus_per_node=2`
  - `actor_rollout_ref.rollout.tensor_model_parallel_size=2`
  - smoke 配置（`train_batch_size=4`、`env.rollout.n=4`、`total_epochs=1`、`test_freq=-1`）。
- 保留 `FLASH_ATTN + enforce_eager`。

### 2.2 修改训练入口

- `examples/gigpo_trainer/run_webshop.sh`

改动点：
- 缓存目录统一重定向到 `/tmp/$USER`：
  - `XDG_CACHE_HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`
  - `TRITON_CACHE_DIR`, `TORCH_HOME`, `VLLM_CONFIG_ROOT`, `FLASHINFER_WORKSPACE_BASE`
- 数据根目录改为可写默认：
  - `VERL_AGENT_DATA_ROOT` 默认 `/tmp/$USER/data/verl-agent`
- `prepare` 改为：
  - `PYTHONPATH="$(pwd):${PYTHONPATH:-}" "$PYTHON_BIN" examples/data_preprocess/prepare.py ...`
- trainer 改为统一 `"$PYTHON_BIN" -m verl.trainer.main_ppo ...`

### 2.3 vLLM 兼容补丁

- `verl/workers/sharding_manager/fsdp_vllm.py`

改动点：
- 兼容 `vllm.distributed.parallel_state` 的 API 差异：
  - 旧：`get_tensor_model_parallel_group()`
  - 新（vLLM 0.15）：`get_tp_group()`

---

## 3. 问题清单（按发现顺序）

## 3.1 Triton 缓存权限错误

- 现象：
  - `PermissionError: ... '/usr2/ziyuhuan/.triton'`
- 根因：
  - HOME 路径不可写，Triton 默认写 `~/.triton`。
- 处理：
  - 设置 `TRITON_CACHE_DIR=/tmp/$USER/.triton-cache`。
- 结果：
  - 该错误消失。

## 3.2 HuggingFace 缓存权限错误

- 现象：
  - `PermissionError: ... '/usr2/ziyuhuan/.cache'`
- 根因：
  - HF/transformers/flashinfer 默认使用 HOME 下 cache。
- 处理：
  - 统一设置到 `/tmp/$USER/.cache/...`。
- 结果：
  - HF 下载与 tokenizer 初始化权限错误消失。

## 3.3 训练数据文件不存在

- 现象：
  - `FileNotFoundError: Unable to find '/usr2/ziyuhuan/data/verl-agent/text/train.parquet'`
- 根因：
  - 原脚本默认数据路径在 HOME，不存在。
- 处理：
  - 训练入口改为先执行 `prepare.py`，并将输出写到 `/tmp/$USER/data/verl-agent/text/*.parquet`。
- 结果：
  - 数据缺失问题解决，训练可读到 parquet。

## 3.4 `examples.data_preprocess.prepare` 导入失败

- 现象：
  - `ModuleNotFoundError: No module named 'examples.data_preprocess'`
- 根因：
  - 当前环境下 `python -m examples.data_preprocess.prepare` 不可导入。
- 处理：
  - 改为脚本路径执行 + `PYTHONPATH=$(pwd)`。
- 结果：
  - `prepare.py` 可成功执行并生成数据。

## 3.5 vLLM TP group API 变化

- 现象：
  - `AttributeError: module 'vllm.distributed.parallel_state' has no attribute 'get_tensor_model_parallel_group'`
- 根因：
  - vLLM 0.15 API 变更。
- 处理：
  - `fsdp_vllm.py` 增加 `get_tp_group()` 兼容分支。
- 结果：
  - 该兼容错误消失，能进入后续训练流程。

## 3.6 无效 Hydra override（会话中临时出现）

- 现象：
  - `ConfigCompositionException: Could not override 'actor_rollout_ref.model.attn_implementation'`
- 根因：
  - 脚本中出现了当前配置不存在的 key。
- 处理：
  - 从 `exps/run_webshop_2gpu_paper_align_simple.sh` 移除该 override。
- 结果：
  - Hydra 配置错误消失，训练继续推进。

## 3.7 当前最终阻塞：Flash-Attn 内核不支持 H100

- 现象：
  - 已进入训练（`Total steps: 4`, `Training Progress: 0/4`）后报错：
  - `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`
- 根因验证：
  - `flash_attn_2_cuda` 二进制仅含 `sm_80`，缺失 `sm_90`。
  - 当前机器为 H100（sm_90），因此运行时触发 `no kernel image`。
- 验证命令：
  ```bash
  so=$(/opt/conda/bin/python - <<'PY'
  import torch, flash_attn_2_cuda
  print(flash_attn_2_cuda.__file__)
  PY
  )
  cuobjdump --list-elf "$so" | rg 'sm_80|sm_90' | head -30
  ```
- 当前输出结论：
  - 仅 `sm_80`，无 `sm_90`。

---

## 4. 关键日志索引

- 最新完整复测日志（截至本记录）：
  - `logs/webshop_gigpo_qwen25_15b_2gpu_paper_align_simple_20260530_082921.log`
- 关键特征（该日志中可 grep）：
  - `processing data for mode: text`
  - `Total steps: 4`
  - `Training Progress:   0%|          | 0/4`
  - `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`

建议定位命令：
```bash
latest=$(ls -1t logs/webshop_gigpo_qwen25_15b_2gpu_paper_align_simple_*.log | head -1)
rg -n "processing data for mode|Total steps|Training Progress|no kernel image|Traceback|Error executing job" "$latest"
```

---

## 5. 当前结论

1. 脚本链路、数据准备、Ray/vLLM/FSDP/rollout 主流程都已打通。  
2. 当前唯一核心阻塞是 H100 上 flash-attn 二进制架构不匹配（缺 `sm_90`）。  
3. 在修复该环境问题前，2GPU 训练会稳定复现同一错误点。  

---

## 6. 2026-05-30 复查处理结果

### 6.1 已确认仍有效的问题修复

以下问题在当前代码中仍保持已解决状态：

- 3.1 Triton cache 权限：`TRITON_CACHE_DIR` 已指向 `/tmp/$USER/.triton-cache`。
- 3.2 HF/cache 权限：`XDG_CACHE_HOME`、`HF_HOME`、`HUGGINGFACE_HUB_CACHE`、`TRANSFORMERS_CACHE`、`TORCH_HOME`、`VLLM_CONFIG_ROOT` 已重定向。
- 3.3 数据文件不存在：`VERL_AGENT_DATA_ROOT` 默认 `/tmp/$USER/data/verl-agent`，并在入口执行 `prepare.py`。
- 3.4 `examples.data_preprocess.prepare` 导入失败：已改为脚本路径执行并设置 `PYTHONPATH=$(pwd)`。
- 3.5 vLLM TP group API：`fsdp_vllm.py` 已兼容 `get_tp_group()`。
- 3.6 无效 Hydra override：当前 `exps/run_webshop_2gpu_paper_align_simple.sh` 已确认不再传 `actor_rollout_ref.model.attn_implementation`。

### 6.2 H100 flash-attn sm_90 问题的当前处理

由于 `/opt/conda` 为 root-owned，当前用户无法把重编的 flash-attn 直接安装到 `/opt/conda/lib/python3.11/site-packages`。

已采用替代方案：将编译好的 `sm_80 + sm_90` flash-attn 包放在容器 overlay root 路径：

```text
/tmp/flash_attn_sm80_sm90_site
```

并在 `exps/run_webshop_2gpu_paper_align_simple.sh` 中：

- 保留 `PYTHONNOUSERSITE=1`，不依赖 user site。
- 启动前检查 `/tmp/flash_attn_sm80_sm90_site/flash_attn_2_cuda*.so` 存在。
- 启动前用 `cuobjdump + awk` 检查 `.so` 包含 `sm_90`。
- 将该目录 prepend 到 `PYTHONPATH`，覆盖 `/opt/conda` 中只有 `sm_80` 的旧 flash-attn。

验证命令：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/flash_attn_sm80_sm90_site /opt/conda/bin/python - <<'PY'
import site, flash_attn, flash_attn_2_cuda
print(site.ENABLE_USER_SITE)
print(flash_attn.__file__)
print(flash_attn_2_cuda.__file__)
PY

cuobjdump --list-elf /tmp/flash_attn_sm80_sm90_site/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so \
  | rg 'sm_80|sm_90' \
  | head -40
```

当前验证结果：

```text
False
/tmp/flash_attn_sm80_sm90_site/flash_attn/__init__.py
/tmp/flash_attn_sm80_sm90_site/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so
```

并且 `cuobjdump` 可看到 `sm_90.cubin`。

### 6.3 当前会话无法完整复测 2GPU H100

当前会话只看到 1 张 A100，因此 2GPU 脚本预检结果为预期失败：

```text
[FATAL] Need >=2 visible GPUs, got 1
```

该失败发生在 sm_90 flash-attn 覆盖包校验之后，说明覆盖包校验已通过。

在 2GPU H100 机器上需要重新执行：

```bash
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent
bash exps/run_webshop_2gpu_paper_align_simple.sh
```

预期：

- 不再出现 `ConfigCompositionException: Could not override 'actor_rollout_ref.model.attn_implementation'`。
- 不再加载 `/opt/conda` 中只有 `sm_80` 的 `flash_attn_2_cuda`。
- 不再在 H100 上触发 flash-attn `no kernel image is available for execution on the device`。
- 至少推进到 `step:1`。
