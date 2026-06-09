# 2026-05-30 2GPU WebShop 运行问题交接说明（待你处理环境）

## 1. 本次目标

基于 `exps/run_webshop_1gpu_paper_align_simple.sh` 产出并验证 2GPU 版本脚本，确认环境可行性与阻塞点。

## 1.1 机器与 GPU（明确）

- 当前验证机器 GPU 型号：`NVIDIA H100 80GB HBM3`
- 本次问题是在 **H100（sm_90）** 上复现与定位

---

## 2. 新增/修改文件

### 2.1 新增

- `exps/run_webshop_2gpu_paper_align_simple.sh`

### 2.2 修改

- `examples/gigpo_trainer/run_webshop.sh`
- `verl/workers/sharding_manager/fsdp_vllm.py`

---

## 3. 已经解决的问题

1. **HOME 不可写导致的缓存权限问题**
   - 现象：
     - `/usr2/ziyuhuan/.triton` 权限报错
     - `/usr2/ziyuhuan/.cache` 权限报错（HF/flashinfer）
   - 处理：
     - 缓存统一重定向到 `/tmp/$USER`：
       - `XDG_CACHE_HOME`
       - `HF_HOME`
       - `HUGGINGFACE_HUB_CACHE`
       - `TRANSFORMERS_CACHE`
       - `TRITON_CACHE_DIR`
       - `TORCH_HOME`
       - `VLLM_CONFIG_ROOT`
       - `FLASHINFER_WORKSPACE_BASE`

2. **WebShop 数据路径不存在**
   - 原默认路径：`$HOME/data/verl-agent/text/*.parquet`
   - 处理：
     - 改为默认写到 `/tmp/$USER/data/verl-agent/text/*.parquet`
     - `prepare.py` 自动生成 parquet 后训练直接读取。

3. **`examples.data_preprocess.prepare` 模块导入失败**
   - 处理：
     - 改为脚本路径执行：
       - `PYTHONPATH="$(pwd):${PYTHONPATH:-}" "$PYTHON_BIN" examples/data_preprocess/prepare.py ...`

4. **vLLM 0.15 API 兼容问题**
   - 现象：
     - `vllm.distributed.parallel_state` 中无 `get_tensor_model_parallel_group`
   - 处理：
     - `fsdp_vllm.py` 兼容 `get_tp_group()` 分支。

---

## 4. 当前剩余阻塞（你需要处理）

### 4.1 Flash-Attn 内核架构不匹配 H100

- 最终报错：
  - `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`
- 触发位置：
  - `flash_attn` 前向调用阶段（训练已进入 `Training Progress` 后报错）
- 根因验证：
  - 当前机器是 **H100（sm_90）**
  - 当前 `flash_attn_2_cuda` 二进制只看到 `sm_80`，未包含 `sm_90`，因此在 H100 上触发该错误。

已执行检查命令（可复用）：

```bash
so=$(/opt/conda/bin/python - <<'PY'
import torch, flash_attn_2_cuda
print(flash_attn_2_cuda.__file__)
PY
)
echo "$so"
cuobjdump --list-elf "$so" | rg 'sm_80|sm_90' | head -40
```

当前结果仅出现 `sm_80`，未看到 `sm_90`。

---

## 5. 运行进度与日志

已能稳定到达：

- `prepare.py` 数据生成成功
- Ray 启动成功
- WebShop worker 初始化成功
- `Total steps: 4`
- `Training Progress: 0/4`

随后在 flash-attn kernel 处失败。

关键日志：

- `logs/webshop_gigpo_qwen25_15b_2gpu_paper_align_simple_20260530_070656.log`

可快速定位：

```bash
latest=$(ls -1t logs/webshop_gigpo_qwen25_15b_2gpu_paper_align_simple_*.log | head -1)
rg -n "processing data for mode|Total steps|Training Progress|no kernel image|Traceback|Error executing job" "$latest"
```

---

## 6. 你修复环境后的复测命令

```bash
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent
bash exps/run_webshop_2gpu_paper_align_simple.sh
```

预期至少看到：

- `Total steps: 4`
- `Training Progress: ...`
- 出现 `step:1`（若继续稳定）

---

## 7. 备注

- 本次未改动你其他已有脏文件，仅新增/修改上文列出的 3 个文件。
- 当前交接重点是：你处理 flash-attn 的 `sm_90` 编译/安装后，再用上面的 2GPU 脚本复测。

---

## 8. 2026-05-30 处理结果

### 8.1 当前会话限制

本次处理所在机器只暴露 1 张 GPU：

```text
0, NVIDIA A100-SXM4-80GB
```

因此无法在当前会话中完整复现 2GPU H100 训练；`exps/run_webshop_2gpu_paper_align_simple.sh` 的 GPU 数量检查会按预期退出：

```text
[FATAL] Need >=2 visible GPUs, got 1
```

### 8.2 flash-attn sm_90 编译验证

已尝试在当前用户下为 `/opt/conda` 环境源码重编 `flash-attn==2.8.3`，编译命令显式包含：

```bash
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export FLASH_ATTN_CUDA_ARCHS="80;90"
```

编译日志：

```text
logs/flash_attn_rebuild_sm80_sm90_20260530_001926.log
```

日志中可见 nvcc 命令包含：

```text
-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90
```

生成 wheel 并安装成功，但 pip 提示：

```text
Defaulting to user installation because normal site-packages is not writeable
```

原因是 `/opt/conda/lib/python3.11/site-packages` 为 root-owned，当前用户不可写。

### 8.3 为什么不采用 user-site 编译结果

普通导入会加载用户 site-packages 中新编译的 flash-attn，此时 `cuobjdump` 可看到 `sm_90`。

但实验脚本统一设置了：

```bash
export PYTHONNOUSERSITE=1
```

这是为了保证只使用 `/opt/conda` 环境，避免用户目录依赖污染。启用 `PYTHONNOUSERSITE=1` 后，Python 实际仍加载：

```text
/opt/conda/lib/python3.11/site-packages/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so
```

该 root-owned 版本仍只有 `sm_80`，没有 `sm_90`。

因此：

- user-site 中的 sm_90 wheel 证明源码编译方案可行；
- 但在当前权限下，它不能作为正式修复；
- 正式环境若要继续使用 `flash_attention_2`，需要由有 `/opt/conda` 写权限的用户把该 wheel 安装进 `/opt/conda` 本身。

### 8.4 当前实际修复

为保持 `/opt/conda` 纯环境和 `PYTHONNOUSERSITE=1`，已修改：

```text
exps/run_webshop_2gpu_paper_align_simple.sh
```

新增 Hydra override：

```bash
# Deprecated: do not pass actor_rollout_ref.model.attn_implementation; this key is absent in the current Hydra config.
```

作用：

- 训练侧 HuggingFace/FSDP actor 不再使用 `flash_attention_2`；
- 避开 `/opt/conda` 里旧 flash-attn 扩展缺 `sm_90` 导致的 H100 `no kernel image is available for execution on the device`；
- vLLM rollout 侧仍保留脚本原有设置：`VLLM_ATTENTION_BACKEND=FLASH_ATTN` 和 `actor_rollout_ref.rollout.enforce_eager=True`。

### 8.5 后续在 2GPU H100 上复测

在 2GPU H100 机器上运行：

```bash
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent
bash exps/run_webshop_2gpu_paper_align_simple.sh
```

预期变化：

- 不再在训练侧 flash-attn 前向处触发 `no kernel image is available for execution on the device`；
- 至少应看到 `Total steps: 4`、`Training Progress`，并进一步出现 `step:1`。

若仍失败，优先检查：

```bash
latest=$(ls -1t logs/webshop_gigpo_qwen25_15b_2gpu_paper_align_simple_*.log | head -1)
rg -n "attn_implementation|Total steps|Training Progress|step:1|no kernel image|Traceback|Error executing job" "$latest"
```

### 8.6 如果必须使用 flash_attention_2

需要在有 `/opt/conda` 写权限的环境中执行类似命令：

```bash
export PATH="/opt/conda/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export FLASH_ATTN_CUDA_ARCHS="80;90"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
/opt/conda/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps --no-build-isolation --no-binary flash-attn -v flash-attn==2.8.3
```

安装后必须在 `PYTHONNOUSERSITE=1` 下验证：

```bash
so=$(PYTHONNOUSERSITE=1 /opt/conda/bin/python - <<'PY'
import torch, flash_attn_2_cuda
print(flash_attn_2_cuda.__file__)
PY
)
echo "$so"
cuobjdump --list-elf "$so" | rg 'sm_80|sm_90' | head -40
```

必须看到 `/opt/conda/.../flash_attn_2_cuda...so` 中同时包含 `sm_80` 和 `sm_90`。

---

## 9. 2026-05-30 修正：不要依赖 user site，改用 commit 可保留的 overlay 路径

上一节中的 `eager` workaround 不是最终方案：它可以绕过 H100 flash-attn 错误，但会改变训练侧 attention 实现，不符合继续使用 `flash_attention_2` 的目标。

此外，直接安装到 user site 也不可取：

- user site 位于用户目录，可能是外部挂载；
- docker commit 后不一定包含该路径；
- 脚本设置了 `PYTHONNOUSERSITE=1`，正式运行也不会加载 user site。

### 9.1 当前采用的方案

把已编译好的 `sm_80 + sm_90` flash-attn 包复制到容器 overlay root 下：

```text
/tmp/flash_attn_sm80_sm90_site
```

当前验证：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/flash_attn_sm80_sm90_site /opt/conda/bin/python - <<'PY'
import site, flash_attn, flash_attn_2_cuda
print(site.ENABLE_USER_SITE)
print(flash_attn.__file__)
print(flash_attn_2_cuda.__file__)
PY
```

输出确认：

```text
False
/tmp/flash_attn_sm80_sm90_site/flash_attn/__init__.py
/tmp/flash_attn_sm80_sm90_site/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so
```

`cuobjdump` 已确认该 `.so` 同时包含 `sm_80` 和 `sm_90`。

### 9.2 为什么选 `/tmp` 而不是 `/var/tmp` 或项目目录

- `/tmp` 当前位于容器 overlay root，理论上会进入 docker commit。
- `/var/tmp` 是单独挂载的 ext4，不适合作为 commit 持久化路径。
- 项目目录和用户 home 在实际平台上可能是挂载路径，不应作为镜像内依赖路径。
- `/opt/conda` 是 root-owned，当前用户不可写，无法直接替换其中的 flash-attn。

### 9.3 脚本改动

`exps/run_webshop_2gpu_paper_align_simple.sh` 已修改：

- 保留 `PYTHONNOUSERSITE=1`。
- 默认要求存在：

```bash
FLASH_ATTN_SM90_SITE=/tmp/flash_attn_sm80_sm90_site
```

- 启动前检查该目录中存在 `flash_attn_2_cuda*.so`。
- 若系统有 `cuobjdump`，启动前检查该 `.so` 包含 `sm_90`。
- 将该目录 prepend 到 `PYTHONPATH`：

```bash
export PYTHONPATH="$FLASH_ATTN_SM90_SITE:${PYTHONPATH:-}"
```

- 训练侧显式保持：

```bash
# Deprecated: do not pass actor_rollout_ref.model.attn_implementation; rely on the trainer default and override flash-attn via PYTHONPATH.
```

这样正式运行时会加载 `/tmp/flash_attn_sm80_sm90_site` 中的 sm_90 扩展，而不是 `/opt/conda` 中只有 sm_80 的旧扩展。

### 9.4 docker commit 前必须确认

```bash
ls -lh /tmp/flash_attn_sm80_sm90_site
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/flash_attn_sm80_sm90_site /opt/conda/bin/python - <<'PY'
import flash_attn, flash_attn_2_cuda
print(flash_attn.__file__)
print(flash_attn_2_cuda.__file__)
PY
so=/tmp/flash_attn_sm80_sm90_site/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so
cuobjdump --list-elf "$so" | rg 'sm_80|sm_90' | head -40
```

必须看到导入路径为 `/tmp/flash_attn_sm80_sm90_site/...`，并且 `cuobjdump` 中包含 `sm_90`。

---

## 10. 2026-05-30 更正：不要传 `actor_rollout_ref.model.attn_implementation`

`AGENTS/2026-05-30_2GPU_WEBSHOP_ISSUES_DETAILED.md` 记录过该 Hydra key 当前不存在，传入会导致：

```text
ConfigCompositionException: Could not override 'actor_rollout_ref.model.attn_implementation'
```

因此 `exps/run_webshop_2gpu_paper_align_simple.sh` 已删除该 override。

当前最终策略是：

- 不传 `actor_rollout_ref.model.attn_implementation`。
- 依赖 trainer 代码默认的 `flash_attention_2`。
- 通过 `PYTHONPATH=/tmp/flash_attn_sm80_sm90_site:$PYTHONPATH` 让默认 `flash_attention_2` 加载包含 `sm_90` 的 flash-attn 扩展。
- 保留 `PYTHONNOUSERSITE=1`，不使用 user site。

同时修复了脚本中的 `cuobjdump | grep -q` 检查：在 `set -o pipefail` 下 `grep -q` 可能使 `cuobjdump` 收到 SIGPIPE，导致误判。当前改为：

```bash
cuobjdump --list-elf "$FLASH_ATTN_SM90_EXT" | awk '/sm_90/{found=1} END{exit !found}'
```
