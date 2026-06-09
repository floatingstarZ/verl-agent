#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/opt/conda/bin/python"
MODE="${1:-check}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[FATAL] Missing expected python: $PYTHON_BIN" >&2
    exit 1
fi

if [ "$MODE" != "check" ] && [ "$MODE" != "apply" ]; then
    echo "Usage: $0 [check|apply]" >&2
    exit 2
fi

echo "[INFO] Mode: $MODE"
echo "[INFO] Python: $PYTHON_BIN"

MISMATCHES="$("$PYTHON_BIN" - <<'PY'
import importlib
import json
import subprocess
import sys

expected = {
    "python": "3.11.11",
    "torch": "2.9.1+cu128",
    "ray": "2.53.0",
    "vllm": "0.15.0",
    "flash_attn": "2.8.3",
    "setuptools": "80.9.0",
}

actual = {"python": sys.version.split()[0]}
for name in ["torch", "ray", "vllm", "flash_attn", "setuptools"]:
    try:
        mod = importlib.import_module(name)
        actual[name] = getattr(mod, "__version__", "unknown")
    except Exception:
        actual[name] = None

print("[INFO] Current versions:")
for k in ["python", "torch", "ray", "vllm", "flash_attn", "setuptools"]:
    print(f"  {k}={actual[k]}")

mismatch = {}
for k, v in expected.items():
    if actual.get(k) != v:
        mismatch[k] = {"expected": v, "actual": actual.get(k)}

print("MISMATCH_JSON=" + json.dumps(mismatch, ensure_ascii=True))
PY
)"

echo "$MISMATCHES"

MISMATCH_JSON="$(echo "$MISMATCHES" | awk -F'MISMATCH_JSON=' '/MISMATCH_JSON=/{print $2}')"

if [ -z "$MISMATCH_JSON" ]; then
    echo "[FATAL] Failed to parse mismatch info." >&2
    exit 3
fi

if [ "$MISMATCH_JSON" = "{}" ]; then
    echo "[OK] /opt/conda baseline already aligned."
    exit 0
fi

echo "[WARN] Found mismatches: $MISMATCH_JSON"

if [ "$MODE" = "check" ]; then
    echo "[INFO] Check mode only; no changes applied."
    exit 4
fi

echo "[INFO] Applying pip-level alignment for Python packages (torch/ray/vllm/flash_attn/setuptools)..."
"$PYTHON_BIN" -m pip install --no-cache-dir -U \
    "setuptools==80.9.0" \
    "ray==2.53.0" \
    "vllm==0.15.0" \
    "flash-attn==2.8.3"

echo "[INFO] Re-checking versions after apply..."
"$PYTHON_BIN" - <<'PY'
import sys
for name in ["torch", "ray", "vllm", "flash_attn", "setuptools"]:
    mod = __import__(name)
    print(name, getattr(mod, "__version__", "n/a"))
print("python", sys.version.split()[0], sys.executable)
PY

