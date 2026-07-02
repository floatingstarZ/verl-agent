#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_DIR="$ROOT_DIR/EXPS/alfworld_effective_summary"
OUT_DIR="${OUT_DIR:-$EXP_DIR/outputs/closed_llm_probe_${RUN_TAG}}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
MODEL="${MODEL:-azure::gpt-5.5}"
BASE_URL="${BASE_URL:-${QGENIE_API_ENDPOINT:-https://qgenie-api.qualcomm.com/v1}}"
TRACE_JSON="${TRACE_JSON:-$ROOT_DIR/EXPS/analysis/024_ssca_retry_full/final_step_150_static_trace_bundle/full_trace_data.json}"

mkdir -p "$OUT_DIR"

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] MODEL=$MODEL"
echo "[INFO] MAX_SAMPLES=$MAX_SAMPLES"
echo "[INFO] TRACE_JSON=$TRACE_JSON"

python3 "$EXP_DIR/scripts/build_summary_samples.py"   --trace-json "$TRACE_JSON"   --prompt-template "$EXP_DIR/prompts/simple_experience_summary_v1.txt"   --output-jsonl "$OUT_DIR/samples.jsonl"   --max-samples "$MAX_SAMPLES"

python3 "$EXP_DIR/scripts/run_closed_llm_summary.py"   --input-jsonl "$OUT_DIR/samples.jsonl"   --output-jsonl "$OUT_DIR/closed_llm_summaries.jsonl"   --model "$MODEL"   --base-url "$BASE_URL"   --max-samples "$MAX_SAMPLES"   --resume

python3 "$EXP_DIR/scripts/evaluate_closed_llm_summary.py"   --input-jsonl "$OUT_DIR/closed_llm_summaries.jsonl"   --output-md "$OUT_DIR/eval_report.md"   --output-json "$OUT_DIR/eval_details.json"

echo "[DONE] $OUT_DIR"
