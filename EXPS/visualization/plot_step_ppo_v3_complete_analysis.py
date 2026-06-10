#!/usr/bin/env python3
"""Parse the completed StepPPO-v3 value-head run and generate analysis figures/tables."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "EXPS" / "figures"
DATA_DIR = ROOT / "EXPS" / "data"

V3_STAGE1_LOG = ROOT / "logs/webshop_gigpo_v1_value_aux_step_ppo_v3_value_head_detach_false_qwen25_15b_4gpu_paper_align_simple_20260609_070507.log"
V3_RESUME_LOG = ROOT / "logs/webshop_gigpo_v1_value_aux_step_ppo_v3_value_head_detach_false_qwen25_15b_4gpu_paper_align_simple_20260610_012702.log"
V3_STAGE1_DIAG = ROOT / "logs/value_diagnostics/step_ppo_v3_value_head_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250"
V3_RESUME_DIAG = ROOT / "logs/value_diagnostics/step_ppo_v3_value_head_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250_resume_20260610_012700"
BASELINE_LOGS = [
    ("GiGPO seed2026", ROOT / "logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260530_114442.log"),
    ("GiGPO seed2077", ROOT / "logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_004613.log"),
    ("GiGPO seed2501", ROOT / "logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_123134.log"),
]

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

WINDOWS = [(1, 50), (51, 100), (101, 150), (151, 200), (201, 250)]
POLICY_METRICS = [
    "episode/success_rate",
    "episode/webshop_task_score (not success_rate)",
    "episode/valid_action_ratio",
    "response_length/mean",
    "response_length/clip_ratio",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/grad_norm",
]
RUNTIME_METRICS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "perf/throughput",
    "perf/max_memory_allocated_gb",
    "perf/cpu_memory_used_gb",
]
VALUE_ACTOR_METRICS = [
    "actor/value_loss",
    "actor/value_rmse",
    "actor/value_mae",
    "actor/value_clipfrac",
    "actor/value_explained_variance",
    "actor/value_pred_mean",
    "actor/value_return_mean",
]
VAL_METRICS = ["val/success_rate", "val/webshop_task_score (not success_rate)"]


@dataclass
class LogSeries:
    label: str
    rows: List[Dict[str, float]]
    path: Path


def parse_log(path: Path, label: str) -> LogSeries:
    rows: List[Dict[str, float]] = []
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="ignore"))
    for line in text.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        rest = match.group(2)
        if " - " not in rest:
            continue
        row: Dict[str, float] = {"step": float(step), "is_validation": float("val/" in rest or "timing_s/testing" in rest)}
        for part in rest.split(" - "):
            if ":" not in part:
                continue
            key, value_text = part.rsplit(":", 1)
            value_match = FLOAT_RE.search(value_text)
            if not value_match:
                continue
            try:
                row[key.strip()] = float(value_match.group(0))
            except ValueError:
                continue
        if len(row) > 2 and 0 <= step <= 300:
            rows.append(row)
    return LogSeries(label, rows, path)


def combine_v3() -> LogSeries:
    stage1 = parse_log(V3_STAGE1_LOG, "StepPPO-v3 stage1")
    resume = parse_log(V3_RESUME_LOG, "StepPPO-v3 resume")
    rows = [r for r in stage1.rows if r["step"] <= 100]
    rows.extend(r for r in resume.rows if r["step"] >= 101)
    rows.sort(key=lambda row: (row["step"], row.get("is_validation", 0.0)))
    return LogSeries("StepPPO-v3 value head", rows, V3_RESUME_LOG)


def rows_with(series: LogSeries, metric: str, include_validation: bool = True) -> List[Dict[str, float]]:
    return [r for r in series.rows if metric in r and (include_validation or not bool(r.get("is_validation", 0.0)))]


def rows_in_window(series: LogSeries, lo: int, hi: int, metric: Optional[str] = None, include_validation: bool = True) -> List[Dict[str, float]]:
    rows = [r for r in series.rows if lo <= int(r["step"]) <= hi and (include_validation or not bool(r.get("is_validation", 0.0)))]
    if metric is not None:
        rows = [r for r in rows if metric in r]
    return rows


def metric_values(rows: Iterable[Dict[str, float]], metric: str) -> np.ndarray:
    return np.asarray([r[metric] for r in rows if metric in r], dtype=float)


def mean_or_nan(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def std_or_nan(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr)) if arr.size else float("nan")


def smooth(y: np.ndarray, window: int = 7) -> np.ndarray:
    if len(y) < 2:
        return y
    window = min(window, len(y))
    kernel = np.ones(window) / window
    return np.convolve(np.pad(y, (window - 1, 0), mode="edge"), kernel, mode="valid")


def xy(series: LogSeries, metric: str, include_validation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    rows = rows_with(series, metric, include_validation=include_validation)
    rows = [r for r in rows if 1 <= int(r["step"]) <= 250]
    return np.asarray([r["step"] for r in rows], dtype=float), np.asarray([r[metric] for r in rows], dtype=float)


def baseline_mean_by_step(series_list: Sequence[LogSeries], metric: str, include_validation: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_step: Dict[int, List[float]] = defaultdict(list)
    for series in series_list:
        for row in rows_with(series, metric, include_validation=include_validation):
            step = int(row["step"])
            if 1 <= step <= 250:
                by_step[step].append(row[metric])
    steps = np.asarray(sorted(by_step), dtype=float)
    means = np.asarray([np.mean(by_step[int(s)]) for s in steps], dtype=float)
    stds = np.asarray([np.std(by_step[int(s)]) for s in steps], dtype=float)
    return steps, means, stds


def load_diag_summary() -> List[Dict[str, float]]:
    rows_by_step: Dict[int, Dict[str, float]] = {}
    for diag_dir, lo, hi in [(V3_STAGE1_DIAG, 1, 100), (V3_RESUME_DIAG, 101, 250)]:
        path = diag_dir / "summary.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                step = int(row["global_step"])
                if lo <= step <= hi:
                    target_std = float(row.get("value_target_std", float("nan")))
                    rmse = float(row.get("value_error_rmse", float("nan")))
                    mae = float(row.get("value_error_mae", float("nan")))
                    pred_std = float(row.get("value_pred_std", float("nan")))
                    error_mean = float(row.get("value_error_mean", float("nan")))
                    row["value_nrmse"] = rmse / target_std if target_std > 1e-12 else float("nan")
                    row["value_nmae"] = mae / target_std if target_std > 1e-12 else float("nan")
                    row["value_std_ratio"] = pred_std / target_std if target_std > 1e-12 else float("nan")
                    row["value_bias_norm"] = error_mean / target_std if target_std > 1e-12 else float("nan")
                    row["value_full_batch_ev"] = 1.0 - (rmse * rmse) / (target_std * target_std) if target_std > 1e-12 else float("nan")
                    rows_by_step[step] = row
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr(rankdata(x), rankdata(y))


def load_sample_rows(lo: int, hi: int) -> List[Dict[str, float]]:
    result: List[Dict[str, float]] = []
    for diag_dir, dlo, dhi in [(V3_STAGE1_DIAG, 1, 100), (V3_RESUME_DIAG, 101, 250)]:
        start = max(lo, dlo)
        end = min(hi, dhi)
        if start > end:
            continue
        for step in range(start, end + 1):
            path = diag_dir / f"step_{step:06d}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    result.append(json.loads(line))
    return result


def sample_window_summary() -> List[Dict[str, float | str]]:
    summaries: List[Dict[str, float | str]] = []
    for lo, hi in WINDOWS:
        sample_rows = load_sample_rows(lo, hi)
        preds = np.asarray([r["value_pred_before_update"] for r in sample_rows], dtype=float)
        targets = np.asarray([r["value_target_gae_return"] for r in sample_rows], dtype=float)
        if not len(preds):
            continue
        target_std = float(np.std(targets))
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        mae = float(np.mean(np.abs(preds - targets)))
        try:
            quintile_ids = np.minimum(4, np.floor((rankdata(preds) - 1) / max(1, len(preds)) * 5).astype(int))
        except Exception:
            quintile_ids = np.zeros(len(preds), dtype=int)
        target_quintiles = [float(np.mean(targets[quintile_ids == i])) for i in range(5)]
        pred_quintiles = [float(np.mean(preds[quintile_ids == i])) for i in range(5)]
        slope = float(np.polyfit(preds, targets, 1)[0]) if np.std(preds) > 1e-12 else float("nan")
        summaries.append(
            {
                "window": f"{lo}-{hi}",
                "lo": lo,
                "hi": hi,
                "num_samples": len(preds),
                "pearson": corr(preds, targets),
                "spearman": spearman(preds, targets),
                "rmse": rmse,
                "mae": mae,
                "target_std": target_std,
                "nrmse": rmse / target_std if target_std > 1e-12 else float("nan"),
                "nmae": mae / target_std if target_std > 1e-12 else float("nan"),
                "slope_target_on_pred": slope,
                "target_q1": target_quintiles[0],
                "target_q2": target_quintiles[1],
                "target_q3": target_quintiles[2],
                "target_q4": target_quintiles[3],
                "target_q5": target_quintiles[4],
                "pred_q1": pred_quintiles[0],
                "pred_q2": pred_quintiles[1],
                "pred_q3": pred_quintiles[2],
                "pred_q4": pred_quintiles[3],
                "pred_q5": pred_quintiles[4],
                "top_bottom_target_gap": target_quintiles[4] - target_quintiles[0],
            }
        )
    return summaries


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_window_tables(v3: LogSeries, baselines: Sequence[LogSeries], diag_rows: Sequence[Dict[str, float]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    policy_rows: List[Dict[str, object]] = []
    runtime_rows: List[Dict[str, object]] = []
    value_rows: List[Dict[str, object]] = []

    for lo, hi in WINDOWS:
        label = f"{lo}-{hi}"
        for metric in POLICY_METRICS:
            v3_mean = mean_or_nan(metric_values(rows_in_window(v3, lo, hi, metric=metric, include_validation=True), metric))
            seed_means = [mean_or_nan(metric_values(rows_in_window(base, lo, hi, metric=metric, include_validation=True), metric)) for base in baselines]
            seed_means = [x for x in seed_means if not math.isnan(x)]
            base_mean = mean_or_nan(seed_means)
            policy_rows.append(
                {
                    "window": label,
                    "metric": metric,
                    "v3_mean": v3_mean,
                    "gigpo_mean": base_mean,
                    "gigpo_std_across_seeds": std_or_nan(seed_means),
                    "delta_v3_minus_gigpo": v3_mean - base_mean if not math.isnan(v3_mean) and not math.isnan(base_mean) else float("nan"),
                    "ratio_v3_over_gigpo": v3_mean / base_mean if base_mean and not math.isnan(v3_mean) and not math.isnan(base_mean) else float("nan"),
                }
            )
        for metric in RUNTIME_METRICS:
            v3_mean = mean_or_nan(metric_values(rows_in_window(v3, lo, hi, metric=metric, include_validation=False), metric))
            seed_means = [mean_or_nan(metric_values(rows_in_window(base, lo, hi, metric=metric, include_validation=False), metric)) for base in baselines]
            seed_means = [x for x in seed_means if not math.isnan(x)]
            base_mean = mean_or_nan(seed_means)
            runtime_rows.append(
                {
                    "window": label,
                    "metric": metric,
                    "v3_mean_non_validation": v3_mean,
                    "gigpo_mean_non_validation": base_mean,
                    "gigpo_std_across_seeds": std_or_nan(seed_means),
                    "delta_v3_minus_gigpo": v3_mean - base_mean if not math.isnan(v3_mean) and not math.isnan(base_mean) else float("nan"),
                    "ratio_v3_over_gigpo": v3_mean / base_mean if base_mean and not math.isnan(v3_mean) and not math.isnan(base_mean) else float("nan"),
                }
            )
        diag_win = [r for r in diag_rows if lo <= int(r["global_step"]) <= hi]
        for metric in [
            "value_pred_target_corr",
            "value_nrmse",
            "value_nmae",
            "value_std_ratio",
            "value_bias_norm",
            "value_full_batch_ev",
            "value_error_rmse",
            "value_error_mae",
            "value_target_std",
            "valid_action_ratio",
        ]:
            values = [float(r[metric]) for r in diag_win if metric in r and not math.isnan(float(r[metric]))]
            value_rows.append({"window": label, "metric": metric, "mean": mean_or_nan(values), "std": std_or_nan(values), "last": values[-1] if values else float("nan")})
    return policy_rows, runtime_rows, value_rows


def validation_summary(v3: LogSeries, baselines: Sequence[LogSeries]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    v3_val_steps = sorted({int(r["step"]) for r in v3.rows if bool(r.get("is_validation", 0.0)) and 1 <= int(r["step"]) <= 250})
    for step in v3_val_steps:
        for metric in VAL_METRICS:
            v3_values = [r[metric] for r in v3.rows if int(r["step"]) == step and metric in r]
            seed_values = []
            for base in baselines:
                vals = [r[metric] for r in base.rows if int(r["step"]) == step and metric in r]
                if vals:
                    seed_values.append(vals[-1])
            rows.append(
                {
                    "step": step,
                    "metric": metric,
                    "v3": v3_values[-1] if v3_values else float("nan"),
                    "gigpo_mean": mean_or_nan(seed_values),
                    "gigpo_std": std_or_nan(seed_values),
                    "delta_v3_minus_gigpo": (v3_values[-1] - mean_or_nan(seed_values)) if v3_values and seed_values else float("nan"),
                }
            )
    return rows


def plot_value_learning(v3: LogSeries, diag_rows: Sequence[Dict[str, float]]) -> Path:
    steps = np.asarray([r["global_step"] for r in diag_rows], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()
    plots = [
        ("value_pred_target_corr", "Pred-target correlation", "corr"),
        ("value_nrmse", "Normalized RMSE", "RMSE / target std"),
        ("value_full_batch_ev", "Full-batch implied explained variance", "1 - NRMSE^2"),
        ("value_std_ratio", "Scale matching", "pred std / target std"),
        ("value_bias_norm", "Normalized bias", "mean error / target std"),
    ]
    for ax, (metric, title, ylabel) in zip(axes[:5], plots):
        y = np.asarray([r.get(metric, float("nan")) for r in diag_rows], dtype=float)
        ax.plot(steps, y, color="#276fbf", alpha=0.25, linewidth=1)
        ax.plot(steps, smooth(y), color="#0b3d91", linewidth=2.4)
        ax.axvline(100, color="#888888", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.25)
    ax = axes[5]
    x1, y1 = xy(v3, "actor/value_loss", include_validation=True)
    x2, y2 = xy(v3, "actor/value_rmse", include_validation=True)
    ax.plot(x1, smooth(y1), label="actor/value_loss", color="#a23b72", linewidth=2.2)
    ax.plot(x2, smooth(y2), label="actor/value_rmse", color="#f18f01", linewidth=2.2)
    ax.axvline(100, color="#888888", linestyle="--", linewidth=1)
    ax.set_title("Raw actor value metrics")
    ax.set_xlabel("training step")
    ax.set_ylabel("raw value")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.suptitle("StepPPO-v3 value-head learning diagnostics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "018_step_ppo_v3_value_learning_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_policy_comparison(v3: LogSeries, baselines: Sequence[LogSeries]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()
    panels = [
        ("episode/success_rate", "Train success rate", True),
        ("episode/webshop_task_score (not success_rate)", "Train task score", True),
        ("episode/valid_action_ratio", "Valid action ratio", True),
        ("val/success_rate", "Validation success rate", True),
        ("val/webshop_task_score (not success_rate)", "Validation task score", True),
        ("response_length/clip_ratio", "Response clip ratio", True),
    ]
    for ax, (metric, title, include_val) in zip(axes, panels):
        for base in baselines:
            bx, by = xy(base, metric, include_validation=True)
            if len(bx):
                ax.plot(bx, smooth(by), color="#8e9aaf", alpha=0.25, linewidth=1)
        bx, bm, bs = baseline_mean_by_step(baselines, metric, include_validation=True)
        if len(bx):
            ax.plot(bx, smooth(bm), color="#444444", linewidth=2.2, label="GiGPO mean (3 seeds)")
            if len(bs):
                ax.fill_between(bx, bm - bs, bm + bs, color="#444444", alpha=0.10, linewidth=0)
        vx, vy = xy(v3, metric, include_validation=True)
        if len(vx):
            ax.plot(vx, smooth(vy), color="#d62828", linewidth=2.4, label="StepPPO-v3")
        ax.axvline(100, color="#888888", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(True, linestyle="--", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("StepPPO-v3 actor behavior vs GiGPO baseline", fontsize=16)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIG_DIR / "018_step_ppo_v3_policy_vs_gigpo_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_runtime_comparison(v3: LogSeries, baselines: Sequence[LogSeries]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()
    panels = [
        ("timing_s/step", "Non-validation step time", "seconds"),
        ("timing_s/gen", "Generation time", "seconds"),
        ("timing_s/update_actor", "Actor update time", "seconds"),
        ("perf/throughput", "Throughput", "tokens/sec"),
        ("perf/max_memory_allocated_gb", "Max GPU memory allocated", "GB"),
        ("perf/cpu_memory_used_gb", "CPU memory used", "GB"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for base in baselines:
            bx, by = xy(base, metric, include_validation=False)
            if len(bx):
                ax.plot(bx, smooth(by), color="#8e9aaf", alpha=0.25, linewidth=1)
        bx, bm, bs = baseline_mean_by_step(baselines, metric, include_validation=False)
        if len(bx):
            ax.plot(bx, smooth(bm), color="#444444", linewidth=2.2, label="GiGPO mean (3 seeds)")
            ax.fill_between(bx, bm - bs, bm + bs, color="#444444", alpha=0.10, linewidth=0)
        vx, vy = xy(v3, metric, include_validation=False)
        if len(vx):
            ax.plot(vx, smooth(vy), color="#d62828", linewidth=2.4, label="StepPPO-v3")
        ax.axvline(100, color="#888888", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("StepPPO-v3 runtime and resource diagnostics", fontsize=16)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIG_DIR / "018_step_ppo_v3_runtime_resource_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_value_calibration(samples: Sequence[Dict[str, float | str]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = [str(r["window"]) for r in samples]
    xs = np.arange(len(labels))
    axes[0].plot(xs, [float(r["pearson"]) for r in samples], marker="o", label="Pearson", color="#0b3d91", linewidth=2.2)
    axes[0].plot(xs, [float(r["spearman"]) for r in samples], marker="s", label="Spearman", color="#2a9d8f", linewidth=2.2)
    axes[0].plot(xs, [float(r["nrmse"]) for r in samples], marker="^", label="NRMSE", color="#f18f01", linewidth=2.2)
    axes[0].set_xticks(xs, labels, rotation=20)
    axes[0].set_title("Sample-level ranking and normalized error")
    axes[0].grid(True, linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(samples)))
    q = np.arange(1, 6)
    for color, row in zip(colors, samples):
        axes[1].plot(q, [float(row[f"target_q{i}"]) for i in range(1, 6)], marker="o", color=color, label=str(row["window"]))
    axes[1].set_title("Target return by prediction quintile")
    axes[1].set_xlabel("prediction quintile (low to high)")
    axes[1].set_ylabel("mean target return")
    axes[1].grid(True, linestyle="--", alpha=0.25)
    axes[1].legend(title="steps", frameon=False, fontsize=9)
    fig.suptitle("StepPPO-v3 value calibration/ranking from dumped samples", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = FIG_DIR / "018_step_ppo_v3_value_calibration_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def selected_lookup(rows: Sequence[Dict[str, float]], steps: Sequence[int]) -> List[Dict[str, object]]:
    by_step = {int(r["global_step"]): r for r in rows}
    out: List[Dict[str, object]] = []
    for step in steps:
        if step not in by_step:
            continue
        r = by_step[step]
        out.append(
            {
                "step": step,
                "corr": r.get("value_pred_target_corr", float("nan")),
                "nrmse": r.get("value_nrmse", float("nan")),
                "full_batch_ev": r.get("value_full_batch_ev", float("nan")),
                "std_ratio": r.get("value_std_ratio", float("nan")),
                "bias_norm": r.get("value_bias_norm", float("nan")),
                "pred_mean": r.get("value_pred_mean", float("nan")),
                "target_mean": r.get("value_target_mean", float("nan")),
                "rmse": r.get("value_error_rmse", float("nan")),
                "target_std": r.get("value_target_std", float("nan")),
            }
        )
    return out


def fmt(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "NA"
    return f"{float(value):.3f}"


def generate_summary_txt(v3: LogSeries, baselines: Sequence[LogSeries], diag_rows: Sequence[Dict[str, float]], policy_rows: Sequence[Dict[str, object]], runtime_rows: Sequence[Dict[str, object]], value_rows: Sequence[Dict[str, object]], sample_rows: Sequence[Dict[str, object]], val_rows: Sequence[Dict[str, object]]) -> None:
    def find(rows: Sequence[Dict[str, object]], window: str, metric: str, key: str) -> float:
        for row in rows:
            if row.get("window") == window and row.get("metric") == metric:
                return float(row.get(key, float("nan")))
        return float("nan")

    lines: List[str] = []
    lines.append("StepPPO-v3 completed run analysis")
    lines.append("==================================")
    lines.append(f"V3 train rows: {len([r for r in v3.rows if 1 <= int(r['step']) <= 250])}")
    lines.append(f"V3 value diagnostics rows: {len(diag_rows)}")
    lines.append(f"V3 diagnostic step range: {int(diag_rows[0]['global_step'])}-{int(diag_rows[-1]['global_step'])}")
    lines.append(f"V3 final checkpoint: 250")
    lines.append("")
    lines.append("Window highlights")
    for window in ["1-50", "51-100", "101-150", "151-200", "201-250"]:
        lines.append(f"[{window}]")
        for metric in ["episode/success_rate", "episode/webshop_task_score (not success_rate)", "episode/valid_action_ratio", "response_length/clip_ratio"]:
            lines.append(
                f"  {metric}: v3={fmt(find(policy_rows, window, metric, 'v3_mean'))}, "
                f"gigpo_mean={fmt(find(policy_rows, window, metric, 'gigpo_mean'))}, "
                f"delta={fmt(find(policy_rows, window, metric, 'delta_v3_minus_gigpo'))}"
            )
        for metric in ["value_pred_target_corr", "value_nrmse", "value_std_ratio", "value_full_batch_ev"]:
            lines.append(f"  {metric}: mean={fmt(find(value_rows, window, metric, 'mean'))}, last={fmt(find(value_rows, window, metric, 'last'))}")
        for metric in ["timing_s/step", "timing_s/update_actor", "perf/max_memory_allocated_gb"]:
            lines.append(
                f"  {metric}: v3={fmt(find(runtime_rows, window, metric, 'v3_mean_non_validation'))}, "
                f"gigpo_mean={fmt(find(runtime_rows, window, metric, 'gigpo_mean_non_validation'))}, "
                f"ratio={fmt(find(runtime_rows, window, metric, 'ratio_v3_over_gigpo'))}"
            )
    lines.append("")
    lines.append("Final validation rows")
    for metric in VAL_METRICS:
        vals = [r for r in val_rows if int(r.get("step", -1)) == 250 and r.get("metric") == metric]
        if vals:
            row = vals[-1]
            lines.append(f"  {metric}: v3={fmt(float(row['v3']))}, gigpo_mean={fmt(float(row['gigpo_mean']))}, delta={fmt(float(row['delta_v3_minus_gigpo']))}")
    lines.append("")
    lines.append("Sample-level ranking summary")
    for row in sample_rows:
        lines.append(
            f"  {row['window']}: pearson={fmt(float(row['pearson']))}, spearman={fmt(float(row['spearman']))}, "
            f"nrmse={fmt(float(row['nrmse']))}, top-bottom target gap={fmt(float(row['top_bottom_target_gap']))}"
        )
    (DATA_DIR / "018_step_ppo_v3_complete_analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    v3 = combine_v3()
    baselines = [parse_log(path, label) for label, path in BASELINE_LOGS]
    diag_rows = load_diag_summary()
    sample_rows = sample_window_summary()
    policy_rows, runtime_rows, value_rows = make_window_tables(v3, baselines, diag_rows)
    val_rows = validation_summary(v3, baselines)
    selected_rows = selected_lookup(diag_rows, [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250])

    write_csv(DATA_DIR / "018_step_ppo_v3_policy_window_summary.csv", policy_rows)
    write_csv(DATA_DIR / "018_step_ppo_v3_runtime_window_summary.csv", runtime_rows)
    write_csv(DATA_DIR / "018_step_ppo_v3_value_window_summary.csv", value_rows)
    write_csv(DATA_DIR / "018_step_ppo_v3_validation_summary.csv", val_rows)
    write_csv(DATA_DIR / "018_step_ppo_v3_value_selected_steps.csv", selected_rows)
    write_csv(DATA_DIR / "018_step_ppo_v3_value_sample_ranking_summary.csv", sample_rows)

    figures = [
        plot_value_learning(v3, diag_rows),
        plot_policy_comparison(v3, baselines),
        plot_runtime_comparison(v3, baselines),
        plot_value_calibration(sample_rows),
    ]
    generate_summary_txt(v3, baselines, diag_rows, policy_rows, runtime_rows, value_rows, sample_rows, val_rows)

    print("Generated figures:")
    for figure in figures:
        print(figure.relative_to(ROOT))
    print("Generated data:")
    for path in sorted(DATA_DIR.glob("018_step_ppo_v3_*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
