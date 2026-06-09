#!/usr/bin/env python3
"""Plot and summarize StepPPO time/resource diagnostics against GiGPO."""

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
METRICS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/adv",
    "timing_s/testing",
    "perf/throughput",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "response_length/mean",
    "response_length/clip_ratio",
    "global_seqlen/mean",
    "actor/value_loss",
    "actor/grad_norm",
    "episode/valid_action_ratio",
]
SUMMARY_METRICS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/adv",
    "perf/throughput",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "response_length/mean",
    "response_length/clip_ratio",
    "global_seqlen/mean",
    "actor/value_loss",
    "actor/grad_norm",
    "episode/valid_action_ratio",
]
PLOT_METRICS = [
    ("timing_s/step", "Non-validation step time", "seconds"),
    ("timing_s/gen", "Rollout generation time", "seconds"),
    ("timing_s/old_log_prob", "Old log prob time", "seconds"),
    ("timing_s/update_actor", "Actor update time", "seconds"),
    ("response_length/mean", "Response length mean", "tokens"),
    ("response_length/clip_ratio", "Response clip ratio", "ratio"),
    ("global_seqlen/mean", "Global sequence length mean", "tokens"),
    ("perf/throughput", "Throughput", "tokens/sec"),
    ("perf/max_memory_allocated_gb", "Max memory allocated", "GB"),
    ("perf/cpu_memory_used_gb", "CPU memory used", "GB"),
]


@dataclass
class Series:
    label: str
    path: str
    rows: List[Dict[str, float]]


def parse(path: str, label: str) -> Series:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = ANSI_RE.sub("", f.read())
    rows: List[Dict[str, float]] = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        rest = m.group(2)
        row: Dict[str, float] = {"step": float(step), "is_validation": float("timing_s/testing" in rest)}
        for key in METRICS:
            mm = re.search(re.escape(key) + r":([-\d.]+)", rest)
            if mm:
                row[key] = float(mm.group(1))
        if len(row) > 2:
            rows.append(row)
    return Series(label=label, path=path, rows=rows)


def rows(series: Series, validation: bool) -> List[Dict[str, float]]:
    return [r for r in series.rows if bool(r.get("is_validation", 0.0)) is validation and "timing_s/step" in r]


def values(series_rows: Iterable[Dict[str, float]], metric: str) -> np.ndarray:
    return np.array([r[metric] for r in series_rows if metric in r], dtype=float)


def xy(series: Series, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    metric_rows = [r for r in rows(series, validation=False) if metric in r]
    return np.array([r["step"] for r in metric_rows]), np.array([r[metric] for r in metric_rows])


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(np.pad(y, (window - 1, 0), mode="edge"), kernel, mode="valid")


def mean_metric(series_rows: Iterable[Dict[str, float]], metric: str) -> float:
    ys = values(series_rows, metric)
    return float(np.mean(ys)) if len(ys) else float("nan")


def fmt(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return f"{value:.3f}"


def summarize_series(series: Series) -> List[str]:
    train_rows = rows(series, validation=False)
    val_rows = rows(series, validation=True)
    lines = [
        f"[{series.label}]",
        f"path={series.path}",
        f"train_rows={len(train_rows)}, validation_rows={len(val_rows)}",
    ]
    for metric in SUMMARY_METRICS:
        ys = values(train_rows, metric)
        if not len(ys):
            continue
        lines.append(
            f"train {metric}: mean={np.mean(ys):.3f}, median={np.median(ys):.3f}, p90={np.percentile(ys, 90):.3f}"
        )
    if val_rows:
        lines.append(f"validation timing_s/step mean={mean_metric(val_rows, 'timing_s/step'):.3f}")
        lines.append(f"validation timing_s/testing mean={mean_metric(val_rows, 'timing_s/testing'):.3f}")
    return lines


def summarize_segments(gigpo: Series, step_ppo: Series) -> List[str]:
    segments = [(1, 50), (51, 100), (101, 150), (151, 187)]
    metrics = [
        "timing_s/step",
        "timing_s/gen",
        "timing_s/old_log_prob",
        "timing_s/update_actor",
        "response_length/mean",
        "response_length/clip_ratio",
        "global_seqlen/mean",
        "perf/throughput",
    ]
    lines = ["[Segment comparison: StepPPO-v1 minus GiGPO seed2026, non-validation rows]"]
    g_rows = rows(gigpo, validation=False)
    s_rows = rows(step_ppo, validation=False)
    for lo, hi in segments:
        g_seg = [r for r in g_rows if lo <= r["step"] <= hi]
        s_seg = [r for r in s_rows if lo <= r["step"] <= hi]
        lines.append(f"steps {lo}-{hi}: gigpo_rows={len(g_seg)}, step_ppo_rows={len(s_seg)}")
        for metric in metrics:
            g_mean = mean_metric(g_seg, metric)
            s_mean = mean_metric(s_seg, metric)
            ratio = s_mean / g_mean if g_mean and not np.isnan(g_mean) and not np.isnan(s_mean) else float("nan")
            lines.append(f"  {metric}: step={fmt(s_mean)}, gigpo={fmt(g_mean)}, delta={fmt(s_mean - g_mean)}, ratio={fmt(ratio)}")
    return lines


def write_summary(out_path: str, series: List[Series]) -> None:
    lines: List[str] = []
    for item in series:
        lines.extend(summarize_series(item))
        lines.append("")
    gigpo, step_ppo = series
    lines.extend(summarize_segments(gigpo, step_ppo))
    lines.append("")
    lines.append("[Component means: StepPPO-v1 / GiGPO seed2026, non-validation rows]")
    g_rows = rows(gigpo, validation=False)
    s_rows = rows(step_ppo, validation=False)
    for metric in SUMMARY_METRICS:
        g_mean = mean_metric(g_rows, metric)
        s_mean = mean_metric(s_rows, metric)
        if np.isnan(g_mean) and np.isnan(s_mean):
            continue
        ratio = s_mean / g_mean if g_mean and not np.isnan(g_mean) and not np.isnan(s_mean) else float("nan")
        lines.append(f"{metric}: step={fmt(s_mean)}, gigpo={fmt(g_mean)}, delta={fmt(s_mean - g_mean)}, ratio={fmt(ratio)}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot(series: List[Series], out_path: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(30, 11))
    colors = {"GiGPO seed2026": "#1f6f5b", "StepPPO-v1 seed2026": "#b5362d"}
    for ax, (metric, title, ylabel) in zip(axes.ravel(), PLOT_METRICS):
        for item in series:
            x, y = xy(item, metric)
            if len(x) == 0:
                continue
            c = colors[item.label]
            ax.plot(x, y, color=c, alpha=0.18, linewidth=1.0)
            ax.plot(x, smooth(y), color=c, linewidth=2.3, label=item.label)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("StepPPO-v1 Time and Resource Diagnostics vs GiGPO seed2026", fontsize=18, y=0.985)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    series = [
        parse(
            os.path.join(root, "logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260530_114442.log"),
            "GiGPO seed2026",
        ),
        parse(
            os.path.join(root, "logs/webshop_step_ppo_qwen25_15b_4gpu_paper_align_simple_20260604_102927.log"),
            "StepPPO-v1 seed2026",
        ),
    ]
    figure_dir = os.path.join(root, "VISULIZATION/figures")
    data_dir = os.path.join(root, "VISULIZATION/data")
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, "fig_step_ppo_time_resource_diagnostics.png")
    summary_path = os.path.join(data_dir, "step_ppo_time_resource_summary.txt")
    plot(series, figure_path)
    write_summary(summary_path, series)
    print(figure_path)
    print(summary_path)


if __name__ == "__main__":
    main()
