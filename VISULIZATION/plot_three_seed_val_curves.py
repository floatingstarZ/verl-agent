#!/usr/bin/env python3
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
SEED_RE = re.compile(r"trainer\.experiment_name=.*_seed(\d+)")

METRICS = {
    "val_task_score": "val/webshop_task_score (not success_rate)",
    "val_success_rate": "val/success_rate",
    "val_text_test_score": "val/text/test_score",
}


@dataclass
class RunData:
    seed: str
    log_path: str
    steps: List[int]
    metrics: Dict[str, List[float]]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_seed(text: str) -> str:
    m = SEED_RE.search(text)
    if not m:
        return "unknown"
    return m.group(1)


def parse_metric_value(blob: str, metric_name: str):
    m = re.search(re.escape(metric_name) + r":([-\d.]+)", blob)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_log(path: str) -> RunData:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    clean = strip_ansi(raw)
    seed = parse_seed(clean)

    steps: List[int] = []
    values: Dict[str, List[float]] = {k: [] for k in METRICS}

    for line in clean.splitlines():
        sm = STEP_RE.search(line)
        if not sm:
            continue
        step = int(sm.group(1))
        rest = sm.group(2)
        row = {}
        for key, metric_name in METRICS.items():
            row[key] = parse_metric_value(rest, metric_name)
        # Keep only val rows that have all three val metrics.
        if all(v is not None for v in row.values()):
            steps.append(step)
            for key in METRICS:
                values[key].append(row[key])  # type: ignore[arg-type]

    # Sort by step
    order = np.argsort(np.array(steps))
    steps_sorted = [steps[i] for i in order]
    values_sorted = {k: [values[k][i] for i in order] for k in METRICS}
    return RunData(seed=seed, log_path=path, steps=steps_sorted, metrics=values_sorted)


def compute_window_average(steps: List[int], vals: List[float], window: int = 30) -> Tuple[List[float], List[float]]:
    if not steps:
        return [], []
    max_step = max(steps)
    x_avg: List[float] = []
    y_avg: List[float] = []
    arr_s = np.array(steps)
    arr_v = np.array(vals)
    for start in range(0, max_step + 1, window):
        end = start + window
        mask = (arr_s >= start) & (arr_s < end)
        if mask.any():
            x_avg.append(start + window / 2.0)  # 15, 45, 75, ...
            y_avg.append(float(arr_v[mask].mean()))
    return x_avg, y_avg


def last10_step_stats(steps: List[int], vals: List[float]) -> Tuple[float, float, int]:
    if not steps:
        return float("nan"), float("nan"), 0
    max_step = max(steps)
    arr_s = np.array(steps)
    arr_v = np.array(vals)
    mask = arr_s >= (max_step - 9)
    use = arr_v[mask]
    if use.size == 0:
        return float("nan"), float("nan"), 0
    return float(use.mean()), float(use.var()), int(use.size)


def save_raw_csv(out_path: str, runs: List[RunData]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "step", "val_task_score", "val_success_rate", "val_text_test_score", "log_path"])
        for run in runs:
            for i, step in enumerate(run.steps):
                w.writerow(
                    [
                        run.seed,
                        step,
                        run.metrics["val_task_score"][i],
                        run.metrics["val_success_rate"][i],
                        run.metrics["val_text_test_score"][i],
                        run.log_path,
                    ]
                )


def save_window_csv(out_path: str, runs: List[RunData], window: int = 30) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "metric", "window_center_step", "window_avg"])
        for run in runs:
            for key in METRICS:
                x_avg, y_avg = compute_window_average(run.steps, run.metrics[key], window=window)
                for x, y in zip(x_avg, y_avg):
                    w.writerow([run.seed, key, x, y])


def plot_metric(runs: List[RunData], metric_key: str, title: str, out_png: str, window: int = 30) -> None:
    plt.figure(figsize=(12, 7))

    # Colors per seed for consistency
    seeds = sorted({r.seed for r in runs})
    cmap = plt.get_cmap("tab10")
    color_map = {seed: cmap(i % 10) for i, seed in enumerate(seeds)}

    stat_lines = []
    for run in sorted(runs, key=lambda r: r.seed):
        c = color_map[run.seed]
        x = run.steps
        y = run.metrics[metric_key]
        plt.plot(x, y, color=c, alpha=0.35, linewidth=1.6, label=f"seed{run.seed} raw")

        x_avg, y_avg = compute_window_average(x, y, window=window)
        plt.plot(x_avg, y_avg, color=c, linewidth=2.6, marker="o", markersize=3.5, label=f"seed{run.seed} avg(w={window})")

        mean10, var10, n10 = last10_step_stats(x, y)
        stat_lines.append(f"seed{run.seed}: mean={mean10:.4f}, var={var10:.4f}, n={n10}")

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Metric")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=2, fontsize=9)

    stats_text = "Last 10-step stats (raw val points):\n" + "\n".join(stat_lines)
    plt.gca().text(
        0.015,
        0.02,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_all_metrics_subplots(runs: List[RunData], out_png: str, window: int = 30) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
    metric_specs = [
        ("val_task_score", "Val Task Score (not success_rate)"),
        ("val_success_rate", "Val Success Rate"),
        ("val_text_test_score", "Val Text Test Score"),
    ]

    seeds = sorted({r.seed for r in runs})
    cmap = plt.get_cmap("tab10")
    color_map = {seed: cmap(i % 10) for i, seed in enumerate(seeds)}

    for ax, (metric_key, title) in zip(axes, metric_specs):
        stat_lines = []
        for run in sorted(runs, key=lambda r: r.seed):
            c = color_map[run.seed]
            x = run.steps
            y = run.metrics[metric_key]
            ax.plot(x, y, color=c, alpha=0.35, linewidth=1.4, label=f"seed{run.seed} raw")

            x_avg, y_avg = compute_window_average(x, y, window=window)
            ax.plot(x_avg, y_avg, color=c, linewidth=2.2, marker="o", markersize=3.0, label=f"seed{run.seed} avg(w={window})")

            mean10, var10, n10 = last10_step_stats(x, y)
            stat_lines.append(f"seed{run.seed}: mean={mean10:.4f}, var={var10:.4f}, n={n10}")

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.text(
            0.015,
            0.02,
            "Last 10-step stats:\n" + "\n".join(stat_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray"),
        )

    axes[0].set_ylabel("Metric")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs = [
        os.path.join(root, "logs", "webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260530_114442.log"),
        os.path.join(root, "logs", "webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_004613.log"),
        os.path.join(root, "logs", "webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_123134.log"),
    ]

    runs = [parse_log(p) for p in logs]
    runs = [r for r in runs if r.steps]

    vis_root = os.path.join(root, "VISULIZATION")
    data_dir = os.path.join(vis_root, "data")
    fig_dir = os.path.join(vis_root, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    save_raw_csv(os.path.join(data_dir, "val_metrics_raw_points.csv"), runs)
    save_window_csv(os.path.join(data_dir, "val_metrics_window30_avg.csv"), runs, window=30)

    plot_metric(
        runs,
        metric_key="val_task_score",
        title="Val Task Score (not success_rate) vs Step",
        out_png=os.path.join(fig_dir, "fig1_val_task_score_not_success_rate.png"),
        window=30,
    )
    plot_metric(
        runs,
        metric_key="val_success_rate",
        title="Val Success Rate vs Step",
        out_png=os.path.join(fig_dir, "fig2_val_success_rate.png"),
        window=30,
    )
    plot_metric(
        runs,
        metric_key="val_text_test_score",
        title="Val Text Test Score vs Step",
        out_png=os.path.join(fig_dir, "fig3_val_text_test_score.png"),
        window=30,
    )
    plot_all_metrics_subplots(
        runs,
        out_png=os.path.join(fig_dir, "fig_all_val_metrics_subplots.png"),
        window=30,
    )

    summary_path = os.path.join(data_dir, "last10step_stats_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for key in ["val_task_score", "val_success_rate", "val_text_test_score"]:
            f.write(f"[{key}]\n")
            for run in sorted(runs, key=lambda r: r.seed):
                mean10, var10, n10 = last10_step_stats(run.steps, run.metrics[key])
                f.write(f"seed{run.seed}: mean={mean10:.6f}, var={var10:.6f}, n={n10}\n")
            f.write("\n")

    print("Generated plots and data under VISULIZATION/")
    for r in sorted(runs, key=lambda r: r.seed):
        print(f"seed{r.seed}: points={len(r.steps)}, first_step={r.steps[0]}, last_step={r.steps[-1]}")


if __name__ == "__main__":
    main()
