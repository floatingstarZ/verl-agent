#!/usr/bin/env python3
"""Plot StepPPO validation curves against three-seed GiGPO results.

The script keeps the existing VISULIZATION style but adds a method-level
comparison: GiGPO is summarized across three complete seeds as mean +/- std, and
available StepPPO logs are drawn as individual curves.
"""

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
SEED_RE = re.compile(r"trainer\.experiment_name=\S*_seed(\d+)")
EXP_RE = re.compile(r"trainer\.experiment_name=([^\s]+)")

METRICS = {
    "val_task_score": "val/webshop_task_score (not success_rate)",
    "val_success_rate": "val/success_rate",
    "val_text_test_score": "val/text/test_score",
}

METRIC_TITLES = {
    "val_task_score": "WebShop Task Score",
    "val_success_rate": "Success Rate",
    "val_text_test_score": "Text Test Score",
}


@dataclass
class RunData:
    method: str
    seed: str
    experiment: str
    log_path: str
    steps: List[int]
    metrics: Dict[str, List[float]]
    total_steps: Optional[int]
    complete: bool


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_metric_value(blob: str, metric_name: str) -> Optional[float]:
    match = re.search(re.escape(metric_name) + r":([-\d.]+)", blob)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def last_match(pattern: re.Pattern[str], text: str) -> Optional[str]:
    value = None
    for match in pattern.finditer(text):
        value = match.group(1)
    return value


def infer_method(experiment: str, path: str) -> str:
    name = experiment or os.path.basename(path)
    lower = name.lower()
    if "step_ppo_v1_step_norm" in lower:
        return "StepPPO-v1_step_norm"
    if "step_ppo_v0_base" in lower:
        return "StepPPO-v0_base"
    if "step_ppo" in lower:
        return "StepPPO"
    if "gigpo" in lower:
        return "GiGPO"
    if "grpo" in lower:
        return "GRPO"
    return "unknown"


def parse_log(path: str) -> RunData:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        clean = strip_ansi(f.read())

    experiment = last_match(EXP_RE, clean) or "unknown"
    seed = last_match(SEED_RE, clean) or "unknown"

    total_match = re.search(r"Total training steps: (\d+)", clean)
    total_steps = int(total_match.group(1)) if total_match else None
    complete = "Training Progress: 100%" in clean or "Final validation metrics" in clean

    steps: List[int] = []
    values: Dict[str, List[float]] = {key: [] for key in METRICS}

    for line in clean.splitlines():
        sm = STEP_RE.search(line)
        if not sm:
            continue
        step = int(sm.group(1))
        rest = sm.group(2)
        row = {key: parse_metric_value(rest, metric_name) for key, metric_name in METRICS.items()}
        if all(value is not None for value in row.values()):
            steps.append(step)
            for key in METRICS:
                values[key].append(float(row[key]))  # type: ignore[arg-type]

    order = np.argsort(np.array(steps)) if steps else np.array([], dtype=int)
    steps_sorted = [steps[i] for i in order]
    values_sorted = {key: [values[key][i] for i in order] for key in METRICS}

    return RunData(
        method=infer_method(experiment, path),
        seed=seed,
        experiment=experiment,
        log_path=path,
        steps=steps_sorted,
        metrics=values_sorted,
        total_steps=total_steps,
        complete=complete,
    )


def aggregate_runs(runs: Iterable[RunData], metric_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_to_values: Dict[int, List[float]] = {}
    for run in runs:
        for step, value in zip(run.steps, run.metrics[metric_key]):
            step_to_values.setdefault(step, []).append(value)
    steps = np.array(sorted(step_to_values), dtype=float)
    means = np.array([np.mean(step_to_values[int(step)]) for step in steps], dtype=float)
    stds = np.array([np.std(step_to_values[int(step)], ddof=0) for step in steps], dtype=float)
    return steps, means, stds


def moving_average(x: List[int], y: List[float], window_points: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if not x:
        return np.array([]), np.array([])
    arr_x = np.array(x, dtype=float)
    arr_y = np.array(y, dtype=float)
    if len(arr_y) < window_points:
        return arr_x, arr_y
    kernel = np.ones(window_points, dtype=float) / window_points
    y_pad = np.pad(arr_y, (window_points - 1, 0), mode="edge")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return arr_x, y_smooth


def save_raw_csv(path: str, runs: List[RunData]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "seed",
            "experiment",
            "step",
            "val_task_score",
            "val_success_rate",
            "val_text_test_score",
            "complete",
            "log_path",
        ])
        for run in runs:
            for idx, step in enumerate(run.steps):
                writer.writerow([
                    run.method,
                    run.seed,
                    run.experiment,
                    step,
                    run.metrics["val_task_score"][idx],
                    run.metrics["val_success_rate"][idx],
                    run.metrics["val_text_test_score"][idx],
                    int(run.complete),
                    run.log_path,
                ])


def save_summary(path: str, runs: List[RunData]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GiGPO vs StepPPO validation comparison\n\n")
        for run in runs:
            f.write(f"[{run.method}] seed={run.seed}\n")
            f.write(f"experiment={run.experiment}\n")
            f.write(f"points={len(run.steps)}, first={run.steps[0] if run.steps else 'NA'}, last={run.steps[-1] if run.steps else 'NA'}\n")
            f.write(f"total_steps={run.total_steps}, complete={run.complete}\n")
            for key in METRICS:
                if run.steps:
                    f.write(f"last_{key}={run.metrics[key][-1]:.6f}\n")
            f.write(f"log={run.log_path}\n\n")


def plot_comparison(gigpo_runs: List[RunData], step_runs: List[RunData], out_png: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
    gigpo_color = "#1f6f5b"
    step_colors = {
        "StepPPO-v1_step_norm": "#b5362d",
        "StepPPO-v0_base": "#9a6b00",
        "StepPPO": "#b5362d",
        "GRPO": "#4b4f9f",
    }

    for ax, metric_key in zip(axes, METRICS):
        x, mean, std = aggregate_runs(gigpo_runs, metric_key)
        ax.plot(x, mean, color=gigpo_color, linewidth=2.8, label="GiGPO mean (3 seeds)")
        ax.fill_between(x, mean - std, mean + std, color=gigpo_color, alpha=0.18, label="GiGPO +/- 1 std")
        for run in gigpo_runs:
            ax.plot(run.steps, run.metrics[metric_key], color=gigpo_color, alpha=0.18, linewidth=1.0)

        for run in step_runs:
            c = step_colors.get(run.method, "#b5362d")
            label = f"{run.method} seed{run.seed}"
            if not run.complete:
                label += " (partial)"
            sx, sy = moving_average(run.steps, run.metrics[metric_key], window_points=3)
            ax.plot(run.steps, run.metrics[metric_key], color=c, alpha=0.28, linewidth=1.2)
            ax.plot(sx, sy, color=c, linewidth=2.6, marker="o", markersize=3.0, label=label)

        ax.set_title(METRIC_TITLES[metric_key], fontsize=14)
        ax.set_xlabel("Training step")
        ax.grid(True, linestyle="--", alpha=0.25)
        if metric_key in {"val_task_score", "val_success_rate"}:
            ax.set_ylim(0.0, 1.0)

    axes[0].set_ylabel("Validation metric")
    handles, labels = axes[0].get_legend_handles_labels()
    # Add labels from other axes while preserving order.
    seen = set(labels)
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    fig.suptitle("WebShop Validation: StepPPO vs GiGPO Three-Seed Baseline", y=0.985, fontsize=16)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        fontsize=10,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(root, "logs")
    gigpo_logs = [
        os.path.join(logs_dir, "webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260530_114442.log"),
        os.path.join(logs_dir, "webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_004613.log"),
        os.path.join(logs_dir, "webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_123134.log"),
    ]
    # Use available StepPPO 4GPU paper-align logs. At the time of writing, this is
    # the v1_step_norm seed2026 partial/full run.
    step_logs = sorted(
        os.path.join(logs_dir, name)
        for name in os.listdir(logs_dir)
        if name.startswith("webshop_step_ppo_qwen25_15b_4gpu_paper_align_simple_") and name.endswith(".log")
    )

    runs = [parse_log(path) for path in gigpo_logs + step_logs if os.path.exists(path)]
    runs = [run for run in runs if run.steps]
    gigpo_runs = [run for run in runs if run.method == "GiGPO"]
    step_runs = [run for run in runs if run.method.startswith("StepPPO")]

    if len(gigpo_runs) != 3:
        raise RuntimeError(f"Expected 3 GiGPO runs, got {len(gigpo_runs)}")
    if not step_runs:
        raise RuntimeError("No StepPPO run with validation points found")

    vis_root = os.path.join(root, "VISULIZATION")
    data_dir = os.path.join(vis_root, "data")
    fig_dir = os.path.join(vis_root, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    save_raw_csv(os.path.join(data_dir, "gigpo_vs_step_ppo_val_points.csv"), runs)
    save_summary(os.path.join(data_dir, "gigpo_vs_step_ppo_summary.txt"), runs)
    plot_comparison(gigpo_runs, step_runs, os.path.join(fig_dir, "fig_gigpo_vs_step_ppo_val_metrics.png"))

    print("Generated VISULIZATION/figures/fig_gigpo_vs_step_ppo_val_metrics.png")
    print("Generated VISULIZATION/data/gigpo_vs_step_ppo_val_points.csv")
    print("Generated VISULIZATION/data/gigpo_vs_step_ppo_summary.txt")
    for run in runs:
        print(
            f"{run.method} seed{run.seed}: points={len(run.steps)}, "
            f"last_step={run.steps[-1]}, complete={run.complete}, exp={run.experiment}"
        )


if __name__ == "__main__":
    main()
