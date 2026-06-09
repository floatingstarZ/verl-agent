#!/usr/bin/env python3
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")

METRICS = [
    "val/webshop_task_score (not success_rate)",
    "val/success_rate",
    "episode/valid_action_ratio",
    "response_length/clip_ratio",
    "actor/grad_norm",
    "actor/kl_loss",
    "actor/pg_loss",
    "actor/value_loss",
    "actor/value_pred_mean",
    "actor/value_return_mean",
    "critic/advantages/max",
    "critic/advantages/min",
]

@dataclass
class Series:
    label: str
    path: str
    rows: List[Dict[str, float]]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse(path: str, label: str) -> Series:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = strip_ansi(f.read())
    rows: List[Dict[str, float]] = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        rest = m.group(2)
        row: Dict[str, float] = {"step": float(step)}
        for key in METRICS:
            mm = re.search(re.escape(key) + r":([-\d.]+)", rest)
            if mm:
                row[key] = float(mm.group(1))
        if len(row) > 1:
            rows.append(row)
    return Series(label=label, path=path, rows=rows)


def xy(series: Series, metric: str):
    rows = [r for r in series.rows if metric in r]
    return np.array([r["step"] for r in rows]), np.array([r[metric] for r in rows])


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(np.pad(y, (window - 1, 0), mode="edge"), kernel, mode="valid")


def plot_pair(ax, series_list: List[Series], metric: str, title: str, ylabel: Optional[str] = None, smooth_window: int = 5):
    colors = {"GiGPO seed2026": "#1f6f5b", "StepPPO-v1 seed2026": "#b5362d"}
    for series in series_list:
        x, y = xy(series, metric)
        if len(x) == 0:
            continue
        c = colors.get(series.label, None)
        ax.plot(x, y, color=c, alpha=0.22, linewidth=1.0)
        ax.plot(x, smooth(y, smooth_window), color=c, linewidth=2.2, label=series.label)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, linestyle="--", alpha=0.25)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gigpo = parse(os.path.join(root, "logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260530_114442.log"), "GiGPO seed2026")
    step = parse(os.path.join(root, "logs/webshop_step_ppo_qwen25_15b_4gpu_paper_align_simple_20260604_102927.log"), "StepPPO-v1 seed2026")
    series = [gigpo, step]

    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    specs = [
        ("val/webshop_task_score (not success_rate)", "Validation task score", "task score"),
        ("val/success_rate", "Validation success rate", "success rate"),
        ("episode/valid_action_ratio", "Train valid action ratio", "valid action ratio"),
        ("response_length/clip_ratio", "Response clip ratio", "clip ratio"),
        ("actor/grad_norm", "Actor grad norm", "grad norm"),
        ("actor/kl_loss", "Actor KL loss", "KL loss"),
        ("actor/value_loss", "Value loss (StepPPO only)", "value loss"),
        ("actor/pg_loss", "Policy gradient loss", "pg loss"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        plot_pair(ax, series, metric, title, ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("StepPPO-v1 Instability Diagnostics vs GiGPO seed2026", fontsize=18, y=0.985)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False, fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_dir = os.path.join(root, "VISULIZATION/figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fig_step_ppo_instability_diagnostics.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)

if __name__ == "__main__":
    main()
