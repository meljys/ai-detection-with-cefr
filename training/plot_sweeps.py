"""
Figures for the learning-rate sweep (Exp 3) and the ordinal-smoothing sweep (Exp 4).

Run from the repo root after the sweeps are trained and evaluated:

    python -m training.plot_sweeps

Reads experiments/training/runs/<name>_metrics.json and eval_<name>_*.json.
Output: experiments/training/figures/{lr_sweep,smoothing_sweep}.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from training.config import EXP_DIR, RUNS_DIR

FIG_DIR = EXP_DIR / "figures"

# Learning rate -> model name. The 1e-4 (LoRA) and 2e-5 (full FT) points are the Exp 2 models.
LORA_LR = {5e-5: "e3_lora_r8_lr5e-5", 1e-4: "e2_lora_r8", 3e-4: "e3_lora_r8_lr3e-4",
           5e-4: "e3_lora_r8_lr5e-4", 1e-3: "e3_lora_r8_lr1e-3"}
FFT_LR = {1e-5: "e3_full_ft_lr1e-5", 2e-5: "e2_full_ft", 5e-5: "e3_full_ft_lr5e-5",
          1e-4: "e3_full_ft_lr1e-4", 3e-4: "e3_full_ft_lr3e-4", 5e-4: "e3_full_ft_lr5e-4"}

# Reference palette, light mode: categorical slots 1-2, text and surface tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _metric(name: str, key: str):
    """Val metrics come from <name>_metrics.json; held-out recall from the eval file."""
    if key == "heldout_recall":
        return json.loads((RUNS_DIR / f"eval_{name}_test_heldout_generator.json").read_text())["ai_recall_prob"]
    return json.loads((RUNS_DIR / f"{name}_metrics.json").read_text()).get(key)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def _series(ax, xs, ys, color: str, label: str) -> None:
    """2px line, 8px markers with a surface ring, best point labelled in text ink."""
    ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", label=label,
            marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    best = max(range(len(ys)), key=lambda i: ys[i])
    ax.annotate(f"{ys[best]:.3f}", (xs[best], ys[best]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color=TEXT)


def _min_span(ax, ys, span: float) -> None:
    """Show at least `span` on the y-axis, so differences within noise don't look like big swings."""
    lo, hi = min(ys), max(ys)
    pad = max(span - (hi - lo), 0) / 2
    ax.set_ylim(lo - pad - 0.1 * span, hi + pad + 0.2 * span)  # headroom for the value label


def lr_sweep() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    panels = [("val_macro_f1", "Validation macro-F1", "macro-F1"),
              ("heldout_recall", "AI recall on held-out generator (Qwen)", "recall, p(ai) > 0.5")]
    for ax, (key, title, ylabel) in zip(axes, panels):
        for runs, color, label in ((LORA_LR, BLUE, "LoRA r=8 (10 epochs)"), (FFT_LR, ORANGE, "Full FT (4 epochs)")):
            lrs = sorted(runs)
            _series(ax, lrs, [_metric(runs[lr], key) for lr in lrs], color, label)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate", color=TEXT_2, fontsize=9)
        _style(ax, title, ylabel)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    out = FIG_DIR / "lr_sweep.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    return out


def smoothing_sweep(runs: dict[float, str]) -> Path:
    """runs: smoothing value -> model name (0 = the best Exp 3 LoRA model)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    panels = [("val_macro_f1", "Validation macro-F1", "macro-F1"),
              ("val_cefr_within1", "Validation CEFR within one level", "share of human texts")]
    xs = sorted(runs)
    for ax, (key, title, ylabel) in zip(axes, panels):
        ys = [_metric(runs[s], key) for s in xs]
        _series(ax, xs, ys, BLUE, "LoRA r=8")
        _min_span(ax, ys, 0.06)
        ax.set_xticks(xs)
        ax.set_xlabel("ordinal smoothing", color=TEXT_2, fontsize=9)
        _style(ax, title, ylabel)
    fig.tight_layout()
    out = FIG_DIR / "smoothing_sweep.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    return out


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("saved", lr_sweep())
    runs_file = RUNS_DIR / "smoothing_sweep_runs.json"  # written once Exp 4 is trained
    if runs_file.exists():
        runs = {float(k): v for k, v in json.loads(runs_file.read_text()).items()}
        print("saved", smoothing_sweep(runs))
