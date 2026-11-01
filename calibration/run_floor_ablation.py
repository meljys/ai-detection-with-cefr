"""
No-floor ablation: what do the calibrated thresholds look like if we remove
the 0.5 safety floor? Uses the cached model probabilities, so no model is run.
Output: experiments/calibration/ablation_no_floor.csv
"""

import pandas as pd

from calibration.cefr_estimator import estimate_cefr
from calibration.fairness_metrics import gap_only, per_level_fpr
from calibration.run_calibration import OUT, PROC, get_probs
from calibration.threshold_calibrator import apply_thresholds, fit_threshold
from common.schema import AI_ID

# (name, target fpr, floor)
RUNS = [
    ("floor 0.5, target 5%", 0.05, 0.5),
    ("no floor, target 5%", 0.05, 0.0),
    ("no floor, target 10%", 0.10, 0.0),
]

def plot_no_floor(summary, n_humans, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = summary["experiment"].str.replace(", ", "\n")
    colors = ["tab:blue", "tab:orange", "tab:orange"]
    bars = ax.bar(labels, summary["humans_flagged"], color=colors)
    for b, (_, r) in zip(bars, summary.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{int(r['humans_flagged'])} flagged\nAI recall {r['ai_recall']:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"Honest texts wrongly flagged (out of {n_humans})")
    ax.set_ylim(0, summary["humans_flagged"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    val = pd.read_parquet(PROC / "val.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    held = pd.read_parquet(PROC / "test_heldout_generator.parquet")
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    pv = get_probs("val", val["text"].tolist())
    pt = get_probs("test", test["text"].tolist())
    ph = get_probs("test_heldout_generator", held["text"].tolist())

    hum = yt != AI_ID
    est = estimate_cefr(pt)
    est_held = estimate_cefr(ph)

    rows = []
    for name, target, floor in RUNS:
        thr = fit_threshold(pv, yv, target_fpr=target, grouping="level", floor=floor)
        _, flagged = apply_thresholds(pt, est, thr, "level")
        _, f_held = apply_thresholds(ph, est_held, thr, "level")
        fpr = per_level_fpr(yt, flagged)
        worst = fpr.loc[fpr["fpr"].idxmax()]
        rows.append({
            "experiment": name,
            "thresholds": str({k: round(v, 3) for k, v in thr.items()}),
            "humans_flagged": int(flagged[hum].sum()),
            "human_fpr": round(float(flagged[hum].mean()), 4),
            "gap": round(gap_only(yt, flagged), 4),
            "worst_level": worst["level"],
            "worst_fpr": round(float(worst["fpr"]), 4),
            "ai_recall": round(float(flagged[~hum].mean()), 4),
            "heldout_recall": round(float(f_held.mean()), 4),
        })

    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "ablation_no_floor.csv", index=False)
    plot_no_floor(summary, int(hum.sum()), OUT / "fig_no_floor.png")
    print("NO-FLOOR ABLATION:")
    print(summary.drop(columns="thresholds").to_string(index=False))
    print()
    print(summary[["experiment", "thresholds"]].to_string(index=False))


if __name__ == "__main__":
    main()