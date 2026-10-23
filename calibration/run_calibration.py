"""
Load splits and model probs
Sanity check against training 
for each experiment, fit thresholds on val and apply them to test, measuring the effect on false positives, gap and recall
"""

from pathlib import Path 

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

from calibration.cefr_estimator import CEFR_ORDER, estimate_cefr, estimator_accuracy
from calibration.fairness_metrics import (ai_recall, fairness_gap, fpr_vs_threshold, gap_only, overall, per_level_fpr)
from calibration.threshold_calibrator import apply_thresholds, fit_threshold, save_thresholds
from common.schema import AI_ID, LABELS

ROOT= Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "calibration" #(figures)
PROBS = OUT / "probs" #saved model outputs (cache)

# (name, grouping, target fpr)
EXPERIMENTS = [
    ("main: per-level, target 5%", "level", 0.05),
    ("per-band, target 5%", "band", 0.05),
    ("per-level, target 1%", "level", 0.01),
    ("per-level, target 10%", "level", 0.10)
]

def run_model(texts, ckpt, chunk=64):
    # predict_probs on 64 texts at a time
    # 64 as multiple of batch size 32
    from training.inference import predict_probs
    parts =[]
    for i in tqdm(range(0, len(texts), chunk, desc="running model", unit="chunk")):
        parts.append(predict_probs(texts[i:i+chunk], ckpt, batch_size=4))
    return np.vstack(parts)

def get_probs(split, texts):
    f = PROBS / f"{split}_probs.npy"
    if f.exists():
        print(f"[{split}] using saved probabilities", flush = True)
        return np.load(f)
    from training.config import FINAL_MODEL, MODELS_DIR
    from training.inference import predict_probs #run model
    probs = predict_probs(texts, str(MODELS_DIR/FINAL_MODEL), batch_size=4)
    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(f, probs)
    return probs


# def get_probs(split, texts):
#     f = PROBS / f"{split}_probs.npy"
#     if f.exists():
#         return np.load(f)
#     from training.config import FINAL_MODEL, MODELS_DIR
#     from training.inference import predict_probs #run model
#     probs = predict_probs(texts, str(MODELS_DIR/FINAL_MODEL))
#     PROBS.mkdir(parents=True, exist_ok=True)
#     np.save(f, probs)
#     return probs

def run_one(grouping, target_fpr, pv, yv, pt, yt, ph):
    thr = fit_threshold(pv, yv, target_fpr=target_fpr, grouping=grouping) # fit on val
    est = estimate_cefr(pt) # guessed levels test
    _, f_est = apply_thresholds(pt, est, thr, grouping) # apply guessed levels
    est_true = est.copy() # use true levels
    hum = yt != AI_ID
    est_true[hum] = np.array(LABELS)[yt[hum]]
    _, f_true = apply_thresholds(pt, est_true, thr, grouping)
    _, f_held = apply_thresholds(ph, estimate_cefr(ph), thr, grouping) # unseen ai generator
    return thr, f_est, f_true, f_held

def plot_fpr(before, after, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, w = np.arange(len(before)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, (df, lab) in enumerate([(before, "Before: one threshold (0.5)"),
                                   (after, "After: calibrated thresholds")]):
        err = [(df["fpr"] - df["ci_low"]).to_numpy(),
               (df["ci_high"] - df["fpr"]).to_numpy()]
        xs = x + (k - 0.5) * w
        ax.bar(xs, df["fpr"], w, yerr=err, capsize=3, label=lab)
        for xi, (_, r) in zip(xs, df.iterrows()):
            ax.text(xi, r["ci_high"] + 0.002, f"{int(r['flagged'])}/{int(r['n'])}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(before["level"])
    ax.set_ylabel("False positive rate (human texts)")
    ax.set_xlabel("Writer's true level (bars: 95% interval; labels: flagged/total)")
    ax.set_ylim(0, before["ci_high"].max() * 1.15)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_tradeoff(curve, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = curve["threshold"].to_numpy()
    j = int(np.abs(ts - 0.5).argmin())
    fig, (top, bot) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    rec = curve["AI recall"].to_numpy()
    top.plot(ts, rec, "k--", lw=2)
    top.axvline(0.5, color="grey", ls=":")
    top.set_ylim(0.7, 1.02)
    top.set_ylabel("AI recall")
    top.annotate(f"{rec[j]:.3f} at 0.5", (0.5, rec[j]), xytext=(0.58, rec[j] - 0.1),
                 arrowprops=dict(arrowstyle="->"))
    for level in CEFR_ORDER:
        bot.plot(ts, curve[level], lw=1.2, label=level)
    bot.axvline(0.5, color="grey", ls=":")
    bot.set_ylim(-0.003, 0.085)
    bot.set_ylabel("Human texts flagged (FPR)")
    bot.set_xlabel("Threshold on p(AI)")
    c2 = curve["C2"].to_numpy()[j]
    bot.annotate(f"C2: {c2:.3f} at 0.5", (0.5, c2), xytext=(0.58, c2 + 0.02),
                 arrowprops=dict(arrowstyle="->"))
    bot.legend(ncol=4, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_thresholds(thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [l for l in CEFR_ORDER if l in thr]
    fig, ax= plt.subplots(figsize=(6,3.5))
    ax.bar(keys, [thr[k] for k in keys])
    ax.axhline(0.5, color="grey", ls="--")
    ax.set_ylabel("Fitted threshold on p(AI)")
    ax.set_ylim(0,1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val = pd.read_parquet(PROC/"val.parquet")
    test = pd.read_parquet(PROC/"test.parquet")
    held = pd.read_parquet(PROC/"test_heldout_generator.parquet")
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    pv = get_probs("val", val["text"].tolist())
    pt = get_probs("test", test["text"].tolist())
    ph = get_probs("test_heldout_generator", held["text"].tolist())
    assert pv.shape == (len(val), 8) and pt.shape == (len(test), 8) and ph.shape == (len(held), 8)

    hum = yt != AI_ID
    base = pt[:, AI_ID] >= 0.5 # before = one threshold
    print(f"CHECK (test, p_ai>0.5): human ok {(~base[hum]).sum()} / flagged {base[hum].sum()}; "
          f"AI missed {(~base[~hum]).sum()} / caught {base[~hum].sum()}")
    print("(training stats: human 542 / ai 1 / 134)\n")
    print("LEVEL ESTIMATOR (test): ", estimator_accuracy(pt, yt), "\n")

    rows = []

    for i, (name, grouping, target) in enumerate(EXPERIMENTS):
        thr, f_est, f_true, f_held = run_one(grouping, target, pv, yv, pt, yt, ph)
        rows.append({
            "experiment": name,
            "thresholds": str({k: round(v,3) for k,v in thr.items()}),
            "humans_flagged_before": int(base[hum].sum()),
            "humans_flagged_after": int(f_est[hum].sum()),
            "gap_before": round(gap_only(yt, base), 4),
            "gap_after": round(gap_only(yt, f_est), 4),
            "gap_after_true_levels": round(gap_only(yt, f_true), 4),
            "ai_recall_before": round(float(base[~hum].mean()), 4),
            "ai_recall_after": round(float(f_est[~hum].mean()), 4),
            "heldout_recall_before": round(float( (ph[:, AI_ID] >= 0.5).mean()), 4),
            "heldout_recall_after": round(float(f_held.mean()), 4)
        })

        if i==0:
            save_thresholds(thr, OUT/"thresholds.json")
            before, after = per_level_fpr(yt, base), per_level_fpr(yt, f_est)
            table = before.merge(after, on=["level", "n"], suffixes= ("_before", "_after"))
            table.to_csv(OUT/"fpr_by_level.csv", index=False)

            print("===================")
            print("MAIN THRESHOLDS: ", thr, "\n")
            print(table.round(3).to_string(index=False), "\n")
            print("fairness gap before: ", fairness_gap(yt, base))
            print("fairness gap after: ", fairness_gap(yt, f_est), "\n")

            print("overall before: ", overall(yt, base, pt[:,AI_ID]), "\n")
            print("overall after: ", overall(yt, f_est, pt[:,AI_ID]), "\n")

            print("AI recall after:")
            print (ai_recall(yt, f_est, test["gen_style"]).round(3).to_string(index=False), "\n")
            plot_fpr(before, after, OUT/"fig_fpr_by_level.png")
            plot_tradeoff(fpr_vs_threshold(yt, pt[:, AI_ID]), OUT/"fig_tradeoff.png")
            plot_thresholds(thr, OUT/"fig_thresholds.png")


    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/"ablations.csv", index=False)
    print("ALL EXPERIMENTS: ")
    print(summary.drop(columns="thresholds").to_string(index=False))
    print()
    print(summary[["experiment", "thresholds"]].to_string(index=False))

if __name__ == "__main__":
    main()