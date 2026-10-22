"""
Fairness metrics

labels: array of TRUE class ids
flagged: boolean array, True where text is flagged as AI
levels: CEFR_ORDER levels

FPR = share of human texts of one level wrongly flagged as AI
recall = share of AI texts flagged
fairness gap = highest FPR across levels minus lowest
wilson interval = 95% range for a rate
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from calibration.cefr_estimator import CEFR_ORDER
from common.schema import AI_ID, LABELS

def wilson_ci(k, n, z=1.96):
    # 95% range for a rate when k of n texts were flagged
    if n==0: #edge
        return np.nan, np.nan
    p = k/n
    denom = 1 + z**2 /n
    centre = (p + z*2 / (2*n)) /denom
    half = z * np.sqrt(p* (1-p) / n + z**2 / (4 * n**2 )) / denom
    return min(max(0.0, centre - half), p), max(min(1.0, centre+half), p)

def per_level_fpr(labels, flagged):
    rows =[]
    for level in CEFR_ORDER:
        m = labels == LABELS.index(level) # texts whose true level 
        n, k = int(m.sum()) , int(flagged[m].sum()) # how many texts / how many wrongly flagged
        lo, hi = wilson_ci(k,n)
        rows.append({"level": level,
                     "n": n,
                     "flagged": k,
                     "fpr": k/n if n else np.nan,,
                     "ci_low": lo,
                     "ci_high": hi})

    return pd.DataFrame(rows)

def ai_recall(labels, flagged, groups=None):
    ai = labels == AI_ID
    out = [{"group": "all AI",
            "n": int(ai.sum()),
            "recall": float(flagged[ai].mean())}]
    if groups is not None: # native vs learner style AI
        g = groups.to_numpy()
        for name in sorted(set(g[ai])):
            m = ai & (g == name)
            out.append({"group": name,
                        "n": int(m.sum()),
                        "recall": float(flagged[m].mean())})

    return pd.DataFrame(out)

def level_fprs(labels, flagged):
    out = []
    for level in CEFR_ORDER:
        m = labels == LABELS.index(level)
        if m.sum() > 0:
            out.append(flagged[m].mean())
    return np.array(out)

def gap_only(labels, flagged):
    f = level_fprs(labels, flagged)
    return float(f.max()- f.min()) # worst level's fpr must beat best level fpr

def fairness_gap(labels, flagged, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    human_idx = np.flatnonzero(labels != AI_ID)
    boots = []

    for _ in range(n_boot): #redraw human texts within replacement
        idx = rng.choice(human_idx, len(human_idx))
        boots.append(gap_only(labels[idx], flagged[idx]))
    return {"gap": gap_only(labels[human_idx], flagged[human_idx]),
            "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))}

def overall(labels, flagged, p_ai):
    ai = labels == AI_ID
    tp, fp = (flagged & ai).sum(), (flagged & -ai).sum()
    precision = tp / max(tp+fp, 1)
    recall = tp / ai.sum()
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(2* precision * recall / max(precision+recall, 1e-12)),
            "auroc": float(roc_auc_score(ai, p_ai))}

def fpr_vs_threshold(labels, p_ai, ts=np.linspace(0.05, 0.99, 95)):
    # at each possible threshold, what share is flagged?
    out = {"threshold": ts}
    for level in CEFR_ORDER:
        s = p_ai[LABELS == LABELS.index(level)]
        out[level] = [(s>=t).mean() for t in ts]
    out["AI recall"] = [(p_ai[labels == AI_ID] >= t).mean() for t in ts]
    return pd.DataFrame(out)