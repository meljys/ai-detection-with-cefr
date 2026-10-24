"""
For each group, find AI score that only 5% of the honest val texts exceed, use as threshold
Floor of 0.5 is to stop it from going lower
Since honest text scores are near 0, most threshold should end at 0.5
Aim is to have low FPR (false positive rate), such that human written texts do not get flagged as AI unfairly.
And to have high recall, such that AI generated texts are flagged correctly
"""

import json
from pathlib import Path

import numpy as np

from calibration.cefr_estimator import to_group
from common.schema import AI_ID, HUMAN_LEVELS, LABELS

def fit_threshold(val_probs, val_labels, target_fpr=0.05, 
                  floor=0.5, ceiling=0.95, 
                  grouping="level", min_n=20):
    p_ai = val_probs[:, AI_ID] # ai score of every val text
    human = val_labels != AI_ID
    scores = p_ai[human] # ai score of human texts
    groups = to_group(np.array(LABELS)[val_labels[human]], grouping) # each human text true group
    q = 1.0 - target_fpr # 0.95 (95th percentile)
    pooled = float(np.quantile(scores, q)) # fallback: all human texts tgt

    thresholds = {}
    for g in sorted(set(to_group(HUMAN_LEVELS, grouping))):
        s = scores[groups == g] # scores of human texts in this group
        tau = float(np.quantile(s, q)) if len(s) >= min_n else pooled # tau = score that only 5% of the honest texts exceed
        thresholds[str(g)] = float(np.clip(tau, floor, ceiling)) # never below 0.5 (flr) or above 0.95 (ceil)

    return thresholds

def apply_thresholds(probs, est_levels, thresholds, grouping="level"):
    p = probs[:, AI_ID]
    tau = np.array([thresholds[str(g)] for g in to_group(est_levels, grouping)]) # each text uses threshold of its guessed grp
    score = np.where(p<tau, 0.5 * p / tau, 0.5 + 0.5 * (p-tau) / (1.0 - tau))
    # rescaling so that p == tau becomes 0.5
    # 0.5 = flagged for every group
    return score, p>= tau # rescale score, flagged yes/no

def save_thresholds(thresholds, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(thresholds, indent=2))

def load_thresholds(path):
    return json.loads(Path(path).read_text())