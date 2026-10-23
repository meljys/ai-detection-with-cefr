"""
Ignores AI column
Picks the most likely CEFR level
Uses accuracy based on training
"""

import numpy as np

from common.schema import AI_ID, HUMAN_LEVELS, LABELS
# AI_ID = 7
# HUMAN_LEVELS = [native, a1, a2, b1, b2, c1, c2]

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]

RANK = {level: i for i, level in enumerate(CEFR_ORDER)}
# a1 -> 0
# a2 -> 1
# b1 -> 2
# b2 -> 3
# c1 -> 4
# c2 -> 5
# native -> 6

BANDS = {"A1": "low", "A2": "low",
         "B1": "mid", "B2": "mid",
         "C1": "high", "C2": "high",
         "native": "high"}
# divide into 3 groups

def human_probs(probs):
    h = np.delete(probs, AI_ID, axis=1) # drop 'ai' column, 7 columns
    return h / h.sum(axis=1, keepdims=True) # rescale each row, each row add up to 1 again

def estimate_cefr(probs):
    # position of the biggest of the 7 human numbers
    return np.array(HUMAN_LEVELS)[human_probs(probs).argmax(axis=1)]

def estimate_cefr_document(chunk_probs):
    # average the whole document chunks first, then pick one CEFR level
    return np.array(HUMAN_LEVELS)[int(human_probs(chunk_probs).mean(axis=0).argmax())]

def to_group(levels, grouping="level"):
    # level names become group thresholds
    if grouping=="band":
        return np.array([BANDS[str(l)] for l in levels])
    return np.array([str(l) for l in levels])

def estimator_accuracy(probs, true_labels):
    mask = true_labels != AI_ID # HUMAN TEXTS ONLY, no ai
    pred = estimate_cefr(probs[mask]) # guess levels
    true = np.array(LABELS)[true_labels[mask]] # true levels
    diff = np.abs(np.array([RANK[p] for p in pred]) - np.array([RANK[t] for t in true]))
    band_ok = to_group(pred, "band") == to_group(true, "band")
    return {"exact": float((diff==0).mean()),
            "within_one_level": float((diff<=1).mean()),
            "mean_levels_off": float(diff.mean()),
            "exact_band": float(band_ok.mean())
            }