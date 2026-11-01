"""compute_metrics for the HF Trainer, plus the CEFR and confusion-matrix measures used in evaluation."""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from common.schema import AI_ID, ID2LABEL, LABEL2ID, NUM_LABELS

# CEFR levels from lowest to highest; natives sit above C2. 'ai' has no position.
CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]
CEFR_POS = np.array([CEFR_ORDER.index(ID2LABEL[i]) if ID2LABEL[i] in CEFR_ORDER else -1
                     for i in range(NUM_LABELS)])
HUMAN_IDS = [i for i in range(NUM_LABELS) if i != AI_ID]


def _human_level_estimates(probs, labels):
    """(true, estimated) label ids for the human texts. Estimate = most likely human class, ignoring 'ai'."""
    labels = np.asarray(labels)
    human = labels != AI_ID
    est = np.asarray(HUMAN_IDS)[probs[human][:, HUMAN_IDS].argmax(axis=-1)]
    return labels[human], est


def cefr_scores(probs, labels):
    """
    CEFR estimate scored on human texts only.
    exact: same level; within1: at most one level off; mae: mean number of levels off;
    macro_f1: F1 averaged over the 7 human classes.
    """
    true, est = _human_level_estimates(probs, labels)
    if len(true) == 0:
        return {k: float("nan") for k in ("cefr_exact", "cefr_within1", "cefr_mae", "cefr_macro_f1")}
    off = np.abs(CEFR_POS[est] - CEFR_POS[true])
    return {"cefr_exact": float((off == 0).mean()),
            "cefr_within1": float((off <= 1).mean()),
            "cefr_mae": float(off.mean()),
            "cefr_macro_f1": float(f1_score(true, est, labels=HUMAN_IDS, average="macro", zero_division=0))}


def cefr_confusion(probs, labels) -> list | None:
    """7x7 counts on human texts: rows = true level, columns = estimated level, in CEFR_ORDER.
    None when there are no human texts (e.g. test_heldout_generator)."""
    true, est = _human_level_estimates(probs, labels)
    if len(true) == 0:
        return None
    order = [LABEL2ID[level] for level in CEFR_ORDER]
    return confusion_matrix(true, est, labels=order).tolist()


def detection_confusion(is_ai, flagged) -> list:
    """2x2 counts: rows = true (human, ai), columns = flagged (human, ai)."""
    return confusion_matrix(np.asarray(is_ai, bool), np.asarray(flagged, bool), labels=[False, True]).tolist()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = softmax(logits, axis=-1)
    preds = probs.argmax(axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(
        labels, preds, average=None, labels=list(range(NUM_LABELS)), zero_division=0
    )

    p_ai = probs[:, AI_ID]
    is_ai = (labels == AI_ID).astype(int)
    try:
        auroc = roc_auc_score(is_ai, p_ai) if len(np.unique(is_ai)) > 1 else float("nan")
    except ValueError:
        auroc = float("nan")

    out = {"macro_f1": macro_f1, "auroc_ai_vs_human": auroc}
    out.update({f"f1_{ID2LABEL[i]}": float(per_class[i]) for i in range(NUM_LABELS)})
    out.update(cefr_scores(probs, labels))
    return out
