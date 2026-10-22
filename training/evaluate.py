"""
Evaluate saved models on the test set and the held-out generator set, and the original
2-class model as the "before" baseline. Results go to experiments/training/runs/*.json.

    from training.evaluate import evaluate, eval_original_binary_model
    eval_original_binary_model("test")
    evaluate(["e3_lora_r8_lr5e-4"])          # test + test_heldout_generator
"""
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.schema import AI_ID, LABEL2ID, NUM_LABELS
from training.config import DATA_DIR, MODELS_DIR, RUNS_DIR
from training.inference import DEVICE, predict_probs
from training.metrics import CEFR_ORDER, cefr_confusion, cefr_scores, detection_confusion
from training.model_factory import MODEL_ID


def _ckpt_name(ckpt: str) -> str:
    p = Path(ckpt)
    # If the folder is a Trainer checkpoint-XXX, use the parent config name
    if p.name.startswith("checkpoint"):
        return p.parent.name
    return p.name


def eval_checkpoint(ckpt: str, split: str = "test", max_length: int = 512,
                    name: str | None = None, overwrite: bool = False) -> dict:
    """
    Reports macro-F1, AI recall, per-CEFR false-positive rate, CEFR scores and confusion
    matrices. Works for full-FT and LoRA checkpoints.

    Results go to experiments/training/runs/eval_<name>_<split>.json; name defaults to the
    model folder's name. Refuses to replace a results file written for a different
    model folder unless overwrite=True (re-evaluating the same model is fine).
    """
    out = RUNS_DIR / f"eval_{name or _ckpt_name(ckpt)}_{split}.json"
    if out.exists() and not overwrite:
        previous = json.loads(out.read_text()).get("ckpt")
        if previous and Path(previous).resolve() != Path(ckpt).resolve():
            raise FileExistsError(
                f"{out.name} holds results for a different model ({previous}). "
                f"Pass name=... to save under another name, or overwrite=True to replace it."
            )

    df = pd.read_parquet(DATA_DIR / f"{split}.parquet").reset_index(drop=True)
    probs = predict_probs(df["text"].tolist(), ckpt, max_length=max_length)
    preds = probs.argmax(axis=-1)
    p_ai = probs[:, AI_ID]

    df["pred"] = preds
    df["p_ai"] = p_ai
    df["flag_argmax"] = preds == AI_ID
    df["flag_prob"] = p_ai > 0.5

    is_ai = df["is_ai"].astype(bool)
    human, ai = df[~is_ai], df[is_ai]

    # Per-CEFR FPR (human texts flagged as AI), using both flag definitions.
    fpr_argmax = (
        human.groupby("cefr")["flag_argmax"].mean().reindex(CEFR_ORDER).round(4)
    )
    fpr_prob = (
        human.groupby("cefr")["flag_prob"].mean().reindex(CEFR_ORDER).round(4)
    )

    ai_recall_argmax = float(ai["flag_argmax"].mean()) if len(ai) else float("nan")
    ai_recall_prob = float(ai["flag_prob"].mean()) if len(ai) else float("nan")

    macro_f1 = float(f1_score(df["label"], preds, average="macro", zero_division=0))
    cm_order = [LABEL2ID[level] for level in CEFR_ORDER] + [AI_ID]  # A1..C2, native, ai
    cm = confusion_matrix(df["label"], preds, labels=cm_order)
    per_class_f1 = f1_score(
        df["label"], preds, average=None, labels=list(range(NUM_LABELS)), zero_division=0
    )

    metrics_file = RUNS_DIR / f"{_ckpt_name(ckpt)}_metrics.json"
    train_info = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}

    valid_a = [v for v in fpr_argmax.values if not pd.isna(v)]
    valid_p = [v for v in fpr_prob.values   if not pd.isna(v)]
    summary = {
        "ckpt": str(ckpt),
        "split": split,
        "n_human": int(len(human)),
        "n_ai":    int(len(ai)),
        "macro_f1": macro_f1,
        "ai_recall_argmax": ai_recall_argmax,
        "ai_recall_prob":   ai_recall_prob,
        "fpr_by_cefr_argmax": {k: (None if pd.isna(v) else float(v)) for k, v in fpr_argmax.items()},
        "fpr_by_cefr_prob":   {k: (None if pd.isna(v) else float(v)) for k, v in fpr_prob.items()},
        "fpr_gap_argmax": float(max(valid_a) - min(valid_a)) if valid_a else None,
        "fpr_gap_prob":   float(max(valid_p) - min(valid_p)) if valid_p else None,
        "per_class_f1": {lbl: float(per_class_f1[LABEL2ID[lbl]]) for lbl in CEFR_ORDER + ["ai"]},
        **cefr_scores(probs, df["label"].to_numpy()),
        # Training cost, copied from <name>_metrics.json so it sits next to the results
        **{k: train_info.get(k) for k in ("trainable_params", "train_time_s", "peak_gpu_mem_gb")},
        "confusion_matrix": cm.tolist(),
        "labels_order": CEFR_ORDER + ["ai"],
        # Same predictions as the two separate questions calibration asks:
        "cefr_confusion_matrix": cefr_confusion(probs, df["label"].to_numpy()),  # human texts, CEFR_ORDER
        "ai_confusion_matrix": detection_confusion(is_ai, df["flag_prob"]),       # p_ai > 0.5
        "cefr_labels_order": CEFR_ORDER,
    }

    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if "confusion" not in k}, indent=2))
    print("confusion matrix, 8 classes (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CEFR_ORDER + ["ai"], columns=CEFR_ORDER + ["ai"]))
    if len(human):
        print("CEFR confusion matrix, human texts (rows=true, cols=estimated):")
        print(pd.DataFrame(summary["cefr_confusion_matrix"], index=CEFR_ORDER, columns=CEFR_ORDER))
    print("AI detection at p_ai > 0.5 (rows=true, cols=flagged):")
    print(pd.DataFrame(summary["ai_confusion_matrix"], index=["human", "ai"], columns=["human", "ai"]))
    print(f"saved {out}")
    return summary


def evaluate(names) -> None:
    """Evaluate saved models (folders in models/) on the test set and the held-out generator set."""
    for name in names:
        for split in ("test", "test_heldout_generator"):
            eval_checkpoint(str(MODELS_DIR / name), split)


# ---------------------------------------------------------------------------
# Original 2-class model ("before" baseline): per-CEFR false-positive rate
# ---------------------------------------------------------------------------
def _ai_index(id2label):
    for i, name in id2label.items():
        if "ai" in str(name).lower():
            return int(i)
    return 1  # fallback: most 2-class models put AI at index 1


def _predict_2class(texts, tok, model, ai_idx, batch_size=32, max_length=512):
    model.to(DEVICE).eval()
    p_ai = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            probs = softmax(logits.float().cpu().numpy(), axis=-1)
            p_ai.extend(probs[:, ai_idx].tolist())
    return np.asarray(p_ai)


def eval_original_binary_model(split="test", max_length=512) -> dict:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    ai_idx = _ai_index(model.config.id2label)
    print(f"[baseline] ai_idx={ai_idx}  id2label={model.config.id2label}")

    df = pd.read_parquet(DATA_DIR / f"{split}.parquet").reset_index(drop=True)
    p_ai = _predict_2class(df["text"].tolist(), tok, model, ai_idx, max_length=max_length)
    flagged = p_ai > 0.5

    is_ai = df["is_ai"].astype(bool)
    human = df[~is_ai].copy()
    ai    = df[is_ai]

    # FPR is only defined when human texts are present (i.e. NOT on the held-out-generator split)
    fpr_by_level = None
    fpr_gap = None
    if len(human):
        human["flag"] = flagged[human.index]
        series = human.groupby("cefr")["flag"].mean().reindex(CEFR_ORDER)
        fpr_by_level = {k: (None if pd.isna(v) else float(v)) for k, v in series.items()}
        valid = [v for v in series.values if not pd.isna(v)]
        if valid:
            fpr_gap = float(max(valid) - min(valid))

    ai_recall = float(flagged[ai.index].mean()) if len(ai) else None

    summary = {
        "model": MODEL_ID,
        "split": split,
        "n_human": int(len(human)),
        "n_ai": int(len(ai)),
        "ai_recall": ai_recall,
        "fpr_by_cefr": fpr_by_level,
        "fpr_max_minus_min": fpr_gap,
        "ai_confusion_matrix": detection_confusion(is_ai, flagged),  # rows = true, cols = flagged
    }
    out = RUNS_DIR / f"baseline_zero_shot_{split}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[baseline] saved {out}")
    print(json.dumps(summary, indent=2))
    return summary


def seed_summary(names) -> dict:
    """Mean and spread (sample standard deviation) of the main metrics over seed repeats."""
    rows = {"val_macro_f1": [], "test_macro_f1": [], "test_ai_recall": [], "heldout_ai_recall": [], "cefr_within1": []}
    for n in names:
        m = json.loads((RUNS_DIR / f"{n}_metrics.json").read_text())
        t = json.loads((RUNS_DIR / f"eval_{n}_test.json").read_text())
        h = json.loads((RUNS_DIR / f"eval_{n}_test_heldout_generator.json").read_text())
        rows["val_macro_f1"].append(m["val_macro_f1"])
        rows["test_macro_f1"].append(t["macro_f1"])
        rows["test_ai_recall"].append(t["ai_recall_prob"])
        rows["heldout_ai_recall"].append(h["ai_recall_prob"])
        rows["cefr_within1"].append(t["cefr_within1"])
    return {k: f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" for k, v in rows.items()}
