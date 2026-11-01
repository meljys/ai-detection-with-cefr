"""
Shared inference contract, used by calibration, the backend and the experiments:

    from training.inference import predict_probs
    probs = predict_probs(["first text", "second text"], "models/e3_lora_r8_lr5e-4")
    # probs.shape == (2, 8); each row sums to 1; columns follow common.schema.LABELS
    # p_ai = probs[:, AI_ID]; level estimate = most likely of the 7 human columns

Works for LoRA adapters (the base model is downloaded from Hugging Face once and cached)
and full fine-tuned folders. Runs on the GPU when there is one, else on the CPU.
"""
import warnings
import zlib
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.schema import ID2LABEL, LABEL2ID, NUM_LABELS
from training.model_factory import MODEL_ID

_MODEL_CACHE: dict[str, tuple] = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _is_lora(ckpt: str) -> bool:
    return (Path(ckpt) / "adapter_config.json").exists()


def _load(ckpt: str):
    if ckpt in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt]

    if _is_lora(ckpt):
        # LoRA adapter: the folder contains only adapter_config.json +
        # adapter_model.safetensors + modules_to_save outputs (classifier,
        # pre_classifier). It has no base-model config or tokenizer, so both
        # must come from the base model.
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = PeftModel.from_pretrained(base, ckpt)
    else:
        # Full fine-tuned checkpoint: everything is self-contained.
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)

    model.to(DEVICE).eval()
    _MODEL_CACHE[ckpt] = (tok, model)
    return tok, model


def _stub_probs(texts: list[str]) -> np.ndarray:
    """Fake probabilities (a Dirichlet sample seeded by each text), for developing without a model."""
    out = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(zlib.crc32(t.encode("utf-8")))  # same text -> same output, every run
        out[i] = rng.dirichlet(np.ones(NUM_LABELS))
    return out


@torch.inference_mode()
def predict_probs(
    texts: list[str],
    ckpt: str | None,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """(n, 8) softmax probabilities. max_length 512 matches training; longer texts are truncated."""
    if ckpt is None:
        warnings.warn("predict_probs called with ckpt=None: returning FAKE probabilities.", stacklevel=2)
        return _stub_probs(texts)

    tok, model = _load(str(ckpt))
    out = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        out[i : i + len(batch)] = softmax(logits.float().cpu().numpy(), axis=-1)
    return out
