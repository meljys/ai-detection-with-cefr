'''
Model backends implementing the shared contract

    predict_probs(texts: list[str]) -> np.ndarray  # shape (n, 8), rows sum to 1

StubModel:     deterministic fake probabilities for backend development (DETECTOR_MODE=stub)
HFModelRunner: loads the fine-tuned 8-class checkpoint once and serves it
               Handles a full checkpoint or a LoRA adapter directory (adapter_config.json)

Preprocessing must match training/train.py (same tokenizer, max_length=512, truncation).
'''

from __future__ import annotations

import hashlib
import logging
import os
from typing import Protocol

import numpy as np

from common.schema import ID2LABEL, LABEL2ID, LABELS

logger = logging.getLogger(__name__)

N_CLASSES = len(LABELS)
# Must match training/model_factory.py MODEL_ID
BASE_MODEL = 'ahmediqbal/ai-text-detector-model'

class ProbModel(Protocol):
    name: str

    def predict_probs(self, texts: list[str]) -> np.ndarray: ...

# -------------------------------------------------------------------------
# Stub
# -------------------------------------------------------------------------

class StubModel:
    '''
    Random but deterministic per text probabilities.
    Tests stay stable and the same text always gets the same result in the UI.
    '''

    name = 'stub'

    def __init__(self, ai_bias: float = 0.0):
        # >0 pushes mass toward 'ai' (useful for demos)
        self.ai_bias = ai_bias

    def predict_probs(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), N_CLASSES), dtype = np.float64)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode('utf-8')).digest()[:8], 'little')
            logits = np.random.default_rng(seed).normal(0.0, 1.0, N_CLASSES)
            logits[-1] += self.ai_bias
            e = np.exp(logits - logits.max())
            out[i] = e / e.sum()
        return out

# -------------------------------------------------------------------------
# Real model
# -------------------------------------------------------------------------

class HFModelRunner:
    '''
    Hugging Face sequence classifier, loaded once and run in batches.
    '''

    name = 'hf'

    def __init__(self, ckpt: str, device: str | None = None, batch_size: int = 16,
                 max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # LoRA adapter or full checkpoint
        is_adapter = os.path.exists(os.path.join(ckpt, 'adapter_config.json'))
        if is_adapter:
            from peft import PeftModel
            # The base model ships a 2-class head; give it the 8-class head the adapter was trained with
            # Same loading as training/inference.py
            base = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL, num_labels = N_CLASSES, ignore_mismatched_sizes = True,
                id2label = ID2LABEL, label2id = LABEL2ID
            )
            model = PeftModel.from_pretrained(base, ckpt)
            # Merge LoRA weights for faster CPU inference
            model = model.merge_and_unload()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(ckpt)

        # Label count and order must match common/schema.py
        n_labels = model.config.num_labels
        if n_labels != N_CLASSES:
            raise ValueError(f'Checkpoint has {n_labels} labels; expected {N_CLASSES} ({LABELS})')
        id2label = {int(k): v for k, v in (model.config.id2label or {}).items()}
        if id2label and [id2label[i] for i in range(N_CLASSES)] != LABELS:
            raise ValueError(f'Checkpoint label order {id2label} does not match schema {LABELS}')

        # Tokenizer saved with the checkpoint, else the base model's
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        self.model = model.to(self.device).eval()
        if self.device == 'cpu':
            torch.set_num_threads(int(os.environ.get('TORCH_THREADS', os.cpu_count() or 1)))
        logger.info('Loaded %s checkpoint from %s on %s',
                    'LoRA' if is_adapter else 'full', ckpt, self.device)

    def predict_probs(self, texts: list[str]) -> np.ndarray:
        # Input validation
        if not texts:
            return np.zeros((0, N_CLASSES))

        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(
                    batch, truncation = True, max_length = self.max_length,
                    padding = True, return_tensors = 'pt'
                ).to(self.device)
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits.float(), dim = -1).cpu().numpy())
        return np.concatenate(out, axis = 0)

# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def load_model(mode: str, ckpt: str | None) -> ProbModel:
    '''
    Returns the model backend selected by DETECTOR_MODE.
    '''

    mode = (mode or 'stub').lower()
    if mode == 'stub':
        return StubModel(ai_bias = float(os.environ.get('STUB_AI_BIAS', '0')))
    if mode in {'hf', 'real', 'model'}:
        if not ckpt:
            raise ValueError('MODEL_CKPT must be set when DETECTOR_MODE=hf')
        return HFModelRunner(ckpt)
    raise ValueError(f'Unknown DETECTOR_MODE: {mode} (use stub or hf)')
