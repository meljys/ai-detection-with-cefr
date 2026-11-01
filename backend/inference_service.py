'''
End-to-end detection pipeline used by the API

    text -> sentences -> chunks -> predict_probs (n, 8)
         -> document-level CEFR estimate (Stage 1)
         -> per-level threshold for that CEFR level, applied per chunk (Stage 2)
         -> per-chunk results + overall % AI and confidence

Calibration functions come from the ``calibration`` package.
If it is not importable, local fallbacks with the same signatures are used.
'''

from __future__ import annotations

import json
import logging
import os

import numpy as np

from backend.chunker import DEFAULT_MAX_WORDS, chunker
from backend.model_runner import ProbModel, load_model
from common.schema import AI_ID, HUMAN_LEVELS, ChunkResult, DetectResponse

logger = logging.getLogger(__name__)

# Used for any level missing from the thresholds file
DEFAULT_THRESHOLD = 0.5

# -------------------------------------------------------------------------
# Calibration hooks (real implementation if available, else fallback)
# -------------------------------------------------------------------------

def _fallback_estimate_cefr(probs: np.ndarray) -> list[str]:
    '''
    Stage 1: drop the 'ai' column, renormalise, argmax over human levels.
    '''
    human = np.delete(np.atleast_2d(probs), AI_ID, axis = 1)
    human = human / np.clip(human.sum(axis = 1, keepdims = True), 1e-12, None)
    return [HUMAN_LEVELS[i] for i in human.argmax(axis = 1)]

def _fallback_apply_thresholds(probs, est_levels, thresholds, grouping = 'level'):
    '''
    Piecewise-linear rescale of the 'ai' column so that p == tau maps to 0.5.
    Same signature as calibration.threshold_calibrator.apply_thresholds.
    '''
    p = np.asarray(probs, dtype = float)[:, AI_ID]
    tau = np.array([thresholds.get(str(l), DEFAULT_THRESHOLD) for l in est_levels], dtype = float)
    tau = np.clip(tau, 1e-6, 1 - 1e-6)
    score = np.where(p < tau, 0.5 * p / tau, 0.5 + 0.5 * (p - tau) / (1 - tau))
    return score, p >= tau

def _fallback_confidence(calibrated_score, probs):
    '''
    0.5 * distance from the decision line + 0.5 * (1 - normalised entropy).
    Same signature as calibration.confidence.confidence.
    '''
    probs = np.asarray(probs, dtype = float)
    margin = np.abs(np.asarray(calibrated_score, dtype = float) - 0.5) * 2.0
    p = np.clip(probs, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis = 1) / np.log(probs.shape[1])
    return 0.5 * margin + 0.5 * (1.0 - entropy)

def _fallback_load_thresholds(path: str) -> dict[str, float]:
    with open(path) as f:
        return {k: float(v) for k, v in json.load(f).items()}

# Use the calibration package if present
try:
    from calibration.cefr_estimator import estimate_cefr
    from calibration.confidence import confidence as chunk_confidence
    from calibration.threshold_calibrator import apply_thresholds, load_thresholds
    CALIBRATION_SOURCE = 'calibration package'
except ImportError:
    estimate_cefr = _fallback_estimate_cefr
    apply_thresholds = _fallback_apply_thresholds
    chunk_confidence = _fallback_confidence
    load_thresholds = _fallback_load_thresholds
    CALIBRATION_SOURCE = 'backend fallback'

# -------------------------------------------------------------------------
# Detector
# -------------------------------------------------------------------------

class Detector:
    '''
    Loaded once at startup; ``detect`` is then safe to call per request.
    '''

    def __init__(self, ckpt: str | None = None, thresholds_path: str | None = None,
                 mode: str = 'stub', max_words: int = DEFAULT_MAX_WORDS,
                 model: ProbModel | None = None):
        self.model: ProbModel = model or load_model(mode, ckpt)
        self.max_words = max_words
        self.thresholds, self.thresholds_loaded = self._load_thresholds(thresholds_path)
        logger.info('Detector ready: model=%s calibration=%s thresholds_loaded=%s',
                    self.model.name, CALIBRATION_SOURCE, self.thresholds_loaded)

    @staticmethod
    def _load_thresholds(path: str | None) -> tuple[dict[str, float], bool]:
        if path and os.path.exists(path):
            th = load_thresholds(path)
            missing = [l for l in HUMAN_LEVELS if l not in th]
            if missing:
                logger.warning('Thresholds missing for %s; using %.2f', missing, DEFAULT_THRESHOLD)
            return {l: float(th.get(l, DEFAULT_THRESHOLD)) for l in HUMAN_LEVELS}, True

        logger.warning('No thresholds file at %r; using fixed %.2f for every level '
                       '(uncalibrated)', path, DEFAULT_THRESHOLD)
        return {l: DEFAULT_THRESHOLD for l in HUMAN_LEVELS}, False

    @property
    def info(self) -> dict:
        return {
            'model': self.model.name,
            'calibration': CALIBRATION_SOURCE,
            'thresholds_loaded': self.thresholds_loaded,
            'thresholds': self.thresholds,
            'max_words_per_chunk': self.max_words,
        }

    def detect(self, text: str) -> DetectResponse:
        '''
        Chunks the text, scores each chunk and returns calibrated results.
        '''

        # Input validation
        chunks = chunker(text, self.max_words)
        if not chunks:
            raise ValueError('No text to analyse')

        texts = [c.text for c in chunks]
        weights = np.array([c.word_count for c in chunks], dtype = float)
        probs = np.asarray(self.model.predict_probs(texts), dtype = float)
        if probs.shape != (len(texts), len(HUMAN_LEVELS) + 1):
            raise RuntimeError(f'Model returned shape {probs.shape}; expected ({len(texts)}, 8)')

        # Stage 1: one CEFR level per document (word-weighted mean of chunk probabilities)
        # Short chunks give noisy level estimates
        doc_probs = (probs * weights[:, None]).sum(axis = 0) / weights.sum()
        doc_level = str(estimate_cefr(doc_probs[None, :])[0])

        # Stage 2: that level's threshold, applied to every chunk
        p_ai = probs[:, AI_ID]
        calibrated, flagged = apply_thresholds(probs, [doc_level] * len(chunks), self.thresholds)
        confs = np.clip(np.asarray(chunk_confidence(np.asarray(calibrated), probs), dtype = float), 0.0, 1.0)

        results: list[ChunkResult] = []
        for i, c in enumerate(chunks):
            results.append(ChunkResult(
                text = c.text,
                start_idx = c.start_idx,
                end_idx = c.end_idx,
                p_ai_raw = round(float(p_ai[i]), 4),
                p_ai_calibrated = round(float(calibrated[i]), 4),
                flagged = bool(flagged[i]),
                confidence = round(float(confs[i]), 4),
            ))

        # Overall % AI is the share of words that sit in flagged chunks
        overall_ai_pct = 100.0 * float((weights * np.asarray(flagged, dtype = float)).sum() / weights.sum())
        overall_conf = float((weights * confs).sum() / weights.sum())

        return DetectResponse(
            estimated_cefr = doc_level,
            overall_ai_pct = round(overall_ai_pct, 2),
            overall_confidence = round(overall_conf, 4),
            chunks = results,
        )
