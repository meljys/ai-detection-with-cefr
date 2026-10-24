# calibration/

CEFR aware calibration for the 8 class AI text detector. Calibrated the frozen fine-tuned model (`models/e3_lora_r8_lr5e-4`) and decides whether a text is flagged as AI. Model was not retrained nor changed.

## details

The model outputs 8 probabilities per text: `native, A1, A2, B1, B2, C1, C2, ai` (order set in `common/schema.py`)

1. Guesses the writer's CEFR level from the 7 human class probabilities
2. Looks up a threshold for the CEFR level (which is fitted on validation dataset) and flags the text as AI if `p_ai` is at or above it
3. Gives a calibrated score (the threshold is rescaled to 0.5) and a confidence value

Results:
1. All fitted thresholds remained at 0.5 floor, so flagging behaves like a 0.5 cutoff
2. This means that the fine-tuned model was already near-fear, so calibration was like a no-op on our data

## Next steps

```python

import numpy as np
from training.inference import predict_probs
from calibration.cefr_estimator import estimate_cefr_document
from calibration.threshold_calibrator import load_thresholds, apply_thresholds
from calibration.confidence import confidence

thresholds = load_thresholds("experiments/calibration/thresholds.json")

probs = predict_probs(chunks, ckpt)
doc_level = estimate_cefr_document(probs) # one detected level for the entire DOCUMENT
levels = [doc_level] * len(chunks) # same level for every chunk
score, flagged = apply_thresholds(probs, levels, thresholds)
conf = confidence(score, probs)
```

1. uses document-level CEFR estimate
2. `flagged` is per chunk, overall "% AI" is a length-weighted mean of `flagged`
3. `thresholds.json` keys are level names

## what each files does

| file | what it does |
| --- | --- |
| `cefr_estimator.py` | Drops the `ai` column, renormalise the 7 human probabilities (to total 1) and takes the argmax. `estimate_cefr(probs)` gives one level per row; `estimate_cefr_document(chunk_probs)` averages the chunks first and returns one level. `to_group` is level to level or low/mid/high band. `estimator_accuracy` is how accurate the prediction is. |
| `threshold_calibrator.py` | `fit_threshold(...)` takes the human texts (from validation dataset) of each group, and sets the threshold at the (1 - target FPR) quantile of their `p_ai`, clipped to [0.5, 0.95]. Groups with less than 20 texts uses pooled value. `apply_thresholds(...)` returns `(calibrated_score, flagged)`. `save_thresholds` and `load_thresholds` read and write the JSON. |
| `confidence.py` | `confidence(score, probs)` = 0.5 x margin (distance of calibrated score from the 0.5 line) + 0.5 x certainty (1 - normalised entropy of the 8 probabilities). Ranges from 0 to 1. |
| `fairness_metrics.py` | Evaluation. FPR per level with Wildon 95% intervals, the fairness gaps (highest minus lowest FPR) with a bootstrap interval, AI recall (overall and by writing style), precision, recall, F1, AUROC, FPR-vs-threshold curve. |
| `run_calibration.py` | Running script. Loads splits, cached model probabilities, sanity check against training results, fits and applies thresholds for 4 experiments. |
| `__init__.py` | To import calibration as package. |

## Reproduction

To run the experiments again, run from repo root:
```bash
python -u -m calibration.run_calibration | tee experiments/calibration/run_log.txt
```

1. Data are from `data/processed/val.parquet`, `data/processed/test.parquet` and `data/processed/test_heldout_generator.parquet`.
2. Model is from `models/`
3. Model probabilities cached in `experiments/calibration/probs/*.npy`
4. Dependencies: numpy, pandas, scikit-learn, matplotlib, tqdm, training package

## output

| file | details |
| --- | --- |
| `thresholds.json` | Fitted per level thresholds |
| `fpr_by_level.csv` | FPR per level, before and after, with 95% intervals. |
| `ablations.csv` | Per level vs per band, target FPR 1%, 5%, 10% |
| `fig_fpr_by_level.png` | FPR per level before and after |
| `fig_tradeoff.png` | AI recall and per level FPR across all possible thresholds |
| `fig_thresholds.png` | Fitted threshold for each level |
| `run_log.txt` | Full output of final run |

- The test set is small (27 native, 42 C2 texts), so confidence intervals are wide.