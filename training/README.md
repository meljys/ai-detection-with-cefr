# Training

Fine-tunes [`ahmediqbal/ai-text-detector-model`](https://huggingface.co/ahmediqbal/ai-text-detector-model)
(DistilBERT, AI vs human) into an 8-class model: `native`, `A1`–`C2` and `ai` (labels in `common/schema.py`).
One forward pass gives both the AI likelihood and a CEFR estimate.

## Using the trained model (calibration, backend, experiments)

```python
from common.schema import AI_ID
from training.inference import predict_probs

probs = predict_probs(texts, "models/e3_lora_r8_lr5e-4")   # shape (n, 8); rows sum to 1
p_ai = probs[:, AI_ID]                                       # AI likelihood
level = probs[:, :AI_ID].argmax(axis=1)                      # CEFR estimate: most likely human class
```

- Run from the repo root, so `common` and `training` are importable.
- The model is a **LoRA adapter** (about 4 MB). On first use, the base model (about 256 MB) is downloaded
  from Hugging Face and cached. Needs `torch`, `transformers` and `peft` (see `requirements.txt`).
- The final adapter is in the repo at `models/e3_lora_r8_lr5e-4/`. Other model folders are not in git
  (`models/` is ignored apart from this one).
- Load once and reuse: `predict_probs` caches the model after the first call. It uses the GPU when there is
  one, else the CPU. Texts are cut at 512 tokens, as in training.
- `predict_probs(texts, None)` returns **fake** probabilities (with a warning), for developing without a model.

## Final model

`FINAL_MODEL = "e3_lora_r8_lr5e-4"` in `config.py`: LoRA r = 8 (alpha 16, dropout 0.1, on `q_lin`, `k_lin`,
`v_lin`, with the new head trained in full), learning rate 5e-4, 10 epochs (best epoch kept by validation
macro-F1), batch size 16 × 2 accumulation steps, inverse-frequency class weights, no ordinal smoothing, seed 2111.

## Reproducing the experiments

`ai_detection_with_cefr.ipynb` runs the experiment series section by section (Exp 1 to Final), logging every
result to `experiments/training/results1.xlsx` and drawing the figures in `experiments/training/figures/`.
Settings for every run are in `config.py`; new runs get new names, and existing model folders are never
overwritten unless `overwrite=True` is passed.

| File | Contents |
|---|---|
| `config.py` | paths, `TrainConfig`, every run's settings (`CONFIGS`), `FINAL_MODEL` |
| `data.py` | loading the parquet splits, tokenizing, class weights |
| `model_factory.py` | the base model with its new 8-class head, optionally wrapped with LoRA |
| `metrics.py` | `compute_metrics` for training; CEFR scores and confusion matrices |
| `train.py` | `train`, `run_all`, ordinal training targets |
| `inference.py` | `predict_probs` |
| `evaluate.py` | evaluation on test and the held-out generator set; the original model as the baseline |
| `results_log.py`, `plot_sweeps.py` | the results workbook and the sweep figures |

`experiments/training/results1.xlsx` is the experiment series reported in the paper.
