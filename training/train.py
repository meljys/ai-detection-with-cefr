"""
Train configurations from training/config.py. On CUDA training uses fp16; on CPU it pins
torch threads instead.

    from training.train import run_all
    run_all(["e3_lora_r8_lr5e-4"])

Each run saves the best epoch (by validation macro-F1) to models/<name>/ and its metrics to
experiments/training/runs/<name>_metrics.json.
"""
import dataclasses
import gc
import json
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from common.schema import LABEL2ID, NUM_LABELS
from training.config import CONFIGS, MODELS_DIR, RUNS_DIR, TrainConfig
from training.data import build_datasets, class_weights
from training.metrics import CEFR_ORDER, compute_metrics
from training.model_factory import count_trainable, load_base_8class


def _pin_cpu_threads():
    if not torch.cuda.is_available():
        n = min(8, os.cpu_count() or 4)
        torch.set_num_threads(n)
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        torch.backends.mkldnn.enabled = True


def ordinal_targets(smoothing: float) -> torch.Tensor:
    """
    8x8 matrix whose row y is the training target for true label y. Moves `smoothing`
    of the probability to the adjacent CEFR levels (half to each side; A1 and native
    have one side only), so being one level off costs less than being far off.
    The 'ai' row stays one-hot.
    """
    targets = torch.eye(NUM_LABELS)
    ids = [LABEL2ID[level] for level in CEFR_ORDER]
    for pos, y in enumerate(ids):
        neighbours = [ids[p] for p in (pos - 1, pos + 1) if 0 <= p < len(ids)]
        targets[y, y] = 1 - smoothing / 2 * len(neighbours)
        for n in neighbours:
            targets[y, n] = smoothing / 2
    return targets


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy, optionally against ordinal soft targets."""

    def __init__(self, class_weights=None, soft_targets=None, **kw):
        super().__init__(**kw)
        self._cw = (
            torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None
        )
        self._targets = soft_targets

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self._cw.to(logits.device) if self._cw is not None else None
        target = self._targets.to(logits.device)[labels] if self._targets is not None else labels
        loss = F.cross_entropy(logits.float(), target, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _ta_kwargs(cfg: TrainConfig, has_cuda: bool, n_train: int):
    kw = dict(
        output_dir=str(MODELS_DIR / cfg.name),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        # Worker processes hang in Jupyter on Windows, so load data in the main process there.
        dataloader_num_workers=0 if not has_cuda or os.name == "nt" else 2,
        fp16=has_cuda,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=True,
    )

    fields = TrainingArguments.__dataclass_fields__

    # transformers renamed eval_strategy -> evaluation_strategy -> eval_strategy
    kw["eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"] = "epoch"
    steps_per_epoch = max(1, n_train // (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    kw["warmup_steps"] = max(1, int(round(cfg.warmup_ratio * total_steps)))

    return kw


def train(cfg: TrainConfig, limit: int | None = None, overwrite: bool = False) -> dict:
    if (MODELS_DIR / cfg.name).exists() and not overwrite:
        raise FileExistsError(
            f"models/{cfg.name} already exists. Use a new config name in CONFIGS, "
            f"or pass overwrite=True to replace that model."
        )
    _pin_cpu_threads()
    set_seed(cfg.seed)
    has_cuda = torch.cuda.is_available()
    # GPU memory already held by other objects (e.g. models cached by predict_probs), so the
    # peak below counts only this run's own memory.
    gpu_in_use_before = torch.cuda.memory_allocated() if has_cuda else 0
    print(f"[train] config={cfg.name}  cuda={has_cuda}  limit={limit}")

    tok, model = load_base_8class(cfg)
    n_trainable = count_trainable(model)
    print(f"[train] trainable params: {n_trainable:,}")

    tokenized, raw = build_datasets(cfg, tok, limit=limit)
    cw = class_weights(raw["train"]) if cfg.use_class_weights else None
    if cw is not None:
        print(f"[train] class weights: {np.round(cw, 3).tolist()}")

    args = TrainingArguments(
        **_ta_kwargs(cfg, has_cuda, n_train=len(raw["train"]))
    )
    soft = ordinal_targets(cfg.ordinal_smoothing) if cfg.ordinal_smoothing > 0 else None
    trainer = WeightedTrainer(
        class_weights=cw,
        soft_targets=soft,
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    # Peak memory this run allocated on the GPU while training (weights, activations, optimizer state).
    peak_gpu_gb = (round((torch.cuda.max_memory_allocated() - gpu_in_use_before) / 2**30, 2)
                   if has_cuda else None)
    # Epoch whose checkpoint was kept (load_best_model_at_end picks the best on val).
    best_epoch = next((round(h["epoch"]) for h in trainer.state.log_history
                       if "eval_macro_f1" in h and h["step"] == trainer.state.best_global_step), None)

    # Best-on-val metrics
    val_metrics = trainer.evaluate(tokenized["val"])
    # Test metrics (use the best checkpoint that `trainer.model` now holds)
    test_out = trainer.predict(tokenized["test"])
    test_metrics = compute_metrics((test_out.predictions, test_out.label_ids))

    summary = {
        "config": cfg.name,
        "trainable_params": int(n_trainable),
        "epochs": cfg.epochs,
        "best_epoch": best_epoch,
        "train_time_s": round(elapsed, 1),
        "peak_gpu_mem_gb": peak_gpu_gb,
        "val_macro_f1": float(val_metrics["eval_macro_f1"]),
        "val_ai_auroc": float(val_metrics["eval_auroc_ai_vs_human"]),
        "val_cefr_within1": float(val_metrics["eval_cefr_within1"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_ai_auroc": float(test_metrics["auroc_ai_vs_human"]),
        "platform": platform.platform(),
    }

    out_dir = MODELS_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_dir)
    if cfg.use_lora:
        trainer.model.save_pretrained(out_dir)
    else:
        trainer.save_model(str(out_dir))
    (RUNS_DIR / f"{cfg.name}_metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "train_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"[train] {cfg.name}: {summary}")

    del trainer, model
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()
    return summary


def run_all(config_names, limit=None, overwrite=False) -> pd.DataFrame:
    """Train several configs and print a summary table of their metrics."""
    names = list(config_names)  # always explicit, so a bare run_all() can't retrain old runs
    rows = []
    for name in names:
        cfg = CONFIGS[name]
        print(f"\n=== {name} ===")
        row = train(cfg, limit=limit, overwrite=overwrite)
        rows.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    return df


def drop_checkpoints(names) -> None:
    """Delete the trainer's checkpoint-* folders: optimizer state, only needed to resume training."""
    for name in names:
        for folder in (MODELS_DIR / name).glob("checkpoint-*"):
            shutil.rmtree(folder)


def measure_gpu_memory(names) -> None:
    """
    Peak GPU memory for configs trained before it was logged: a throwaway one-epoch run of
    each (same model and batch settings; the peak is reached in the first epoch). Only the
    number is kept, written into <name>_metrics.json.
    """
    for name in names:
        cfg = dataclasses.replace(CONFIGS[name], name=f"_mem_{name}", epochs=1)
        peak = train(cfg, overwrite=True)["peak_gpu_mem_gb"]  # throwaway folder
        shutil.rmtree(MODELS_DIR / cfg.name, ignore_errors=True)
        (RUNS_DIR / f"{cfg.name}_metrics.json").unlink(missing_ok=True)
        metrics_file = RUNS_DIR / f"{name}_metrics.json"
        metrics = json.loads(metrics_file.read_text())
        metrics["peak_gpu_mem_gb"] = peak
        metrics_file.write_text(json.dumps(metrics, indent=2))
        print(f"[mem] {name}: {peak} GB")
