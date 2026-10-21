"""
Paths and training configurations. One TrainConfig per run; add new runs to CONFIGS
under a new name (models are saved to models/<name>/, so names are never reused).
"""
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
EXP_DIR = REPO_ROOT / "experiments" / "training"        # results1.xlsx, comparison_table.csv, figures/
RUNS_DIR = EXP_DIR / "runs"                              # per-run JSON: <name>_metrics.json, eval_<name>_<split>.json

for _d in (MODELS_DIR, EXP_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    name: str
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lr: float = 2e-5
    epochs: int = 4
    batch_size: int = 16
    grad_accum: int = 2
    max_length: int = 512
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    use_class_weights: bool = True
    ordinal_smoothing: float = 0.0 # share of the target moved to adjacent CEFR levels (0 = plain labels)
    freeze_layers: int = 0         # full FT only: freeze bottom N transformer layers
    seed: int = 2111

    def to_dict(self):
        return asdict(self)


CONFIGS = {
    # ---- Experiment series for results1.xlsx (Exp 1 = the Colab models, copied to models/e1_*) ----
    # Exp 2: Exp 1 with more epochs (LoRA 10, full FT 4), LoRA lr unchanged
    "e2_full_ft": TrainConfig(name="e2_full_ft", use_lora=False, lr=2e-5, epochs=4),
    "e2_lora_r4": TrainConfig(name="e2_lora_r4", use_lora=True, lora_r=4, lr=1e-4, epochs=10),
    "e2_lora_r8": TrainConfig(name="e2_lora_r8", use_lora=True, lora_r=8, lr=1e-4, epochs=10),
    "e2_lora_r16": TrainConfig(name="e2_lora_r16", use_lora=True, lora_r=16, lr=1e-4, epochs=10),
    
    # Exp 3: learning-rate sweeps (LoRA r=8 and full FT); lr 1e-4 / 2e-5 are e2_lora_r8 / e2_full_ft
    **{f"e3_lora_r8_lr{tag}": TrainConfig(name=f"e3_lora_r8_lr{tag}", use_lora=True, lora_r=8, lr=float(tag), epochs=10)
       for tag in ("5e-5", "3e-4", "5e-4", "1e-3")},
    **{f"e3_full_ft_lr{tag}": TrainConfig(name=f"e3_full_ft_lr{tag}", use_lora=False, lr=float(tag), epochs=4)
       for tag in ("1e-5", "5e-5", "1e-4", "3e-4", "5e-4")},
    
    # Exp 4: ordinal-smoothing sweep at the best LoRA lr from Exp 3 (5e-4); smoothing 0 is e3_lora_r8_lr5e-4
    **{f"e4_lora_r8_lr5e-4_s{tag}": TrainConfig(name=f"e4_lora_r8_lr5e-4_s{tag}", use_lora=True, lora_r=8, lr=5e-4,
                                                epochs=10, ordinal_smoothing=int(tag) / 10)
       for tag in ("01", "02", "03")},
    
    # Final: the chosen settings repeated with two more seeds (the seed-2111 runs are
    # e3_lora_r8_lr5e-4 and e3_full_ft_lr1e-4), to measure run-to-run variation
    **{f"final_lora_seed{sd}": TrainConfig(name=f"final_lora_seed{sd}", use_lora=True, lora_r=8, lr=5e-4, epochs=10, seed=sd)
       for sd in (1, 2)},
    **{f"final_fft_seed{sd}": TrainConfig(name=f"final_fft_seed{sd}", use_lora=False, lr=1e-4, epochs=4, seed=sd)
       for sd in (1, 2)},

    # ---- One-epoch smoke test: use with limit=200 to check the pipeline in under a minute ----
    "smoke": TrainConfig(name="smoke", use_lora=True, lora_r=8, lr=1e-4, epochs=1),
}

# The model handed to calibration and the backend (see training/README.md).
FINAL_MODEL = "e3_lora_r8_lr5e-4"
