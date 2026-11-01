"""Build the 8-class model. Swaps the base 2-class head and optionally wraps it with LoRA."""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.schema import ID2LABEL, LABEL2ID, NUM_LABELS
from training.config import TrainConfig

MODEL_ID = "ahmediqbal/ai-text-detector-model"


def load_base_8class(cfg: TrainConfig):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,      # swap the 2-class head for 8-class
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if cfg.freeze_layers and not cfg.use_lora:
        for layer in model.distilbert.transformer.layer[: cfg.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

    if cfg.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_lin", "k_lin", "v_lin"],
            # The new head must be trained from scratch — save it with the adapter.
            modules_to_save=["pre_classifier", "classifier"],
        )
        model = get_peft_model(model, lora_cfg)

    return tok, model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
