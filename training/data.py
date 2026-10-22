"""Load the parquet splits, tokenize them, and compute class weights."""
import numpy as np
import pandas as pd
from datasets import Dataset

from common.schema import NUM_LABELS
from training.config import DATA_DIR, TrainConfig


def load_split(split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m data.build_splits` first."
        )
    return pd.read_parquet(path).reset_index(drop=True)


def build_datasets(cfg: TrainConfig, tok, limit: int | None = None):
    """
    Returns (tokenized, raw_dfs).
      tokenized: dict split -> datasets.Dataset with input_ids/attention_mask/label
      raw_dfs:   dict split -> pd.DataFrame (for per-CEFR metrics)
    `limit` truncates each split for smoke tests; leave None for real runs.
    """
    splits = ["train", "val", "test"]
    raw, tokenized = {}, {}

    for split in splits:
        df = load_split(split)
        if limit is not None and len(df) > limit:
            df = df.sample(n=limit, random_state=cfg.seed).reset_index(drop=True)
        raw[split] = df
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

        def _tok(batch):
            return tok(batch["text"], truncation=True, max_length=cfg.max_length)

        tokenized[split] = ds.map(_tok, batched=True, remove_columns=["text"])

    return tokenized, raw


def class_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Inverse-frequency class weights for cross-entropy. Safe if a class is absent."""
    counts = train_df["label"].value_counts().reindex(range(NUM_LABELS), fill_value=0)
    counts = counts.clip(lower=1)  # avoid div-by-zero if a class is missing
    w = len(train_df) / (NUM_LABELS * counts.values)
    return w.astype(np.float32)
