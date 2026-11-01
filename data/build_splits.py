"""
Combine human (COREFL) and AI texts and split them into train / val / test,
plus a separate test set for the held-out generator.

Run from the repo root (after corefl_loader.py and generate_ai_text.py --combine):

    python -m data.build_splits

Notes:
- All texts from the held-out generator go only into test_heldout_generator.
- Everything else is split ~70 / 15 / 15 at random, stratified by label
  and grouped by learner_id, so no writer appears in more than one split.
- Same seed -> same split every time. Regenerated AI texts keep their
  text_id, so they stay in the same split.

Output: data/processed/{train,val,test,test_heldout_generator}.parquet
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common.schema import DATASET_COLUMNS, ID2LABEL, SPLITS
from data.corefl_loader import DATA_DIR, INTERIM_DIR
from data.generate_ai_text import HELD_OUT_MODEL, SEED

PROCESSED_DIR = DATA_DIR / "processed"

N_FOLDS = 20           # 20 folds of ~5% each
TEST_FOLDS = {0, 1, 2} # 3 folds -> ~15%
VAL_FOLDS = {3, 4, 5}  # 3 folds -> ~15%; remaining 14 -> ~70% train


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Return a split name for every row of df (not including held-out rows)."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    split = pd.Series("train", index=df.index)
    for fold, (_, idx) in enumerate(sgkf.split(df, y=df["label"], groups=df["learner_id"])):
        if fold in TEST_FOLDS:
            split.iloc[idx] = "test"
        elif fold in VAL_FOLDS:
            split.iloc[idx] = "val"
    return split


def build_splits() -> pd.DataFrame:
    human = pd.read_parquet(INTERIM_DIR / "corefl_clean.parquet")
    ai = pd.read_parquet(INTERIM_DIR / "ai_texts.parquet")
    df = pd.concat([human, ai], ignore_index=True)

    heldout = df["source_model"] == HELD_OUT_MODEL
    df.loc[heldout, "split"] = "test_heldout_generator"
    df.loc[~heldout, "split"] = assign_splits(df[~heldout])

    # Sanity checks
    assert df["text_id"].is_unique, "duplicate text_id" # check 1: no duplicate IDs
    writers_per_split = df.groupby("learner_id")["split"].nunique()
    assert (writers_per_split == 1).all(), "a writer appears in more than one split" # check 2: no writer in two splits

    # Schema columns
    df = df[DATASET_COLUMNS]

    PROCESSED_DIR.mkdir(exist_ok=True)
    for name in SPLITS:
        part = df[df["split"] == name].reset_index(drop=True)
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    # Summary
    counts = pd.crosstab(df["label"].map(ID2LABEL), df["split"], margins=True)
    print("Texts per class and split:\n", counts[SPLITS + ["All"]])
    shares = df[~heldout]["split"].value_counts(normalize=True).round(3)
    print("\nShare of main data per split:\n", shares)
    return df


if __name__ == "__main__":
    build_splits()
