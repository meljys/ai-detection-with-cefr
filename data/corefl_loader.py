"""
Load COREFL written texts (learners + natives) into the shared schema format.

Run from the repo root:

    python -m data.corefl_loader

Output columns: text_id, text, label, cefr, is_ai, source_model,
                prompt_id, task_title, learner_id, l1
"""
import re
from pathlib import Path
import pandas as pd
from common.schema import LABEL2ID

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

CEFR_RE = r"^(A1|A2|B1|B2|C1|C2)"


def _clean(text: str) -> str:
    """
    Normalise whitespace only: replace non-breaking spaces, remove the
    byte-order mark, collapse runs of spaces/tabs/newlines into one space,
    and strip the ends. Spelling, grammar and punctuation are left unchanged.
    """
    text = str(text).replace("\u00a0", " ").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def _learner_id(filename: str, task: str) -> str:
    """
    COREFL filenames follow L1_medium_level_age_LoI_task_initials (learners)
    or L1_medium_age_task_initials (natives). Drop the task number so 
    the same writer's texts on different tasks share one ID. Falls back to the 
    filename if the pattern doesn't match.
    """
    parts = filename.split("_")
    if len(parts) >= 3 and parts[-2] == task:
        return "_".join(parts[:-2] + parts[-1:])
    return filename


def load_corefl(learners_path: str, natives_path: str, min_words: int = 1) -> pd.DataFrame:
    """
    Load and clean COREFL learner + native texts into schema columns.
    Returns one row per text; `split` is added later by build_splits.py.
    """
    lrn = pd.read_csv(learners_path, sep="\t")
    nat = pd.read_csv(natives_path, sep="\t")

    lrn["cefr"] = lrn["Proficiency"].astype(str).str.extract(CEFR_RE)[0]
    nat["cefr"] = "native"

    df = pd.concat([lrn, nat], ignore_index=True)
    n0 = len(df)

    df = df[df["Medium"].astype(str).str.lower() == "written"]
    df = df[df["cefr"].notna() & df["Text"].notna()]

    df["text"] = df["Text"].map(_clean)
    df["task"] = (pd.to_numeric(df["Task number"], errors="coerce")
                  .astype("Int64").astype(str))
    df = df[df["task"] != "2"] # task 2 has only 17 texts in total
    df["n_words"] = df["text"].str.split().str.len()

    df = df[df["n_words"] >= min_words]
    df = df.drop_duplicates(subset="text")

    out = pd.DataFrame({
        "text_id": df["Filename"].astype(str),
        "text": df["text"],
        "label": df["cefr"].map(LABEL2ID),
        "cefr": df["cefr"],
        "is_ai": False,
        "source_model": "human",
        "prompt_id": "T" + df["task"],
        "task_title": df["Task title"],
        "learner_id": [_learner_id(f, t) for f, t in zip(df["Filename"].astype(str), df["task"])],
        "l1": df["L1"],
    }).reset_index(drop=True)

    print(f"Loaded {n0} rows -> kept {len(out)} after filtering")
    return out


if __name__ == "__main__":
    df = load_corefl(RAW_DIR / "corefl_learners_raw.csv",
                     RAW_DIR / "corefl_natives_raw.csv")

    print("\nTexts per level:\n", df["cefr"].value_counts())
    print("\nLevel x task:\n", pd.crosstab(df["prompt_id"], df["cefr"]))
    print("\nDuplicate text_ids:", df["text_id"].duplicated().sum())
    per_learner = df["learner_id"].value_counts()
    print("Learners:", len(per_learner), "| with >1 text:", (per_learner > 1).sum())
    print("\nWords per level:\n", df.assign(n=df["text"].str.split().str.len())
          .groupby("cefr")["n"].describe()[["min", "50%", "max"]])

    INTERIM_DIR.mkdir(exist_ok=True)
    df.to_parquet(INTERIM_DIR / "corefl_clean.parquet", index=False)
