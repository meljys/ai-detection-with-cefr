"""
Shared contract.

Every module imports labels, column names and API models from here.
"""
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Labels (list index = class id used by the model)
# ---------------------------------------------------------------------------
LABELS = ["native", "A1", "A2", "B1", "B2", "C1", "C2", "ai"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

HUMAN_LEVELS = LABELS[:-1] # native + A1..C2
CEFR_LEVELS = LABELS[1:-1] # A1..C2 only
AI_ID = LABEL2ID["ai"]
NUM_LABELS = len(LABELS)   # 8

# ---------------------------------------------------------------------------
# Dataset columns (data/processed/*.parquet)
# Produced by Data; consumed by Training, Calibration, Experiments
# ---------------------------------------------------------------------------
DATASET_COLUMNS = [
    "text_id",          # str: unique per text
    "text",             # str: cleaned text
    "label",            # int: index into LABELS
    "cefr",             # str: "native", "A1".."C2", or None for AI text
    "is_ai",            # bool: True for AI-generated text
    "source_model",     # str: LLM name for AI text, "human" otherwise
    "prompt_id",        # str: COREFL writing task the text responds to
    "task_title",       # str: COREFL task title, e.g. "13. Frog"
    "learner_id",       # str: writer ID derived from the COREFL filename, or text_id for AI text
    "l1",               # str: writer's first language, or None for AI text
    "gen_style",        # str: "native" or "learner", or None for human text
    "gen_level",        # str: CEFR level requested in the prompt; learner-style AI text only
    "gen_temperature",  # float: sampling temperature, or None for human text
    "gen_target_words", # int: requested length, or None for human text
    "gen_film",         # str: film named in the prompt; AI T3 text only
    "split",            # str: "train", "val", "test", "test_heldout_generator" (set by build_splits.py)
]

SPLITS = ["train", "val", "test", "test_heldout_generator"]

# ---------------------------------------------------------------------------
# Model output contract
#   predict_probs(texts: list[str]) -> np.ndarray of shape (n, NUM_LABELS),
#   each row sums to 1, columns ordered as LABELS.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str


class ChunkResult(BaseModel):
    text: str
    start_idx: int         # index of the chunk's first sentence
    end_idx: int           # index of the chunk's last sentence (inclusive)
    p_ai_raw: float        # softmax prob of "ai" class
    p_ai_calibrated: float # after CEFR calibration
    flagged: bool
    confidence: float


class DetectResponse(BaseModel):
    estimated_cefr: str
    overall_ai_pct: float
    overall_confidence: float
    chunks: list[ChunkResult]
