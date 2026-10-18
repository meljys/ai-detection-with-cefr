"""
Generate AI-written texts matched to the COREFL writing tasks (T3, T13, T14).

Run from the repo root. Each model has its own rate limit, so run them in
separate terminals at the same time:

    python -m data.generate_ai_text --models qwen gpt-oss gemini --limit 5 # quick test first
    python -m data.generate_ai_text --models qwen                          # held out: run first
    python -m data.generate_ai_text --models gpt-oss
    python -m data.generate_ai_text --models gemini
    python -m data.generate_ai_text --combine                              # when all are done

Every response is cached in data/cache/<model>/, so you can stop and re-run at
any time (e.g. after a daily limit) and it continues where it left off.
Output of --combine: data/interim/ai_texts.parquet
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from common.schema import AI_ID
from data.corefl_loader import DATA_DIR, INTERIM_DIR, _clean

ROOT = DATA_DIR.parent
CACHE_DIR = DATA_DIR / "cache"
load_dotenv(ROOT / ".env")

SEED = 2111

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
GENERATORS = {
    "qwen":    {"provider": "groq",   "model": "qwen/qwen3.8-27b",      "n": 135, "pause": 6.0}, # held-out set
    "gpt-oss": {"provider": "groq",   "model": "openai/gpt-oss-120b",   "n": 450, "pause": 6.0},
    "gemini":  {"provider": "gemini", "model": "gemini-3.5-flash-lite", "n": 450, "pause": 4.5},
}
HELD_OUT_MODEL = GENERATORS["qwen"]["model"] # used by build_splits.py

# ---------------------------------------------------------------------------
# Tasks (instructions from the COREFL task table - modified)
# ---------------------------------------------------------------------------
FILMS = [
    # From the data (human T3 texts)
    "Spider-Man: No Way Home", "Close (2022)", "Jaws", "Live Twice, Love Once (Vivir dos veces)",
    "Black Panther: Wakanda Forever", "The Great Gatsby (2013)",
    "The Lovely Bones", "Mia and the White Lion", "Harry Potter and the Philosopher's Stone",
    "Lost in Translation", "Shrek", "The Menu", "Ex Machina", "Wonder", "City of God",
    "Titanic", "The Godfather", "The Little Mermaid", "Smile", "King Richard",
    "While at War (Mientras dure la guerra)", "Frozen II", "Oppenheimer", "The Hunger Games",
    "We Live in Time", "The Avengers", "Elemental",
    # Added by us: generic prompts
    "a film of your own choice", "a foreign-language film of your own choice",
    "a film of your own choice; you do not remember its exact title",
    "a foreign-language film of your own choice; you do not remember its exact title",
]

FROG_STORY = (
    "The pictures are from the wordless picture book \"Frog, Where Are You?\" by "
    "Mercer Mayer. A boy has a pet frog in his room. One night, while the boy and "
    "his dog are sleeping, the frog escapes. In the morning, they look for it "
    "everywhere, first in the house and then outside in the forest. Different "
    "animals surprise or scare them while they search. In the end, a deer carries "
    "the boy and drops him and the dog into some water, and there they find the "
    "frog with a frog family. The boy takes one small frog home. "
    "Do not mention the book's title or author."
)

CHAPLIN_STORY = (
    "A baby has been left alone on a street. Charlie Chaplin is walking by and "
    "finds it. He does not want the baby and tries several times to leave it "
    "somewhere else, for example with a woman who has her own baby, but she gets "
    "angry, and the police keep watching him. In the end, he finds a note with the "
    "baby: \"Please love and care for this orphan child.\" He decides to take the "
    "baby home and look after it."
)

TASKS = {
    "T3": "Summarise a film you have seen recently. Write about: {film}.",
    "T13": ("You looked at a series of pictures that tell a story. "
            "Tell the story. You can add new aspects to the story or "
            "ignore some aspects in the pictures. Start with \"One day...\". "
            "Use your own words.\n\nThe pictures show this: " + FROG_STORY),
    "T14": ("You watched a Charlie Chaplin video. Summarise the story. "
            "Use your own words.\n\nThe video shows this: " + CHAPLIN_STORY),
}
TASK_TITLES = {"T3": "3. Film", "T13": "13. Frog", "T14": "14. Chaplin"}

# Writing styles, so the AI class is not all one voice. (name, weight, instruction)
STYLES = [
    ("native",  0.6, "You are a native English speaker writing a short text for a language study."),
    ("learner", 0.4, "You are a learner of English at CEFR level {level}. "
                     "Write the way a learner at that level typically writes, including "
                     "the grammar, spelling and word-choice mistakes typical of that level."),
]
LEARNER_LEVELS = ["A2", "B1", "B2", "C1"]


def build_prompt(task: str, style_text: str, n_words: int, film: str | None) -> str:
    return (f"{style_text}\n\nTask: {TASKS[task].format(film=film)}\n\n"
            f"Write about {n_words} words. Write only the text itself, as plain "
            "paragraphs: no title, no headings, no lists, no comments before or after.")


# ---------------------------------------------------------------------------
# Plan: which task / style / length / temperature each text gets
# ---------------------------------------------------------------------------
def make_plan(name: str, human: pd.DataFrame) -> list[dict]:
    """
    Fixed random plan per model (same seed -> same plan on every run).
    Tasks follow the human task proportions; target lengths are sampled
    from human lengths for that task, clipped to its 5th-95th percentile.
    """
    rng = np.random.default_rng(SEED + list(GENERATORS).index(name))
    n = GENERATORS[name]["n"]

    task_p = human["prompt_id"].value_counts(normalize=True)
    lengths = {t: human.loc[human["prompt_id"] == t, "text"].str.split().str.len().to_numpy()
               for t in task_p.index}
    bounds = {t: np.percentile(v, [5, 95]) for t, v in lengths.items()}
    weights = [w for _, w, _ in STYLES]

    plan = []
    for i in range(n):
        task = str(rng.choice(task_p.index, p=task_p.values))
        style_name, _, style_text = STYLES[rng.choice(len(STYLES), p=weights)]
        level = str(rng.choice(LEARNER_LEVELS)) if style_name == "learner" else None
        lo, hi = bounds[task]
        n_words = int(round(np.clip(rng.choice(lengths[task]), lo, hi), -1))
        film = str(rng.choice(FILMS)) if task == "T3" else None
        temperature = float(round(rng.uniform(0.7, 1.0), 2))

        plan.append({
            "text_id": f"AI_{name}_{i:04d}",
            "prompt_id": task,
            "style": style_name,
            "level": level,
            "film": film,
            "target_words": n_words,
            "temperature": temperature,
            "prompt": build_prompt(task, style_text.format(level=level), n_words, film),
        })
    return plan


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------
_clients = {}


def _client(provider: str):
    if provider not in _clients:
        if provider == "gemini":
            from google import genai
            _clients[provider] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        else:
            from groq import Groq
            _clients[provider] = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _clients[provider]


def call_model(cfg: dict, prompt: str, temperature: float, n_words: int) -> str:
    max_tokens = int(n_words * 2.5) + 200

    if cfg["provider"] == "gemini":
        from google.genai import types
        resp = _client("gemini").models.generate_content(
            model=cfg["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return resp.text or ""

    # Groq
    extra = {}
    if "gpt-oss" in cfg["model"]:
        extra = {"reasoning_effort": "low"} # keep reasoning short to save tokens
        max_tokens += 600 # reasoning tokens count towards the limit
    elif "qwen" in cfg["model"]:
        extra = {"reasoning_effort": "none"}
    resp = _client("groq").chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Cleaning model output
# ---------------------------------------------------------------------------
THINK_RE = re.compile(r"<think>.*?</think>", re.S) # remove reasoning if present


def clean_output(raw: str) -> str:
    """
    Remove thinking blocks, markdown and a leading title line,
    then normalise whitespace the same way as the human texts.
    """
    text = THINK_RE.sub("", raw)
    text = re.sub(r"[*#]+", "", text) # remove markdown symbols
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if (len(lines) > 1 and len(lines[0].split()) <= 10
            and not lines[0].endswith((".", "!", "?"))):
        lines = lines[1:] # drop a title line
    text = _clean(" ".join(lines))
    return text.strip("\"'\u201c\u201d ") # remove quote marks wrapped around the whole text


# ---------------------------------------------------------------------------
# Generation loop with retries and caching
# ---------------------------------------------------------------------------
class LimitReached(Exception):
    pass


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted"))


def generate_one(cfg: dict, item: dict, max_tries: int = 6):
    """
    Return (raw, text), or None if this item keeps failing.
    Raises LimitReached if the rate/daily limit doesn't clear after retries.
    """
    delay, rate_limited = 10, False
    for _ in range(max_tries):
        try:
            raw = call_model(cfg, item["prompt"], item["temperature"], item["target_words"])
            text = clean_output(raw)
            if len(text.split()) >= 0.5 * item["target_words"]:
                return raw, text
            print(f"  {item['text_id']}: reply too short ({len(text.split())} words), retrying")
            rate_limited = False # empty/too short: try again
        except Exception as e:
            rate_limited = _is_rate_limit(e)
            print(f"  {item['text_id']}: {'rate limit' if rate_limited else 'error'}: {str(e)[:200]}")
        time.sleep(delay)
        delay = min(delay * 2, 300)
    if rate_limited:
        raise LimitReached
    return None


def run(name: str, limit: int | None = None) -> None:
    cfg = GENERATORS[name]
    human = pd.read_parquet(INTERIM_DIR / "corefl_clean.parquet")
    plan = make_plan(name, human)
    if limit:
        plan = plan[:limit]

    out_dir = CACHE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in plan if not (out_dir / f"{it['text_id']}.json").exists()]
    print(f"{name}: {len(plan) - len(todo)} cached, {len(todo)} to generate")

    for k, item in enumerate(todo, 1):
        try:
            result = generate_one(cfg, item)
        except LimitReached:
            print("Rate/daily limit reached. Stopping - re-run later to resume.")
            break
        if result:
            raw, text = result
            record = {**item, "model": cfg["model"], "raw": raw, "text": text}
            (out_dir / f"{item['text_id']}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=1))
        if k % 10 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}")
        time.sleep(cfg["pause"])


# ---------------------------------------------------------------------------
# Combine cached results
# ---------------------------------------------------------------------------
def combine() -> pd.DataFrame:
    rows = []
    for name in GENERATORS:
        for f in sorted((CACHE_DIR / name).glob("*.json")):
            r = json.loads(f.read_text())
            rows.append({
                "text_id": r["text_id"],
                "text": r["text"],
                "label": AI_ID,
                "cefr": None,
                "is_ai": True,
                "source_model": r["model"],
                "prompt_id": r["prompt_id"],
                "task_title": TASK_TITLES[r["prompt_id"]],
                "learner_id": r["text_id"],
                "l1": None,
                "gen_style": r["style"],
                "gen_level": r["level"],
                "gen_temperature": r["temperature"],
                "gen_target_words": r["target_words"],
                "gen_film": r["film"],
            })

    df = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)
    INTERIM_DIR.mkdir(exist_ok=True)
    df.to_parquet(INTERIM_DIR / "ai_texts.parquet", index=False)

    n_words = df["text"].str.split().str.len()
    print(f"Saved {len(df)} AI texts")
    print("\nModel x task:\n", pd.crosstab(df["source_model"], df["prompt_id"]))
    print("\nWords per model:\n", df.assign(n=n_words).groupby("source_model")["n"]
          .describe()[["min", "50%", "max"]])
    frog = df[df["prompt_id"] == "T13"]
    if len(frog):
        print("\nT13 texts starting with 'One day':",
              f"{frog['text'].str.lower().str.startswith('one day').mean():.0%}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=list(GENERATORS))
    p.add_argument("--limit", type=int, help="only the first N texts (for testing)")
    p.add_argument("--combine", action="store_true")
    args = p.parse_args()

    if args.combine:
        combine()
    else:
        for m in args.models or []:
            run(m, args.limit)
