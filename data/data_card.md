# Data card

Dataset for CEFR-aware AI-text detection: human learner and native English writing (COREFL) plus AI-generated texts on the same tasks.

## 1. Data files

| File | Contents |
|---|---|
| `data/processed/train.parquet` | ~70% of human texts and of AI texts from Gemini and gpt-oss |
| `data/processed/val.parquet` | ~15% (same sources) |
| `data/processed/test.parquet` | ~15% (same sources) |
| `data/processed/test_heldout_generator.parquet` | All AI texts from Qwen (held-out generator), test only |

Raw COREFL files (`data/raw/`), intermediate files (`data/interim/`) and the AI generation cache (`data/cache/`) are not committed. See **8. Reproducing** below.

## 2. Scripts

| Script | Purpose |
|---|---|
| `common/schema.py` | Shared labels and column definitions |
| `data/corefl_loader.py` | Loads and cleans COREFL -> `data/interim/corefl_clean.parquet` |
| `data/generate_ai_text.py` | Generates AI texts -> `data/interim/ai_texts.parquet` |
| `data/build_splits.py` | Combines human + AI texts and builds the splits -> `data/processed/` |

## 3. Columns

Defined in `common/schema.py` (`DATASET_COLUMNS`).

| Column | Type | Description |
|---|---|---|
| `text_id` | str | Unique per text: COREFL filename for human texts, `AI_<model>_<nnnn>` for AI texts |
| `text` | str | Cleaned text (whitespace normalised only; spelling, grammar and punctuation unchanged) |
| `label` | int | Index into `LABELS`: native=0, A1–C2=1–6, ai=7 |
| `cefr` | str | `"native"`, `"A1"`–`"C2"`, or `None` for AI text |
| `is_ai` | bool | True for AI-generated text |
| `source_model` | str | LLM name for AI text, `"human"` otherwise |
| `prompt_id` | str | COREFL writing task the text responds to: `T3` (Film), `T13` (Frog), `T14` (Chaplin) |
| `task_title` | str | COREFL task title, e.g. `"13. Frog"` |
| `learner_id` | str | Writer ID derived from the COREFL filename (see **4. Human data (COREFL)** below), or `text_id` for AI text |
| `l1` | str | Writer's first language, or `None` for AI text |
| `gen_style` | str | `"native"` or `"learner"`, or `None` for human text |
| `gen_level` | str | CEFR level **requested in the prompt** (not an assessed level); learner-style AI text only |
| `gen_temperature` | float | Sampling temperature, or `None` for human text |
| `gen_target_words` | int | Requested length in words, or `None` for human text |
| `gen_film` | str | Film named in the prompt; AI T3 text only |
| `split` | str | `train`, `val`, `test`, `test_heldout_generator` (set by `build_splits.py`) |

## 4. Human data (COREFL)

**Source:** COREFL (Corpus of English as a Foreign Language), written subcorpus, learners and English natives, downloaded October 2026 from https://corefl.learnercorpora.com/.

**Citation:** C. Lozano, A. Díaz-Negrillo, and M. Callies, "Designing and compiling a learner corpus of written and spoken narratives: COREFL," in *What's in a Narrative? Variation in Story-Telling at the Interface Between Language and Literacy*, C. Bongartz and J. Torregrossa, Eds. Bern, Switzerland: Peter Lang, 2020, pp. 21–46, doi: 10.3726/978-3-653-05182-7.

**Processing** (`data/corefl_loader.py`):
- Kept written texts only.
- CEFR level taken from the `Proficiency` column (placement-test based); natives labelled `native`.
- Whitespace normalised (non-breaking spaces, byte-order marks, repeated spaces and line breaks). Learner errors left unchanged.
- Removed empty texts and exact duplicate texts.
- No minimum length applied (the shortest texts, ~9 words, are genuine beginner attempts).
- Dropped task T2 (Famous Person): only 17 texts in total.
- **Writer IDs:** COREFL has no participant ID column. Filenames follow `L1_medium_level_age_LoI_task_initials` (learners) or `L1_medium_age_task_initials` (natives) (COREFL User Guide: Metadata, https://corefl.learnercorpora.com/user_guide/metadata). Removing the task number gives one ID per writer, so a writer's texts on different tasks share an ID. 29 writers have more than one text.

**Counts:** 3,657 rows loaded -> **3,638 kept**.

| Task | Texts | Share | Median words |
|---|---|---|---|
| T3 Film | 1,487 | 40.9% | 114 |
| T13 Frog | 852 | 23.4% | 142 |
| T14 Chaplin | 1,299 | 35.7% | 193 |

| Class | Texts |
|---|---|
| A1 | 263 |
| A2 | 563 |
| B1 | 714 |
| B2 | 922 |
| C1 | 711 |
| C2 | 278 |
| native | 187 |
| **Total** | **3,638** |

Learner L1: Spanish (~75%), Italian, German, Turkish, Chinese, Czech, Greek, Estonian, French. Natives: English (mostly American English).

## 5. AI-generated data

**Script:** `data/generate_ai_text.py`. All responses are cached as JSON.

**Generators** (free tiers, October 2026):

| Model | Provider | Texts | Role |
|---|---|---|---|
| Qwen 3.8 27B | Groq | 135 | **Held out** (test only) |
| gpt-oss-120b | Groq | 450 | Training |
| Gemini 3.5 Flash Lite | Google AI Studio | 450 | Training |

**Sizes:** 900 texts from the two training models, chosen to approximate the largest human class (B2, 922). 135 held-out texts, matching the number of AI texts in the main test set (15% of 900).

**Prompt design.** Each text has a fixed, seeded plan:
- **Task:** sampled in the human task proportions.
- **Style:** native English writer (60%) or learner of English at A2, B1, B2 or C1 (40%, levels equally likely). Learner-style prompts ask for the grammar, spelling and word-choice mistakes typical of the level. A1 was excluded (LLM imitations of beginner writing are unrealistic), as was C2 (it overlaps with the native style). Native style is the majority to reflect typical AI output; learner style tests whether simple, error-containing AI text evades CEFR-calibrated thresholds.
- **Length:** sampled from human word counts for the same task, clipped to that task's 5th–95th percentile.
- **Temperature:** uniform 0.7–1.0.
- **Film (T3):** one of 27 titles named in a sample of ~40 human T3 texts, or one of 4 generic prompts (e.g. "a film of your own choice; you do not remember its exact title").
- **Frog (T13) and Chaplin (T14):** the models cannot see the pictures or video, so prompts include a short plain-language summary written by the team (with AI assistance). T13 prompts ask texts to start with "One day...", as in the original task.

**Prompt-echo fix:** initial outputs copied distinctive vocabulary from the story summaries and prompts (e.g. "tramp" in 92% of AI Chaplin texts against 1% of human texts). Therefore, the summaries were rewritten in simpler, more general language, and all T13 and T14 texts were regenerated (see **7. Limitations** below).

**Output cleaning:** removal of reasoning blocks, markdown, leading title lines and wrapping quotes; same whitespace normalisation as human texts. Replies shorter than half the target length were regenerated.

**Task mix and length.** AI task proportions roughly match the human data; held-out Qwen varies slightly due to its smaller sample. Median words per text:

| Source | T3 Film | T13 Frog | T14 Chaplin |
|---|---|---|---|
| Human | 114 | 142 | 193 |
| Gemini 3.5 Flash Lite | 128 | 186 | 167 |
| gpt-oss-120b | 124 | 154 | 184 |
| Qwen 3.8 27B (held out) | 135 | 151 | 261 |

## 6. Splits

**Script:** `data/build_splits.py`.

- All held-out generator (Qwen) texts -> `test_heldout_generator` only.
- All other texts -> ~70/15/15 train/val/test with scikit-learn `StratifiedGroupKFold` (20 folds: 14 train, 3 val, 3 test), **stratified by class** and **grouped by writer**, so no writer appears in more than one split. Fixed seed (2111).

| Class | train | val | test | test_heldout_generator | Total |
|---|---|---|---|---|---|
| native | 132 | 28 | 27 | 0 | 187 |
| A1 | 183 | 40 | 40 | 0 | 263 |
| A2 | 395 | 84 | 84 | 0 | 563 |
| B1 | 498 | 108 | 108 | 0 | 714 |
| B2 | 646 | 138 | 138 | 0 | 922 |
| C1 | 498 | 106 | 107 | 0 | 711 |
| C2 | 194 | 42 | 42 | 0 | 278 |
| ai | 630 | 135 | 135 | 135 | 1,035 |
| **Total** | **3,176** | **681** | **681** | **135** | **4,673** |

Shares of the main data (excluding the held-out set): train 0.70, val 0.15, test 0.15.

## 7. Limitations

**Human data**
- Learner L1 is ~75% Spanish; results may not generalise to other L1s.
- Natives are the smallest class (187 texts); consider class weights in training.
- Task and level are linked (A1 is mostly Frog; B2/C1 mostly Film and Chaplin), so a model may partly learn task as a proxy for level.
- Writer IDs are derived from filenames, not true participant IDs.

**AI data**
- Frog and Chaplin texts were generated from written summaries, not the original pictures or video.
- AI texts were initially found to echo distinctive vocabulary from the prompts. After the summaries were rewritten, some vocabulary differences remain, partly because the original book and film are likely in the models' training data, e.g. "owl" appears in 41% of AI Frog texts against 1% of human texts, and "tramp" in 25% of AI Chaplin texts against 1%.
- Learner-style AI texts approximate CEFR levels but contain fewer and more uniform errors than real learner writing.
- Only two training generators from two companies; detection of other generators is tested on one held-out model only.
- Free-tier models and versions available in October 2026; outputs may not be reproducible exactly if models change.

## 8. Reproducing

From the repo root, with `GEMINI_API_KEY` and `GROQ_API_KEY` in `.env` and the COREFL files in `data/raw/`:

```
python -m data.corefl_loader                     # -> data/interim/corefl_clean.parquet
python -m data.generate_ai_text --models qwen
python -m data.generate_ai_text --models gpt-oss
python -m data.generate_ai_text --models gemini
python -m data.generate_ai_text --combine        # -> data/interim/ai_texts.parquet
python -m data.build_splits                      # -> data/processed/*.parquet
```

Generation resumes from the cache after interruptions or rate limits.
