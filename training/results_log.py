"""
Collect evaluation results into one Excel workbook for comparing experiments.

Each experiment adds rows (one per model) to every sheet; re-logging the same
experiment + model replaces its rows. Other rows and any notes you add by hand
are left alone.

    from training.results_log import log_experiment
    log_experiment("Exp 4: LoRA alpha = 2r", ["baseline", "full_ft", "lora_r8"])

Reads the files written by the notebook:
    models/<name>/train_config.json, experiments/training/runs/<name>_metrics.json,
    experiments/training/runs/eval_<name>_{test,test_heldout_generator}.json,
    experiments/training/runs/baseline_zero_shot_{test,test_heldout_generator}.json

Output: experiments/training/results1.xlsx
"""
import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from training.config import EXP_DIR, MODELS_DIR, RUNS_DIR

XLSX_PATH = EXP_DIR / "results1.xlsx"

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
HEADER_FILL = PatternFill("solid", start_color="D9E1F2")
THIN = Side(style="thin", color="BFBFBF")
PCT, DEC = "0.0%", "0.000"

KEY = ["Experiment", "Model"]

# Levels from lowest to highest, natives above C2; used for every level column and matrix.
CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]

# sheet name -> list of (header, number format, column width)
SHEETS = {
    "Summary": [
        ("Experiment", None, 34), ("Model", None, 12), ("Settings", None, 34),
        ("Val macro-F1", DEC, 13), ("Test macro-F1", DEC, 13),
        ("Test AUROC (AI vs human)", DEC, 14), ("Test AI recall", PCT, 12),
        ("Held-out AI recall (Qwen)", PCT, 14), ("CEFR exact", PCT, 10),
        ("CEFR within 1 level", PCT, 11), ("CEFR mean levels off", DEC, 11),
        ("CEFR macro-F1", DEC, 11), ("Trainable params", "#,##0", 13), ("Train time (s)", "0.0", 10),
        ("Peak GPU memory (GB)", "0.00", 11), ("Source", None, 40),
    ],
    "FPR by CEFR": (
        [("Experiment", None, 34), ("Model", None, 12)]
        + [(lvl, PCT, 9) for lvl in CEFR_ORDER]
        + [("Highest FPR", PCT, 11), ("FPR gap (max - min)", PCT, 13), ("Source", None, 40)]
    ),
    "Per-class F1": (
        [("Experiment", None, 34), ("Model", None, 12)]
        + [(lbl, DEC, 9) for lbl in CEFR_ORDER + ["ai"]]
        + [("Source", None, 40)]
    ),
}

# row key -> (sheet name, row/column labels, corner text). All on test.parquet.
MATRICES = {
    "confusion_matrix": ("Confusion (8 classes)", CEFR_ORDER + ["ai"], "true \\ pred"),
    "cefr_confusion_matrix": ("CEFR confusion (7x7)", CEFR_ORDER, "true \\ estimated"),
    "ai_confusion_matrix": ("AI confusion (2x2)", ["human", "ai"], "true \\ flagged"),
}

NOTES = [
    ("Sheet / column", "Meaning"),
    ("Summary", "One row per model per experiment. Test numbers are on test.parquet (546 human + 135 AI texts)."),
    ("Val macro-F1", "Macro-F1 over the 8 classes on val.parquet, at the best epoch. Use this to choose between settings."),
    ("Test macro-F1", "Same on test.parquet. Report it; do not use it to choose settings."),
    ("Test AUROC", "How well p(ai) separates AI from human texts (1.0 = perfect), regardless of threshold."),
    ("Test AI recall", "Share of AI texts whose top class is 'ai'."),
    ("Held-out AI recall", "Same, on test_heldout_generator.parquet: 135 texts from Qwen, a generator never seen in training."),
    ("CEFR exact / within 1", "CEFR estimate = most likely human class (ignoring 'ai'), on human test texts. Share at the right level / at most one level off (order A1 < ... < C2 < native)."),
    ("CEFR mean levels off", "Average number of levels between the estimate and the true level. Lower is better."),
    ("CEFR macro-F1", "F1 averaged over the 7 human classes, using the CEFR estimate on human test texts (unlike Test macro-F1, which covers all 8 classes)."),
    ("Trainable params / time", "Weights updated in training, and wall-clock training time (all epochs). Times depend on the GPU: Exp 1 ran on a Colab T4, later experiments on an RTX 3080."),
    ("Peak GPU memory", "Most GPU memory PyTorch allocated while training (torch.cuda.max_memory_allocated), in GB. Not recorded for Exp 1-2."),
    ("FPR by CEFR", "Share of human texts at each level whose top class is 'ai' (falsely flagged). Lower is better."),
    ("Highest FPR / gap", "Formulas over the level columns: worst level, and worst minus best level (fairness gap)."),
    ("Per-class F1", "F1 for each of the 8 classes on test.parquet."),
    ("Confusion (8 classes)", "All test texts. Rows = true class, columns = the most likely of all 8 classes. Ordered A1 ... C2, native, ai."),
    ("CEFR confusion (7x7)", "Human test texts only. Rows = true level, columns = estimated level (most likely of the 7 human classes, ignoring 'ai', as calibration does). Ordered A1 ... C2, native, so one-level mistakes sit next to the diagonal."),
    ("AI confusion (2x2)", "All test texts. Rows = true (human / ai), columns = flagged as AI when p(ai) > 0.5. Note: 'Test AI recall' and 'FPR by CEFR' use the most likely class instead, so they can differ slightly."),
    ("baseline", "The original 2-class model (ahmediqbal/ai-text-detector-model), flagged as AI when p(AI) > 0.5. It has no CEFR classes, so F1 is blank."),
    ("Source", "Where the numbers came from. Rows recovered from earlier printed output may be rounded."),
]


def _style_header(ws, row: int, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font, cell.fill = BOLD, HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = Border(bottom=THIN)


def _new_workbook() -> Workbook:
    wb = Workbook()
    wb.remove(wb.active)
    for name, cols in SHEETS.items():
        ws = wb.create_sheet(name)
        for i, (header, _, width) in enumerate(cols, start=1):
            ws.cell(row=1, column=i, value=header)
            ws.column_dimensions[get_column_letter(i)].width = width
        _style_header(ws, 1, len(cols))
        ws.freeze_panes = "C2"
    for sheet, _, _ in MATRICES.values():
        wb.create_sheet(sheet).column_dimensions["A"].width = 14
    ws = wb.create_sheet("Notes")
    for r, (a, b) in enumerate(NOTES, start=1):
        ws.cell(row=r, column=1, value=a).font = BOLD if r == 1 else FONT
        ws.cell(row=r, column=2, value=b).font = BOLD if r == 1 else FONT
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 110
    _style_header(ws, 1, 2)
    return wb


def _find_row(ws, experiment: str, model: str) -> int:
    """Row holding (experiment, model), or the first empty row."""
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, 1).value == experiment and ws.cell(r, 2).value == model:
            return r
    r = ws.max_row + 1
    while r > 2 and ws.cell(r - 1, 1).value is None:
        r -= 1
    return r


def _write_table_row(ws, cols, row: int, values: dict) -> None:
    headers = [h for h, _, _ in cols]
    for i, (header, fmt, _) in enumerate(cols, start=1):
        cell = ws.cell(row=row, column=i)
        if header == "Highest FPR" or header.startswith("FPR gap"):
            first = get_column_letter(headers.index(CEFR_ORDER[0]) + 1)
            last = get_column_letter(headers.index(CEFR_ORDER[-1]) + 1)
            rng = f"{first}{row}:{last}{row}"
            cell.value = (f'=IF(COUNT({rng})=0,"",MAX({rng}))' if header == "Highest FPR"
                          else f'=IF(COUNT({rng})=0,"",MAX({rng})-MIN({rng}))')
        else:
            cell.value = values.get(header)
        cell.font = FONT
        if fmt:
            cell.number_format = fmt


def _write_confusion(ws, experiment: str, model: str, cm, labels, corner: str) -> None:
    """Each matrix is a block: title row, header row, one row per label, blank row."""
    title = f"{experiment} | {model}"
    start = None
    for r in range(1, ws.max_row + 1):
        if ws.cell(r, 1).value == title:
            start = r
            break
    if start is None:
        start = 1 if ws.max_row == 1 and ws.cell(1, 1).value is None else ws.max_row + 2
    ws.cell(start, 1, title).font = BOLD
    ws.cell(start + 1, 1, corner).font = BOLD
    for j, lbl in enumerate(labels, start=2):
        ws.cell(start + 1, j, lbl).font = BOLD
        ws.column_dimensions[get_column_letter(j)].width = 8
    for i, lbl in enumerate(labels):
        ws.cell(start + 2 + i, 1, lbl).font = BOLD
        for j in range(len(labels)):
            c = ws.cell(start + 2 + i, j + 2, cm[i][j])
            c.font = BOLD if i == j else FONT


def _reorder(cm, from_labels, to_labels) -> list:
    """Rearrange a confusion matrix's rows and columns from one label order to another."""
    idx = [list(from_labels).index(lbl) for lbl in to_labels]
    return [[cm[i][j] for j in idx] for i in idx]


def _check_headers(wb, path: Path) -> None:
    """Refuse to write if the file's columns differ from SHEETS (values would land under wrong headings)."""
    for name, cols in SHEETS.items():
        expected = [h for h, _, _ in cols]
        if name not in wb.sheetnames:
            raise ValueError(f"{path.name} has no '{name}' sheet. Rename or delete the file to start a new one.")
        found = [c.value for c in wb[name][1]][:len(expected)]
        if found != expected:
            raise ValueError(
                f"The columns of sheet '{name}' in {path.name} don't match the current layout:\n"
                f"  file:     {found}\n  expected: {expected}\n"
                f"Nothing was written. Rename or delete the file and re-log your experiments "
                f"to start a new one with the current columns."
            )


def write_rows(experiment: str, rows: list[dict], path: Path = XLSX_PATH) -> Path:
    """
    Write one experiment's rows. Each row dict has keys "Model", the Summary
    column names, the CEFR levels under "fpr", the classes under "f1", and
    optionally "Source" and the MATRICES keys.
    """
    wb = load_workbook(path) if path.exists() else _new_workbook()
    _check_headers(wb, path)
    for row in rows:
        base = {"Experiment": experiment, "Model": row["Model"], "Source": row.get("Source")}
        sheet_values = {
            "Summary": {**base, **row},
            "FPR by CEFR": {**base, **(row.get("fpr") or {})},
            "Per-class F1": {**base, **(row.get("f1") or {})},
        }
        for name, cols in SHEETS.items():
            ws = wb[name]
            r = _find_row(ws, experiment, row["Model"])
            _write_table_row(ws, cols, r, sheet_values[name])
        for key, (sheet, labels, corner) in MATRICES.items():
            if row.get(key):
                cm = row[key]
                if key == "confusion_matrix" and row.get("confusion_labels", labels) != labels:
                    cm = _reorder(cm, row["confusion_labels"], labels)  # results saved in the old order
                ws = wb[sheet] if sheet in wb.sheetnames else wb.create_sheet(sheet, wb.sheetnames.index("Notes"))
                _write_confusion(ws, experiment, row["Model"], cm, labels, corner)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _settings(name: str) -> str:
    cfg = _read(MODELS_DIR / name / "train_config.json") or {}
    metrics = _read(RUNS_DIR / f"{name}_metrics.json") or {}
    if not cfg:
        return ""
    method = f"LoRA r={cfg['lora_r']}, alpha {cfg['lora_alpha']}" if cfg["use_lora"] else "Full FT"
    best = metrics.get("best_epoch")
    epochs = f"{cfg['epochs']} ep" + (f" (best {best})" if best else "")
    ordinal = f", ordinal {cfg['ordinal_smoothing']:g}" if cfg.get("ordinal_smoothing") else ""
    seed = f", seed {cfg['seed']}" if cfg.get("seed", 2111) != 2111 else ""
    return f"{method}, lr {cfg['lr']:g}, {epochs}{ordinal}{seed}"


def collect_rows(model_names: list[str]) -> list[dict]:
    """Build rows from the result files on disk ("baseline" = original 2-class model)."""
    rows = []
    for name in model_names:
        if name == "baseline":
            test = _read(RUNS_DIR / "baseline_zero_shot_test.json")
            held = _read(RUNS_DIR / "baseline_zero_shot_test_heldout_generator.json")
            if not test:
                raise FileNotFoundError("No baseline results. Run the evaluation cell first.")
            rows.append({
                "Model": "baseline",
                "Settings": "Original 2-class model, no fine-tuning, p(AI) > 0.5",
                "Test AI recall": test["ai_recall"],
                "Held-out AI recall (Qwen)": held["ai_recall"] if held else None,
                "fpr": test["fpr_by_cefr"],
                "ai_confusion_matrix": test.get("ai_confusion_matrix"),
                "Source": "baseline_zero_shot_*.json",
            })
            continue
        test = _read(RUNS_DIR / f"eval_{name}_test.json")
        held = _read(RUNS_DIR / f"eval_{name}_test_heldout_generator.json")
        metrics = _read(RUNS_DIR / f"{name}_metrics.json") or {}
        if not test:
            raise FileNotFoundError(f"No eval results for {name}. Run the evaluation cell first.")
        rows.append({
            "Model": name,
            "Settings": _settings(name),
            "Val macro-F1": metrics.get("val_macro_f1"),
            "Test macro-F1": test["macro_f1"],
            "Test AUROC (AI vs human)": metrics.get("test_ai_auroc"),
            "Test AI recall": test["ai_recall_argmax"],
            "Held-out AI recall (Qwen)": held["ai_recall_argmax"] if held else None,
            "CEFR exact": test.get("cefr_exact"),
            "CEFR within 1 level": test.get("cefr_within1"),
            "CEFR mean levels off": test.get("cefr_mae"),
            "Trainable params": metrics.get("trainable_params"),
            "Train time (s)": metrics.get("train_time_s"),
            "Peak GPU memory (GB)": metrics.get("peak_gpu_mem_gb"),
            "fpr": test["fpr_by_cefr_argmax"],
            "f1": test["per_class_f1"],
            "CEFR macro-F1": test.get("cefr_macro_f1"),
            "confusion_matrix": test["confusion_matrix"],
            "confusion_labels": test.get("labels_order"),
            "cefr_confusion_matrix": test.get("cefr_confusion_matrix"),
            "ai_confusion_matrix": test.get("ai_confusion_matrix"),
            "Source": f"{name}_metrics.json, eval_{name}_*.json",
        })
    return rows


def log_experiment(experiment: str, model_names: list[str], path: Path = XLSX_PATH) -> Path:
    """Add (or replace) this experiment's rows in results1.xlsx from the files on disk."""
    out = write_rows(experiment, collect_rows(model_names), path)
    print(f"Logged {len(model_names)} models as '{experiment}' -> {out}")
    return out
