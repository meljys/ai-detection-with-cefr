"""
EDA for the processed dataset.

Run from the repo root (after build_splits.py): python -m data.eda
Input:  data/processed/*.parquet (all splits)
Output: reports/eda/  (tables as CSV, figures as PNG)

notebooks/eda.ipynb imports these functions to show the same tables and
figures with commentary; this script is the reproducible source of the report outputs.
"""
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from common.schema import SPLITS
from data.corefl_loader import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR.parent / "reports" / "eda"

CLASS_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "native", "ai"]
HUMAN_ORDER = CLASS_ORDER[:-1]
TASKS = {"T3": "T3 Film", "T13": "T13 Frog", "T14": "T14 Chaplin"}
TTR_WORDS = 50 # type-token ratio on the first 50 words, so text length doesn't bias it

# Colours
HUMAN = "steelblue"
AI = "maroon"
TASK_COLOURS = {"T3": "orange", "T13": "seagreen", "T14": "palevioletred"}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(PROCESSED_DIR / f"{s}.parquet") for s in SPLITS],
                   ignore_index=True)
    df["class"] = df["cefr"].fillna("ai")
    df["words"] = df["text"].str.split().str.len()
    df["ttr"] = df["text"].map(type_token_ratio)
    # Level an AI text imitates: the requested CEFR level, or "native" for native-style prompts
    df["imitates"] = df["gen_level"].where(df["gen_style"] == "learner", df["gen_style"])
    return df


def type_token_ratio(text: str) -> float | None:
    """Unique words / total words over the first TTR_WORDS words; None if the text is shorter."""
    tokens = re.findall(r"[a-z']+", text.lower())
    if len(tokens) < TTR_WORDS:
        return None
    first = tokens[:TTR_WORDS]
    return len(set(first)) / len(first)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All EDA tables, keyed by the CSV name they are saved under."""
    human = df[~df["is_ai"]]
    ai = df[df["is_ai"]]

    counts = pd.crosstab(df["class"], df["split"], margins=True, margins_name="Total")
    counts = counts.reindex(CLASS_ORDER + ["Total"])[SPLITS + ["Total"]]

    task = pd.crosstab(df["prompt_id"], df["class"]).reindex(columns=CLASS_ORDER)

    l1 = pd.crosstab(human["l1"], human["class"], margins=True, margins_name="Total")
    l1 = l1.reindex(columns=HUMAN_ORDER + ["Total"]).sort_values("Total", ascending=False)

    length =df.groupby("class")["words"].describe()[["count", "min", "25%", "50%", "75%", "max"]]

    src = df.groupby(["source_model", "prompt_id"])["words"].median().unstack()

    ttr = df.groupby("class")["ttr"].agg(["count", "median", "mean"])

    # Real writers at each level next to AI texts prompted to imitate that level
    stats = lambda g: g.agg(n=("text", "size"), median_words=("words", "median"),
                            median_ttr=("ttr", "median"))
    ai_style = pd.concat({"human": stats(human.groupby("class")),
                          "ai_imitating": stats(ai.groupby("imitates"))}, axis=1)
    ai_style = ai_style.reindex(HUMAN_ORDER).astype({("ai_imitating", "n"): "Int64"}) # A1/C2 have no AI

    return {
        "class_counts_by_split": counts,
        "task_by_class": task,
        "l1_by_class": l1,
        "words_by_class": length.reindex(CLASS_ORDER).round(1),
        "median_words_by_source_task": src,
        "ttr_by_class": ttr.reindex(CLASS_ORDER).round(3),
        "ai_style_vs_human": ai_style.round(3),
    }


def save_tables(tabs: dict[str, pd.DataFrame]) -> None:
    for name, table in tabs.items():
        table.to_csv(OUT_DIR / f"{name}.csv")

    print("Texts per class and split:\n", tabs["class_counts_by_split"], "\n")
    print("Median words per class:\n", tabs["words_by_class"]["50%"], "\n")
    print(f"Median TTR (first {TTR_WORDS} words) per class:\n", tabs["ttr_by_class"]["median"], "\n")
    print("Human writers vs AI imitating the same level:\n", tabs["ai_style_vs_human"])


# ---------------------------------------------------------------------------
# Figures (each saves a PNG to OUT_DIR and returns the figure for the notebook)
# ---------------------------------------------------------------------------
def _style(ax, title: str, ylabel: str, xlabel: str = "Class", grid_axis: str = "y") -> None:
    ax.set_title(title, fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _new_fig(figsize: tuple, **kw):
    fig, ax = plt.subplots(figsize=figsize, dpi=150, **kw)
    return fig, ax


def _save(fig, fname: str):
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, bbox_inches="tight")
    return fig


def _human_ai_legend(ax) -> None:
    ax.legend(handles=[Patch(facecolor=HUMAN, alpha=0.75, label="Human"),
                       Patch(facecolor=AI, alpha=0.75, label="AI")],
              frameon=False, loc="upper left", bbox_to_anchor=(1, 1), labelcolor=INK_MUTED)


def _boxes(ax, data: list, positions: list, colours: list, width: float):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                    showfliers=False, medianprops={"color": INK, "linewidth": 1.5},
                    whiskerprops={"color": INK_MUTED}, capprops={"color": INK_MUTED})
    for patch, colour in zip(bp["boxes"], colours):
        patch.set(facecolor=colour, alpha=0.75, edgecolor=colour)


def plot_task_by_class(df: pd.DataFrame):
    """100% stacked bars: which writing task each class's texts come from."""
    share = pd.crosstab(df["class"], df["prompt_id"], normalize="index")
    share = share.reindex(index=CLASS_ORDER, columns=list(TASKS))
    fig, ax = _new_fig((8, 4.5))
    left = pd.Series(0.0, index=share.index)
    for task, name in TASKS.items():
        ax.barh(share.index, share[task], left=left, height=0.7, color=TASK_COLOURS[task],
                alpha=0.75, linewidth=2, label=name)
        for cls, p in share[task].items():
            if p >= 0.08: # label segments wide enough to hold the text
                ax.text(left[cls] + p / 2, cls, f"{p:.0%}", ha="center", va="center", color=INK)
        left += share[task]
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1), labelcolor=INK_MUTED)
    _style(ax, "Writing task by class", "Class", xlabel="Share of texts", grid_axis="x")
    return _save(fig, "task_by_class.png")


def plot_l1(df: pd.DataFrame):
    """Learner texts per first language (natives excluded: their L1 is English)."""
    counts = df.loc[(~df["is_ai"]) & (df["class"] != "native"), "l1"].value_counts()
    fig, ax = _new_fig((8, 4))
    ax.barh(counts.index, counts.values, height=0.65, color=HUMAN, alpha=0.75)
    total = counts.sum()
    for l1, n in counts.items():
        pct = f"{n / total:.0%}" if n / total >= 0.005 else "<1%"
        ax.annotate(f"{n:,} ({pct})", (n, l1), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK_MUTED)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.2)
    _style(ax, f"Learner texts by first language (n={total:,})", "", xlabel="Texts",
           grid_axis="x")
    return _save(fig, "l1_distribution.png")


def boxplot_by_class(df: pd.DataFrame, col: str, title: str, ylabel: str, fname: str,
                     ylim: tuple | None = None):
    """One box per class, human in one hue and AI in another."""
    data = [df.loc[df["class"] == c, col].dropna() for c in CLASS_ORDER]
    fig, ax = _new_fig((8, 4.5))
    _boxes(ax, data, list(range(1, len(CLASS_ORDER) + 1)),
           [AI if c == "ai" else HUMAN for c in CLASS_ORDER], 0.55)
    ax.set_xticks(range(1, len(CLASS_ORDER) + 1),
                  [f"{c}\nn={len(d)}" for c, d in zip(CLASS_ORDER, data)])
    _human_ai_legend(ax)
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax, title, ylabel)
    return _save(fig, fname)


def plot_length(df: pd.DataFrame):
    return boxplot_by_class(df, "words", "Text length by class", "Words per text",
                            "length_by_class.png")


def plot_ttr(df: pd.DataFrame):
    return boxplot_by_class(
        df, "ttr", f"Vocabulary variety by class (type-token ratio, first {TTR_WORDS} words)",
        "Type-token ratio", "ttr_by_class.png")


def paired_boxplot_by_level(df: pd.DataFrame, col: str, title: str, ylabel: str, fname: str):
    """Real writers at each level beside AI texts prompted to imitate that level."""
    human = df[~df["is_ai"]]
    ai = df[df["is_ai"]]
    fig, ax = _new_fig((8, 4.5))
    for i, level in enumerate(HUMAN_ORDER, start=1):
        h = human.loc[human["class"] == level, col].dropna()
        a = ai.loc[ai["imitates"] == level, col].dropna()
        if len(a): # paired boxes: human left, AI right
            _boxes(ax, [h, a], [i - 0.2, i + 0.2], [HUMAN, AI], 0.36)
        else: # A1 and C2 were never requested from the LLMs
            _boxes(ax, [h], [i], [HUMAN], 0.36)
    ax.set_xticks(range(1, len(HUMAN_ORDER) + 1), HUMAN_ORDER)
    _human_ai_legend(ax)
    _style(ax, title, ylabel,
           xlabel="CEFR level (for AI: level requested in the prompt; A1 and C2 not generated)")
    return _save(fig, fname)


def plot_ai_style_ttr(df: pd.DataFrame):
    return paired_boxplot_by_level(
        df, "ttr", "Vocabulary variety: human writing vs AI imitating the same level",
        f"Type-token ratio (first {TTR_WORDS} words)", "ai_style_ttr.png")


def plot_ai_style_length(df: pd.DataFrame):
    return paired_boxplot_by_level(
        df, "words", "Text length: human writing vs AI imitating the same level",
        "Words per text", "ai_style_length.png")


def make_figures(df: pd.DataFrame) -> None:
    plot_task_by_class(df)
    plot_l1(df)
    plot_length(df)
    plot_ttr(df)
    plot_ai_style_ttr(df)
    plot_ai_style_length(df)


def main() -> None:
    plt.switch_backend("Agg") # no windows when run as a script
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    save_tables(tables(df))
    make_figures(df)
    plt.close("all")
    print(f"\nSaved tables and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
