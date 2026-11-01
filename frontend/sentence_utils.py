"""Replicate the backend chunker's split_sentences so the frontend can map
chunk start_idx/end_idx back to sentence positions for inline highlighting.

If the local split differs from the API's chunk count (e.g. version skew),
fall back to rendering chunks as blocks.
"""
from __future__ import annotations

import html
import re

import pysbd

DEFAULT_MAX_WORDS = 150
ABS_MAX_WORDS = 300

_QUOTE_TRANS = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'",
})


def _normalise(s: str) -> str:
    s = s.translate(_QUOTE_TRANS)
    return re.sub(r"\s+", " ", s).strip()


def _split_long_sentence(sentence: str, piece_words: int = DEFAULT_MAX_WORDS) -> list[str]:
    """Same as backend.chunker._split_long_sentence."""
    words = sentence.split()
    if len(words) <= ABS_MAX_WORDS:
        return [sentence]
    pieces: list[str] = []
    while len(words) > piece_words:
        n_pieces = -(-len(words) // piece_words)
        target = -(-len(words) // n_pieces)
        cut = target
        for i in range(target, (target * 3) // 4, -1):
            if words[i - 1][-1] in ",;:":
                cut = i
                break
        pieces.append(" ".join(words[:cut]))
        words = words[cut:]
    if words:
        pieces.append(" ".join(words))
    return pieces


def split_paragraphs(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences_in_paragraph(para: str, piece_words: int = DEFAULT_MAX_WORDS) -> list[str]:
    seg = pysbd.Segmenter(language="en", clean=False)
    out: list[str] = []
    for p in seg.segment(para):
        p = _normalise(p)
        if p:
            out.extend(_split_long_sentence(p, piece_words))
    return out


def split_sentences(text: str, piece_words: int = DEFAULT_MAX_WORDS) -> list[str]:
    out: list[str] = []
    for para in split_paragraphs(text):
        out.extend(split_sentences_in_paragraph(para, piece_words))
    return out


def highlight_flagged(raw_text: str, response: dict) -> str:
    """HTML of the raw text with flagged sentences wrapped in <mark>.

    Falls back to per-chunk paragraphs if the local sentence split does not
    match the API's chunk indices.
    """
    chunks = response.get("chunks", [])
    if not chunks:
        return html.escape(raw_text)

    sentences = split_sentences(raw_text)
    expected_total = chunks[-1]["end_idx"] + 1
    if len(sentences) != expected_total:
        return _chunk_blocks(chunks)

    # sentence index -> (flagged, p_ai_calibrated, confidence)
    per_sentence: dict[int, tuple[bool, float, float]] = {}
    for chunk in chunks:
        for i in range(chunk["start_idx"], chunk["end_idx"] + 1):
            per_sentence[i] = (
                bool(chunk["flagged"]),
                float(chunk["p_ai_calibrated"]),
                float(chunk["confidence"]),
            )

    rendered_paragraphs: list[str] = []
    idx = 0
    for para in split_paragraphs(raw_text):
        rendered: list[str] = []
        for s in split_sentences_in_paragraph(para):
            flagged, p, c = per_sentence.get(idx, (False, 0.0, 0.0))
            esc = html.escape(s)
            if flagged:
                rendered.append(
                    f'<mark class="flagged" title="p(AI)={p:.3f}  confidence={c:.2f}">{esc}</mark>'
                )
            else:
                rendered.append(esc)
            idx += 1
        rendered_paragraphs.append(" ".join(rendered))
    return "\n\n".join(rendered_paragraphs)


def _chunk_blocks(chunks: list[dict]) -> str:
    """Fallback: show chunks as paragraphs, flagged ones highlighted."""
    parts: list[str] = []
    for c in chunks:
        esc = html.escape(c["text"])
        if c["flagged"]:
            parts.append(f'<p><mark class="flagged">{esc}</mark></p>')
        else:
            parts.append(f"<p>{esc}</p>")
    return "\n".join(parts)
