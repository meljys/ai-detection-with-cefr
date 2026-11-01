"""Streamlit frontend for the fair AI-text detector.

Run locally (with the API running on :8000):
    streamlit run frontend/app.py
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import requests
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from frontend.sentence_utils import highlight_flagged, split_sentences

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
USE_MOCK = os.environ.get("USE_MOCK", "0") == "1"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
MIN_TEXT_WORDS = int(os.environ.get("MIN_TEXT_WORDS", "20"))

st.set_page_config(
    page_title="Fair AI-Text Detector",
    page_icon="📝",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Force dark text on the yellow highlight, even in dark mode */
    .flagged {
        background-color: #ffe0a3;
        color: #0b0b0b !important;
        padding: 1px 2px;
        border-radius: 3px;
    }

    /* Card styling using Streamlit's theme variables */
    .chunk-card {
        border: 1px solid var(--secondary-background-color, #e4e3df);
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
        background-color: var(--secondary-background-color, #fcfcfb);
        color: var(--text-color, #0b0b0b) !important;
        white-space: pre-wrap;
        line-height: 1.5;
    }

    /* Distinctive left border for flagged chunks */
    .chunk-card.f {
        border-left: 5px solid #eb6834;
    }

    /* Ensure the chunk text follows the theme text color */
    .small-muted {
        opacity: 0.85;
        font-size: 14px;
        color: var(--text-color, #0b0b0b) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- API helpers ----------

def _health() -> dict | None:
    if USE_MOCK:
        return {"status": "ok", "model": "mock", "calibration": "mock (no backend)"}
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _mock_response(text: str) -> dict:
    """Deterministic mock for developing without the API."""
    sentences = split_sentences(text)
    chunks: list[dict] = []
    for i, s in enumerate(sentences):
        p = 0.05 + (abs(hash(s)) % 90) / 100.0
        chunks.append({
            "text": s,
            "start_idx": i,
            "end_idx": i,
            "p_ai_raw": round(p, 4),
            "p_ai_calibrated": round(p, 4),
            "flagged": p >= 0.5,
            "confidence": round(min(1.0, abs(p - 0.5) * 2 + 0.3), 4),
        })
    flagged_words = sum(len(c["text"].split()) for c in chunks if c["flagged"])
    total_words = sum(len(c["text"].split()) for c in chunks) or 1
    return {
        "estimated_cefr": "B2",
        "overall_ai_pct": round(100 * flagged_words / total_words, 2),
        "overall_confidence": 0.78,
        "chunks": chunks,
    }


def call_api_text(text: str) -> dict:
    r = requests.post(f"{API_URL}/detect/text", json={"text": text}, timeout=90)
    r.raise_for_status()
    return r.json()


def call_api_file(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes)}
    r = requests.post(f"{API_URL}/detect/file", files=files, timeout=180)
    r.raise_for_status()
    return r.json()


# ---------- Sidebar ----------

with st.sidebar:
    st.header("Detector Status")
    health = _health()
    if health:
        st.success(f"Backend: {health.get('model', 'stub')}")
        if health.get("calibration"):
            st.caption(f"Calibration: {health['calibration']}")
        thresholds = health.get("thresholds") or {}
        if thresholds:
            with st.expander("Per-level thresholds"):
                for lvl, t in thresholds.items():
                    st.write(f"- `{lvl}`: {t:.3f}")
    else:
        st.error(f"Backend unreachable at {API_URL}")
    st.markdown("---")
    st.caption(
        "This tool supports human review, not replacing it. "
        "Do not use it for high-stakes decisions."
    )


# ---------- Main ----------

st.title("Fair AI-Text Detector")
st.caption("CEFR-calibrated AI-likelihood scoring with sentence-level highlighting.")

input_mode = st.radio("Input", ["Text", "File"], horizontal=True, label_visibility="collapsed")

text = ""
file_bytes: bytes | None = None
filename: str | None = None

if input_mode == "Text":
    text = st.text_area(
        "Paste your text here",
        height=240,
        placeholder=f"Minimum {MIN_TEXT_WORDS} words…",
    )
else:
    upload = st.file_uploader(
        "Upload a .txt, .md, .docx or .pdf file",
        type=["txt", "md", "docx", "pdf"],
    )
    if upload is not None:
        file_bytes = upload.getvalue()
        filename = upload.name
        if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
            file_bytes = None
        else:
            st.caption(f"{filename} · {len(file_bytes) / 1024:.1f} KB")

detect = st.button("Detect", type="primary")

if detect:
    original_text = ""

    if input_mode == "Text":
        n_words = len(text.split())
        if n_words < MIN_TEXT_WORDS:
            st.warning(f"Text is too short ({n_words} words). Minimum is {MIN_TEXT_WORDS}.")
            st.stop()
        if USE_MOCK:
            response = _mock_response(text)
        else:
            with st.spinner("Analysing…"):
                try:
                    response = call_api_text(text)
                except requests.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
                    st.stop()
                except Exception as e:
                    st.error(f"Could not reach the backend: {e}")
                    st.stop()
        original_text = text
    else:
        if file_bytes is None:
            st.warning("Upload a file first.")
            st.stop()
        if USE_MOCK:
            mock_text = file_bytes.decode("utf-8", errors="ignore")
            response = _mock_response(mock_text)
            original_text = mock_text
        else:
            with st.spinner("Extracting and analysing…"):
                try:
                    response = call_api_file(file_bytes, filename)
                except requests.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
                    st.stop()
                except Exception as e:
                    st.error(f"Could not reach the backend: {e}")
                    st.stop()
            if filename and filename.lower().endswith((".txt", ".md")):
                original_text = file_bytes.decode("utf-8", errors="ignore")

    # --- Headline metrics ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall AI-likelihood", f"{response['overall_ai_pct']:.1f}%")
    with c2:
        st.metric("Confidence", f"{response['overall_confidence']:.2f}")
    with c3:
        st.metric("Estimated CEFR level", response["estimated_cefr"])

    n_flagged = sum(1 for c in response["chunks"] if c["flagged"])
    st.caption(
        f"{n_flagged} of {len(response['chunks'])} chunk(s) flagged as AI-likely."
    )

    # --- Highlighted text ---
    st.subheader("Highlighted text")
    st.caption(
        "Flagged sentences are shaded. Hover for p(AI) and confidence."
    )

    if original_text:
        # For TXT/MD: accurate sentence-level highlighting
        html_content = highlight_flagged(original_text, response)
        st.markdown(
            f'<div style="line-height:1.6; font-size:15px; white-space:pre-wrap;">'
            f"{html_content}</div>",
            unsafe_allow_html=True,
        )
    else:
        # For PDF/DOCX: reconstruct the view using the backend's chunks
        st.caption("Showing extracted chunks (original layout may differ for PDF/DOCX).")
        html_content = "".join(
            f"<p><mark class='flagged'>{html.escape(c['text'])}</mark></p>" if c['flagged']
            else f"<p>{html.escape(c['text'])}</p>"
            for c in response['chunks']
        )
        st.markdown(
            f'<div style="line-height:1.6; font-size:15px;">{html_content}</div>',
            unsafe_allow_html=True,
        )

    # --- Per-chunk detail ---
    st.subheader("Per-chunk detail")
    for i, c in enumerate(response["chunks"]):
        cls = "chunk-card f" if c["flagged"] else "chunk-card"
        badge = "🚩 " if c["flagged"] else ""
        st.markdown(
            f'<div class="{cls}"><b>Chunk {i + 1}</b> {badge}· '
            f'p(AI) raw={c["p_ai_raw"]:.3f} · '
            f'calibrated={c["p_ai_calibrated"]:.3f} · '
            f'confidence={c["confidence"]:.2f}<br>'
            f'<span class="small-muted">{html.escape(c["text"])}</span></div>',
            unsafe_allow_html=True,
        )

    with st.expander("Legend"):
        st.markdown(
            "- **Overall AI-likelihood**: share of words sitting in flagged chunks.\n"
            "- **Estimated CEFR level**: most likely human class (native / A1–C2).\n"
            "- **p(AI) raw**: softmax probability of the AI class.\n"
            "- **p(AI) calibrated**: after CEFR-aware threshold rescaling "
            "(0.5 = decision line).\n"
            "- **Confidence**: `0.5 × distance-from-threshold + 0.5 × (1 − normalised entropy)`.\n"
            "- **Flagged**: `p(AI) ≥ τ_level`."
        )
