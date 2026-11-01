# Fair AI-Text Detector: Reducing Non-Native Bias via CEFR Calibration

**DSA2111 Group Project (Group 13)**

## Overview
AI-text detectors are widely used in educational and professional settings, but recent studies show they disproportionately misclassify writing by non-native English speakers as AI-generated. This is because lower-proficiency writing shares statistical properties (e.g., lower lexical diversity, predictable phrasing) with AI-generated text.

This project introduces a **two-stage system** to mitigate this bias:
1. **Proficiency Estimation:** Predicts the writer's CEFR level (A1–C2 or Native) using an 8-class DistilBERT model.
2. **Threshold Calibration:** Adjusts the AI-detection threshold based on the estimated CEFR level, ensuring a fair false-positive rate across all proficiency levels without sacrificing the detection of genuine AI text.

## Key Features
- **8-Class Classification:** Replaces the base binary head with a custom head (`native`, `A1`-`C2`, `ai`).
- **CEFR-Aware Calibration:** Dynamically adjusts the flagging threshold per CEFR level.
- **Interpretable Output:** Provides sentence-level highlighting, raw vs. calibrated AI-likelihood, and a confidence score.
- **Document Extraction:** Supports `.txt`, `.md`, `.docx`, and `.pdf` (with OCR fallback for scanned documents).
- **Containerized Deployment:** Fully dockerized backend (FastAPI) and frontend (Streamlit) for reproducible deployment.

## System Architecture
1. **Frontend (Streamlit):** Accepts text/file input, displays results, and maps flagged chunks back to the original text.
2. **Backend (FastAPI):** Handles file extraction (Tesseract OCR), text chunking, and model inference.
3. **Model (PyTorch/Hugging Face):** A DistilBERT model fine-tuned with LoRA (Low-Rank Adaptation) for efficient 8-class classification.
4. **Calibration Engine:** Computes per-level thresholds on validation data using a target False Positive Rate (FPR).

## Quick Start (Docker)
The easiest way to run the full application (Frontend + Backend + Model) is via Docker.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Run the application
```bash
# Build and start the containers
docker compose up --build
```

- **Frontend UI:** Open [http://localhost:8501](http://localhost:8501)
- **Backend API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs)

*(Note: The first build may take 5–10 minutes to download PyTorch and system dependencies. Subsequent runs are fast.)*

## Local Development Setup
If you want to run the code without Docker (e.g., for training or debugging):

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Backend
```bash
# Run with the fine-tuned model (requires model weights)
export DETECTOR_MODE="hf"
export MODEL_CKPT="models/e3_lora_r8_lr5e-4"
export THRESHOLDS_PATH="experiments/calibration/thresholds.json"
python -m uvicorn backend.api:app --port 8000
```
*(Tip: Use `export DETECTOR_MODE="stub"` for frontend development without loading the heavy model).*

### 3. Start the Frontend
Open a new terminal, activate the virtual environment, and run:
```bash
export API_URL="http://localhost:8000"
python -m streamlit run frontend/app.py
```

## Repository Structure
```text
├── common/            # Shared schemas, labels, and API data models
├── data/              # Data processing scripts (COREFL loader, AI generation)
├── training/          # Model fine-tuning, LoRA setup, and evaluation
├── calibration/       # CEFR estimator, threshold fitting, fairness metrics
├── backend/           # FastAPI service, text extractors, chunker, inference
├── frontend/          # Streamlit application and HTML highlighting logic
├── experiments/       # Calibration results, thresholds, and ablation scripts
├── docker/            # Dockerfiles for API and UI
└── docker-compose.yml # Multi-container orchestration
```

## Model & Data Details
- **Base Model:** `ahmediqbal/ai-text-detector-model` (DistilBERT, 67M params).
- **Training Data:** COREFL (human learners A1–C2 + natives) matched with AI-generated texts from Gemini, GPT-OSS, and Qwen.
- **Fine-tuning:** LoRA (rank 8) targeting `q_lin`, `k_lin`, `v_lin`, with the new classifier head trained from scratch.
- **Ablations:** We evaluated global vs. per-level thresholds, true vs. estimated CEFR, and different target FPRs (1%, 5%, 10%).
