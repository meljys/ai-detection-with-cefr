'''
FastAPI service for the detector

Run locally (from repo root):
    DETECTOR_MODE=stub uvicorn backend.api:app --reload --port 8000

Environment variables
    DETECTOR_MODE      stub | hf                      (default: stub)
    MODEL_CKPT         path to fine-tuned checkpoint  (required for hf)
    THRESHOLDS_PATH    path to per-level thresholds JSON from calibration
    MAX_UPLOAD_MB      upload size limit              (default: 10)
    MAX_TEXT_CHARS     pasted/extracted text limit    (default: 50000)
    MIN_TEXT_WORDS     minimum words to analyse       (default: 20)
    CORS_ORIGINS       comma-separated, e.g. http://localhost:8501 (default: *)
'''

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.extractors import SUPPORTED_EXTENSIONS, ExtractionError, extract_text
from backend.inference_service import Detector
from common.schema import DetectResponse, TextRequest

logging.basicConfig(
    level = os.environ.get('LOG_LEVEL', 'INFO'),
    format = '%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(float(os.environ.get('MAX_UPLOAD_MB', '10')) * 1024 * 1024)
MAX_TEXT_CHARS =   int(os.environ.get('MAX_TEXT_CHARS', '50000'))
MIN_TEXT_WORDS =   int(os.environ.get('MIN_TEXT_WORDS', '20'))

# -------------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup
    app.state.detector = Detector(
        ckpt = os.environ.get('MODEL_CKPT'),
        thresholds_path = os.environ.get('THRESHOLDS_PATH'),
        mode = os.environ.get('DETECTOR_MODE', 'stub'),
    )
    yield
    app.state.detector = None

app = FastAPI(title = 'Fair AI-Text Detector', version = '1.0.0', lifespan = lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins = [o.strip() for o in os.environ.get('CORS_ORIGINS', '*').split(',')],
    allow_methods = ['GET', 'POST'],
    allow_headers = ['*'],
)

# -------------------------------------------------------------------------
# Request logging
# -------------------------------------------------------------------------

@app.middleware('http')
async def log_requests(request: Request, call_next):
    rid = request.headers.get('x-request-id') or uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception('[%s] %s %s crashed', rid, request.method, request.url.path)
        return JSONResponse({'detail': 'Internal server error', 'request_id': rid}, status_code = 500)

    # Never log submitted text: essays are personal data
    ms = (time.perf_counter() - start) * 1000
    logger.info('[%s] %s %s -> %d (%.0f ms)', rid, request.method, request.url.path,
                response.status_code, ms)
    response.headers['x-request-id'] = rid
    return response

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _get_detector(request: Request) -> Detector:
    det = getattr(request.app.state, 'detector', None)
    if det is None:
        raise HTTPException(503, 'Model is not loaded yet')
    return det

def _validate_text(text: str) -> str:
    '''
    Checks that text is non-empty and within the length limits.
    '''

    text = (text or '').strip()
    if not text:
        raise HTTPException(400, 'Text is empty')

    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f'Text is {len(text):,} characters; the limit is {MAX_TEXT_CHARS:,}')

    n_words = len(text.split())
    if n_words < MIN_TEXT_WORDS:
        raise HTTPException(
            400,
            f'Text has {n_words} words; at least {MIN_TEXT_WORDS} are needed for a reliable result'
        )

    return text

async def _run_detect(det: Detector, text: str) -> DetectResponse:
    try:
        # Model inference is CPU-bound; keep the event loop free
        return await run_in_threadpool(det.detect, text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------

@app.get('/health')
def health(request: Request):
    det = getattr(request.app.state, 'detector', None)
    if det is None:
        return JSONResponse({'status': 'loading'}, status_code = 503)
    return {'status': 'ok', **det.info}

@app.post('/detect/text', response_model = DetectResponse)
async def detect_text(req: TextRequest, request: Request) -> DetectResponse:
    det = _get_detector(request)
    return await _run_detect(det, _validate_text(req.text))

@app.post('/detect/file', response_model = DetectResponse)
async def detect_file(request: Request, file: UploadFile = File(...)) -> DetectResponse:
    det = _get_detector(request)

    # Check for file extension before reading the upload
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415,
            f'Unsupported file type: {ext or 'none'}. '
            f'Supported file types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}'
        )

    # Read one byte past the limit so oversize files are rejected
    # without loading arbitrarily large uploads into memory
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB')

    try:
        # OCR can take seconds per page; run off the event loop
        text = await run_in_threadpool(extract_text, data, file.filename)
    except ExtractionError as e:
        raise HTTPException(422, str(e)) from e

    return await _run_detect(det, _validate_text(text))
