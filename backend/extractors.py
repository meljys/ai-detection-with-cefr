'''
Text extraction for uploaded files (text, Word docs, PDF with OCR fallback)

Public entry point: extract_text(file_bytes, filename) -> str
Raises ExtractionError with message on anything the API reports back as 4xx
'''

from __future__ import annotations

import io
import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.docx', '.pdf'}

OCR_MIN_CHAR_PER_PAGE = 25
OCR_DPI =               300

class ExtractionError(ValueError):
    '''
    Raised when a file cannot be turned into text.
    '''

# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def extract_text(file_bytes: bytes, filename: str) -> str:
    '''
    Pulls and cleans text from input file.
    Filetype is determined from file extension.
    '''

    # If no file data is passed
    if not file_bytes:
        raise ExtractionError('Uploaded file is empty')

    # Check for file extension
    ext = os.path.splitext(filename  or '')[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(
            f'Unsupported file type: {ext or 'none'}. '
            f'Supported file types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}'
        )

    if ext in {'.txt', '.md'}:
        raw = _extract_txt(file_bytes)
    elif ext == '.docx':
        raw = _extract_docx(file_bytes)
    else:
        raw = _extract_pdf(file_bytes)

    text = clean_text(raw)
    if not text:
        raise ExtractionError('No text found in file')

    return text

# -------------------------------------------------------------------------
# Extractors
# -------------------------------------------------------------------------

def _extract_txt(file_bytes: bytes) -> str:
    # UTF-16 only with a BOM: without one, almost any even-length bytes "decode" to garbage
    if file_bytes[:2] in (b'\xff\xfe', b'\xfe\xff'):
        try:
            return file_bytes.decode('utf-16')
        except UnicodeError:
            pass

    # UTF-8-SIG strips a BOM if present, CP1252 as a Windows-based fallback
    # Latin-1 serves as the final fallback
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            text = file_bytes.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue

        return text

    raise ExtractionError('Text file could not be decoded') # Should not be reachable

def _extract_docx(file_bytes: bytes) -> str:
    import docx

    # Try open the word file
    try:
        document = docx.Document(io.BytesIO(file_bytes))
    except Exception as e:
        raise ExtractionError('DOCX file could not be opened') from e

    parts: list[str] = [p.text for p in document.paragraphs]

    # Extract from tables
    for table in document.tables:
        for row in table.rows:
            seen = set()
            for cell in row.cells:
                # Merged cell check
                if id(cell._tc) in seen:
                    continue
                seen.add(id(cell._tc))
                parts.append(cell.text)

    return '\n\n'.join(p for p in parts if p.strip())

def _extract_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    # Try open the PDF
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            # Try empty password first
            if reader.decrypt('') == 0:
                raise ExtractionError('PDF is password-protected')
        pages = reader.pages
        n_pages = len(pages)
    except ExtractionError:
        # Keep the password-protected message instead of the generic one
        raise
    except Exception as e:
        raise ExtractionError('PDF file could not be opened') from e

    page_text: list[str] = []
    ocr_pages = 0
    for i, page in enumerate(pages):
        try:
            text = page.extract_text() or ''
        except Exception:
            logger.warning('pypdf failed on page %d; falling back to OCR', i + 1)
            text = ''
        if len(text.strip()) < OCR_MIN_CHAR_PER_PAGE:
            text = _ocr_pdf_page(file_bytes, i)
            ocr_pages += 1
        page_text.append(text)

    if ocr_pages:
        logger.info('OCR was used on %d/%d PDF pages', ocr_pages, n_pages)
    return '\n\n'.join(page_text)

def _ocr_pdf_page(file_bytes:bytes, page_index: int) -> str:
    import pytesseract
    from pdf2image import convert_from_bytes

    # Try to convert the PDF page to an image
    try:
        images = convert_from_bytes(
            file_bytes, dpi = OCR_DPI, first_page = page_index + 1, last_page = page_index + 1
        )
        return '\n'.join(pytesseract.image_to_string(img, lang = 'eng') for img in images)
    except Exception as e:
        logger.warning('OCR failed on page %d: %s', page_index + 1, e)
        return ''

# -------------------------------------------------------------------------
# Cleaning
# -------------------------------------------------------------------------

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": " - ", "−": "-",
    " ": " ", "​": "", "﻿": "",
})

def clean_text(text: str) -> str:
    '''
    Normalise whitespace, keep paragraph breaks
    '''

    if not text:
        return ''

    # Character normalisation:
    # Fixes issue where strings look identical but use different underlying characters
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(_QUOTE_MAP)
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', '')

    # Drop control characters that aren't newline or tab
    text = ''.join(ch for ch in text if ch in '\n\t' or unicodedata.category(ch)[0] != 'C')

    # Re-join words hyphenated across PDF line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Single newlines in a paragraph (PDF line wrapping) become spaces
    # Blank lines stay as paragraph breaks
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()