'''
Tests each individual component in the backend.
Test files are NOT included because they include:
    - three academic articles
    - two fanfics written by other people
    - one published novel
This test script should work without those. It should also NEVER be run in the actual deployment.

Files:
(mostly testing PDFs because they are the most problematic)

Actual research articles
    - Haraway - The Cyborg Manifesto.pdf
    - Latour - Give Me a Laboratory and I will Raise the World.pdf
    - Seaver - On recognising the handiwork of AI.pdf

True digial PDF
    - tiredpjofan - The Request Revolution.pdf
        Downloaded from AO3 as a PDF

Scanned PDF
    - Rick Riordan, Mark Oshiro - The Sun and the Star.pdf
        One page, scanned by phone

Text
    - decayaslife - Staccato.txt
        Copy-pasted into a text file directly from AO3

Word
    - decayaslife - Staccato.docx
        Copy-pasted into Word directly from AO3

Run from the repo root:
    python -m unittest backend.backend_test -v

Optional tests (skipped unless their requirements are met):
    OCR        Tesseract and Poppler (pdftoppm) on PATH
    API        python-multipart and httpx2 (or httpx) installed
    HF model   RUN_SLOW_TESTS=1, uses MODEL_CKPT or models/e3_lora_r8_lr5e-4
'''

from __future__ import annotations

import functools
import importlib.util
import io
import json
import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from backend import chunker as chunker_mod
from backend import extractors
from backend import inference_service as inf
from backend.chunker import ABS_MAX_WORDS, DEFAULT_MAX_WORDS, MIN_CHUNK_WORDS, Chunk, chunker, split_sentences
from backend.extractors import SUPPORTED_EXTENSIONS, ExtractionError, clean_text, extract_text
from backend.model_runner import N_CLASSES, StubModel, load_model
from common.schema import AI_ID, HUMAN_LEVELS, LABELS

# pypdf is very chatty about font quirks in the research PDFs
logging.getLogger('pypdf').setLevel(logging.ERROR)

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_files')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HARAWAY =   'Haraway - The Cyborg Manifesto.pdf'
LATOUR =    'Latour - Give Me a Laboratory and I will Raise the World.pdf'
SEAVER =    'Seaver - On recognising the handiwork of AI.pdf'
REQUEST =   'tiredpjofan - The Request Revolution.pdf'
SCANNED =   'Rick Riordan, Mark Oshiro - The Sun and the Star.pdf'
STACCATO_TXT =  'decayaslife - Staccato.txt'
STACCATO_DOCX = 'decayaslife - Staccato.docx'

TEXT_PDFS = [HARAWAY, LATOUR, SEAVER, REQUEST]
TEXT_FILES = TEXT_PDFS + [STACCATO_TXT, STACCATO_DOCX]

# Characters clean_text must never leave behind
FORBIDDEN_CHARS = '‘’‚‛“”„‟ ​﻿\r\x00'

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def _ocr_available() -> bool:
    if shutil.which('pdftoppm') is None:
        return False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def _read(name: str) -> bytes:
    with open(os.path.join(TEST_DIR, name), 'rb') as f:
        return f.read()

@functools.lru_cache(maxsize = None)
def _extracted(name: str) -> str:
    # Extraction of the larger PDFs takes seconds, so share it across tests
    return extract_text(_read(name), name)

def _low_text_pages(name: str) -> list[int]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(_read(name)))
    return [i for i, p in enumerate(reader.pages)
            if len((p.extract_text() or '').strip()) < extractors.OCR_MIN_CHAR_PER_PAGE]

def _words(n: int, prefix: str = 'w') -> str:
    return ' '.join(f'{prefix}{i}' for i in range(n))

def _sentences(n: int, words_each: int = 8) -> str:
    return ' '.join(f'Sentence {i} ' + _words(words_each - 2, 'x') + '.' for i in range(n))

class ChunkAssertions(unittest.TestCase):
    '''
    Shared checks that every chunker output must satisfy.
    '''

    def assertValidChunks(self, text: str, chunks: list[Chunk], max_words: int = DEFAULT_MAX_WORDS):
        sentences = split_sentences(text, min(max_words, ABS_MAX_WORDS))
        if not sentences:
            self.assertEqual(chunks, [])
            return

        # Indices start at 0, have no gaps and cover every sentence
        self.assertEqual(chunks[0].start_idx, 0)
        self.assertEqual(chunks[-1].end_idx, len(sentences) - 1)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a.end_idx + 1, b.start_idx)

        for c in chunks:
            self.assertTrue(c.sentences, 'empty chunk')
            self.assertEqual(c.sentences, sentences[c.start_idx:c.end_idx + 1])
            self.assertLessEqual(c.word_count, ABS_MAX_WORDS)

        # No words lost or reordered
        self.assertEqual(' '.join(c.text for c in chunks).split(), ' '.join(sentences).split())

        # A short chunk is only allowed when no neighbour could take it
        if len(chunks) > 1:
            for k, c in enumerate(chunks):
                if c.word_count >= MIN_CHUNK_WORDS:
                    continue
                for j in (k - 1, k + 1):
                    if 0 <= j < len(chunks):
                        n = chunks[j]
                        self.assertFalse(
                            len(n.sentences) > 1 or n.word_count + c.word_count <= ABS_MAX_WORDS,
                            f'chunk {k} has {c.word_count} words but could have been merged'
                        )

class FixedModel:
    '''
    Returns the same probability row for every text.
    '''

    name = 'fixed'

    def __init__(self, row):
        self.row = np.asarray(row, dtype = float)

    def predict_probs(self, texts):
        return np.tile(self.row, (len(texts), 1))

class KeywordModel:
    '''
    High 'ai' probability for chunks containing a keyword, low otherwise.
    '''

    name = 'keyword'

    def __init__(self, keyword: str):
        self.keyword = keyword

    def predict_probs(self, texts):
        out = np.full((len(texts), N_CLASSES), 0.02)
        for i, t in enumerate(texts):
            out[i, AI_ID] = 0.86 if self.keyword in t else 0.02
            out[i, LABELS.index('B1')] += 1 - out[i].sum()
        return out

def _row(level: str, p_ai: float) -> list[float]:
    '''
    Probability row with ``p_ai`` on 'ai', most of the rest on ``level``.
    '''
    row = np.full(N_CLASSES, 0.01)
    row[AI_ID] = p_ai
    row[LABELS.index(level)] = 0
    row[LABELS.index(level)] = 1 - row.sum()
    return row.tolist()

# -------------------------------------------------------------------------
# Extractors: synthetic inputs
# -------------------------------------------------------------------------

class TestExtractorValidation(unittest.TestCase):

    def test_empty_file(self):
        with self.assertRaisesRegex(ExtractionError, 'empty'):
            extract_text(b'', 'a.txt')

    def test_unsupported_extension(self):
        for name in ('a.rtf', 'a', 'a.txt.bak', None):
            with self.subTest(name = name), self.assertRaisesRegex(ExtractionError, 'Unsupported'):
                extract_text(b'hello', name)

    def test_extension_is_case_insensitive(self):
        self.assertEqual(extract_text(b'Hello there.', 'NOTES.TXT'), 'Hello there.')
        self.assertEqual(extract_text(b'# Heading', 'readme.Md'), '# Heading')

    def test_supported_extensions(self):
        self.assertEqual(SUPPORTED_EXTENSIONS, {'.txt', '.md', '.docx', '.pdf'})

    def test_whitespace_only_file(self):
        with self.assertRaisesRegex(ExtractionError, 'No text'):
            extract_text(b' \n\n\t ', 'a.txt')

class TestTxtDecoding(unittest.TestCase):

    TEXT = 'Café crème, naïve résumé.'

    def test_encodings(self):
        cases = {
            'utf-8': self.TEXT.encode('utf-8'),
            'utf-8 with BOM': self.TEXT.encode('utf-8-sig'),
            'utf-16 with BOM': self.TEXT.encode('utf-16'),
            'cp1252': self.TEXT.encode('cp1252'),
        }
        for label, data in cases.items():
            with self.subTest(encoding = label):
                self.assertEqual(extract_text(data, 'a.txt'), self.TEXT)

    def test_cp1252_even_length_is_not_read_as_utf16(self):
        # Even-length non-UTF-8 bytes used to "decode" as UTF-16 garbage
        data = 'Café ok!'.encode('cp1252')
        self.assertEqual(len(data) % 2, 0)
        self.assertEqual(extract_text(data, 'a.txt'), 'Café ok!')

    def test_cp1252_smart_quotes(self):
        data = '“Hi,” she said – it’s fine.'.encode('cp1252')
        self.assertEqual(extract_text(data, 'a.txt'), '"Hi," she said - it\'s fine.')

class TestCleanText(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(clean_text(''), '')
        self.assertEqual(clean_text(None), '')

    def test_line_endings(self):
        self.assertEqual(clean_text('one\r\ntwo\r\n\r\nthree\rfour'), 'one two\n\nthree four')

    def test_quotes_and_dashes(self):
        self.assertEqual(clean_text('‘a’ “b” c–d e—f'), '\'a\' "b" c-d e - f')

    def test_invisible_characters(self):
        self.assertEqual(clean_text('a b​c﻿d\x00e\x07f'), 'a bcdef')

    def test_hyphenated_line_break(self):
        self.assertEqual(clean_text('docu-\nment'), 'document')

    def test_single_newlines_become_spaces(self):
        self.assertEqual(clean_text('line one\nline two'), 'line one line two')

    def test_paragraph_breaks_kept_and_collapsed(self):
        self.assertEqual(clean_text('para one\n\n\n\n  para two  '), 'para one\n\npara two')

    def test_repeated_spaces(self):
        self.assertEqual(clean_text('a  \t  b'), 'a b')

    def test_nfkc(self):
        # Ligatures and ellipsis are expanded
        self.assertEqual(clean_text('ﬁne…'), 'fine...')

class TestDocxSynthetic(unittest.TestCase):

    def _docx_bytes(self, build) -> bytes:
        import docx
        document = docx.Document()
        build(document)
        buf = io.BytesIO()
        document.save(buf)
        return buf.getvalue()

    def test_paragraphs(self):
        data = self._docx_bytes(lambda d: [d.add_paragraph('First one.'), d.add_paragraph(''),
                                           d.add_paragraph('Second one.')])
        self.assertEqual(extract_text(data, 'a.docx'), 'First one.\n\nSecond one.')

    def test_tables_and_merged_cells(self):
        def build(d):
            d.add_paragraph('Intro.')
            table = d.add_table(rows = 2, cols = 2)
            table.cell(0, 0).merge(table.cell(0, 1))
            table.cell(0, 0).text = 'Merged header'
            table.cell(1, 0).text = 'Left'
            table.cell(1, 1).text = 'Right'

        text = extract_text(self._docx_bytes(build), 'a.docx')
        self.assertEqual(text.split('\n\n'), ['Intro.', 'Merged header', 'Left', 'Right'])

    def test_corrupt_docx(self):
        with self.assertRaisesRegex(ExtractionError, 'DOCX'):
            extract_text(b'not a zip file', 'a.docx')

class TestPdfSynthetic(unittest.TestCase):

    def test_corrupt_pdf(self):
        with self.assertRaisesRegex(ExtractionError, 'PDF'):
            extract_text(b'%PDF-1.4 garbage', 'a.pdf')

    def test_password_protected_pdf(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.encrypt('secret')
        buf = io.BytesIO()
        writer.write(buf)

        with self.assertRaisesRegex(ExtractionError, 'password'):
            extract_text(buf.getvalue(), 'locked.pdf')

    def test_blank_pdf_falls_back_to_ocr(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.add_blank_page(100, 100)
        buf = io.BytesIO()
        writer.write(buf)

        with mock.patch.object(extractors, '_ocr_pdf_page', return_value = 'Scanned words.') as ocr:
            text = extract_text(buf.getvalue(), 'blank.pdf')
        self.assertEqual([c.args[1] for c in ocr.call_args_list], [0, 1])
        self.assertEqual(text, 'Scanned words.\n\nScanned words.')

    def test_failed_ocr_gives_no_text_error(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        buf = io.BytesIO()
        writer.write(buf)

        with mock.patch.object(extractors, '_ocr_pdf_page', return_value = ''):
            with self.assertRaisesRegex(ExtractionError, 'No text'):
                extract_text(buf.getvalue(), 'blank.pdf')

# -------------------------------------------------------------------------
# Extractors: real test files
# -------------------------------------------------------------------------

@unittest.skipUnless(os.path.isdir(TEST_DIR), 'backend/test_files not present')
class TestExtractorRealFiles(unittest.TestCase):

    def test_all_text_files_extract_cleanly(self):
        for name in TEXT_FILES:
            with self.subTest(file = name):
                text = _extracted(name)
                self.assertGreater(len(text.split()), 1000)
                self.assertFalse(set(text) & set(FORBIDDEN_CHARS), 'uncleaned characters left')
                self.assertNotRegex(text, r'(?<!\n)\n(?!\n)', 'single newline left in text')
                self.assertNotRegex(text, r'\w-\n\w', 'hyphenated line break left in text')
                self.assertNotIn('\n\n\n', text)
                self.assertEqual(text, text.strip())

    def test_txt_and_docx_match(self):
        # The same story pasted into Notepad and into Word must extract identically
        self.assertEqual(_extracted(STACCATO_TXT), _extracted(STACCATO_DOCX))

    def test_staccato_paragraphs_kept(self):
        text = _extracted(STACCATO_TXT)
        self.assertTrue(text.startswith('If anyone asked'))
        self.assertGreater(text.count('\n\n'), 50)

    def test_known_content(self):
        cases = {
            HARAWAY: 'Cyborg Manifesto',
            LATOUR: 'École des Mines',     # accented capital survives NFKC
            SEAVER: '10.1111/amet.70062',
            REQUEST: 'Archive of Our Own',
        }
        for name, needle in cases.items():
            with self.subTest(file = name):
                self.assertIn(needle, _extracted(name))

    def test_request_revolution_ocr_only_on_empty_page(self):
        # The AO3 PDF has one page with no text layer; only that page should go to OCR
        expected = _low_text_pages(REQUEST)
        self.assertEqual(len(expected), 1)
        with mock.patch.object(extractors, '_ocr_pdf_page', return_value = '') as ocr:
            extract_text(_read(REQUEST), REQUEST)
        self.assertEqual([c.args[1] for c in ocr.call_args_list], expected)

    def test_research_pdfs_need_no_ocr(self):
        for name in (HARAWAY, LATOUR, SEAVER):
            with self.subTest(file = name), mock.patch.object(extractors, '_ocr_pdf_page') as ocr:
                extract_text(_read(name), name)
                ocr.assert_not_called()

    def test_scanned_pdf_routes_to_ocr(self):
        with mock.patch.object(extractors, '_ocr_pdf_page', return_value = 'Line one\nline two.') as ocr:
            text = extract_text(_read(SCANNED), SCANNED)
        ocr.assert_called_once()
        self.assertEqual(text, 'Line one line two.')

    @unittest.skipUnless(_ocr_available(), 'Tesseract and Poppler are not installed')
    def test_scanned_pdf_real_ocr(self):
        text = _extracted(SCANNED)
        self.assertGreater(len(text.split()), 150)
        self.assertFalse(set(text) & set(FORBIDDEN_CHARS))

    @unittest.expectedFailure
    def test_pdf_page_breaks_do_not_split_sentences(self):
        # Known issue: pages are joined with a blank line, which split_sentences treats
        # as a sentence end, so a sentence running over a page break is cut in two.
        # Haraway: every page join falls mid-sentence. Remove this decorator once fixed.
        text = _extracted(HARAWAY)
        breaks = text.split('\n\n')[:-1]
        mid_sentence = [b for b in breaks if b.rstrip()[-1:] not in '.!?"\')']
        self.assertLess(len(mid_sentence), len(breaks) / 2)

# -------------------------------------------------------------------------
# Chunker
# -------------------------------------------------------------------------

class TestChunkDataclass(unittest.TestCase):

    def test_properties(self):
        c = Chunk(3, ['One two three.', 'Four  five.'])
        self.assertEqual(c.end_idx, 4)
        self.assertEqual(list(c.sentence_indices), [3, 4])
        self.assertEqual(c.text, 'One two three. Four  five.')
        self.assertEqual(c.word_count, 5)

class TestSplitSentences(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(split_sentences(''), [])
        self.assertEqual(split_sentences('  \n\n '), [])
        self.assertEqual(split_sentences(None), [])

    def test_basic(self):
        self.assertEqual(split_sentences('Hello there. How are you? Fine!'),
                         ['Hello there.', 'How are you?', 'Fine!'])

    def test_abbreviations_do_not_split(self):
        self.assertEqual(split_sentences('Dr. Smith met Mr. Jones at 3 p.m. on Friday. They talked.'),
                         ['Dr. Smith met Mr. Jones at 3 p.m. on Friday.', 'They talked.'])

    def test_paragraph_break_ends_sentence(self):
        self.assertEqual(split_sentences('No full stop here\n\nNext paragraph'),
                         ['No full stop here', 'Next paragraph'])

    def test_paragraph_break_with_spaces(self):
        self.assertEqual(split_sentences('First part\n   \nSecond part'), ['First part', 'Second part'])

    def test_whitespace_and_quotes_normalised(self):
        self.assertEqual(split_sentences('“Hi,”  she  said.'), ['"Hi," she said.'])

    @unittest.expectedFailure
    def test_single_newline_does_not_end_sentence(self):
        # Known issue: pysbd treats a single newline as a sentence end.
        # Files are fine (clean_text removes single newlines), but /detect/text sends
        # pasted text straight to the chunker, so hard-wrapped pastes split at every line.
        self.assertEqual(split_sentences('This sentence was\nwrapped by an email client.'),
                         ['This sentence was wrapped by an email client.'])

    def test_sentence_at_cap_not_split(self):
        sentence = _words(ABS_MAX_WORDS) + '.'
        self.assertEqual(split_sentences(sentence), [sentence])

    def test_long_sentence_split(self):
        for n in (ABS_MAX_WORDS + 1, 700, 2000):
            for piece_words in (50, DEFAULT_MAX_WORDS, ABS_MAX_WORDS):
                with self.subTest(words = n, piece_words = piece_words):
                    sentence = _words(n) + '.'
                    pieces = split_sentences(sentence, piece_words)
                    sizes = [len(p.split()) for p in pieces]
                    self.assertGreater(len(pieces), 1)
                    self.assertLessEqual(max(sizes), piece_words)
                    self.assertEqual(' '.join(pieces).split(), sentence.split())
                    # Balanced: no tiny leftover piece
                    self.assertGreaterEqual(min(sizes), piece_words // 2)

    def test_long_sentence_prefers_clause_boundary(self):
        words = [f'w{i}' for i in range(400)]
        words[129] += ','
        pieces = split_sentences(' '.join(words) + '.', DEFAULT_MAX_WORDS)
        self.assertTrue(pieces[0].endswith('w129,'))

class TestChunker(ChunkAssertions):

    def test_empty(self):
        self.assertEqual(chunker(''), [])

    def test_single_short_text(self):
        chunks = chunker('Just a few words.')
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].start_idx, chunks[0].end_idx), (0, 0))

    def test_respects_max_words(self):
        text = _sentences(60)
        for max_words in (50, 100, DEFAULT_MAX_WORDS):
            with self.subTest(max_words = max_words):
                chunks = chunker(text, max_words)
                self.assertValidChunks(text, chunks, max_words)
                # Only the merged tail may exceed max_words
                for c in chunks[:-1]:
                    self.assertLessEqual(c.word_count, max_words)

    @unittest.expectedFailure
    def test_respects_max_words_below_min_chunk(self):
        # Known issue: with max_words < MIN_CHUNK_WORDS every chunk counts as short,
        # so the merge step glues them into chunks larger than max_words.
        chunks = chunker(_sentences(60), 20)
        self.assertLessEqual(max(c.word_count for c in chunks), 20)

    def test_max_words_capped(self):
        text = _sentences(100)
        chunks = chunker(text, 1000)
        self.assertValidChunks(text, chunks, 1000)
        self.assertLessEqual(max(c.word_count for c in chunks), ABS_MAX_WORDS)
        self.assertGreater(len(chunks), 1)

    def test_short_tail_merged(self):
        # 18 sentences fill 144 words, the last 3 (24 words) are merged back in
        chunks = chunker(_sentences(21))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].word_count, 168)

    def test_short_sentence_before_long_merged(self):
        text = 'A short intro sentence here. ' + _words(700) + '. A short outro.'
        chunks = chunker(text)
        self.assertValidChunks(text, chunks)
        self.assertGreaterEqual(min(c.word_count for c in chunks), MIN_CHUNK_WORDS)
        self.assertTrue(chunks[0].text.startswith('A short intro'))

    def test_short_sentence_between_long_sentences(self):
        text = _words(290) + '. Tiny one here. ' + _words(290) + '.'
        chunks = chunker(text)
        self.assertValidChunks(text, chunks)
        self.assertEqual(len(chunks), 2)

    def test_borrowing_when_neighbours_are_full(self):
        text = 'A short intro sentence here. ' + _words(700) + '. A short outro.'
        chunks = chunker(text, ABS_MAX_WORDS)
        self.assertValidChunks(text, chunks, ABS_MAX_WORDS)
        self.assertGreaterEqual(min(c.word_count for c in chunks), MIN_CHUNK_WORDS)

    def test_randomised_invariants(self):
        rng = np.random.default_rng(0)
        for case in range(60):
            max_words = int(rng.choice([50, 100, 150, 300]))
            sentences = []
            for _ in range(int(rng.integers(1, 20))):
                n = int(rng.choice([rng.integers(1, 15), rng.integers(20, 120), rng.integers(200, 900)]))
                sentences.append(' '.join(f'w{k}' + (',' if rng.random() < 0.05 else '') for k in range(n)) + '.')
            text = ' '.join(sentences)
            with self.subTest(case = case, max_words = max_words):
                self.assertValidChunks(text, chunker(text, max_words), max_words)

@unittest.skipUnless(os.path.isdir(TEST_DIR), 'backend/test_files not present')
class TestChunkerRealFiles(ChunkAssertions):

    def test_real_files(self):
        for name in TEXT_FILES:
            with self.subTest(file = name):
                text = _extracted(name)
                self.assertValidChunks(text, chunker(text))

# -------------------------------------------------------------------------
# Model runner
# -------------------------------------------------------------------------

class TestStubModel(unittest.TestCase):

    def test_shape_and_rows_sum_to_one(self):
        probs = StubModel().predict_probs(['a', 'b', 'c'])
        self.assertEqual(probs.shape, (3, N_CLASSES))
        np.testing.assert_allclose(probs.sum(axis = 1), 1.0)
        self.assertTrue((probs > 0).all())

    def test_empty_input(self):
        self.assertEqual(StubModel().predict_probs([]).shape, (0, N_CLASSES))

    def test_deterministic_per_text(self):
        a = StubModel().predict_probs(['same text', 'other text'])
        b = StubModel().predict_probs(['same text'])
        np.testing.assert_array_equal(a[0], b[0])
        self.assertFalse(np.allclose(a[0], a[1]))

    def test_ai_bias(self):
        texts = [f'text {i}' for i in range(50)]
        plain = StubModel().predict_probs(texts)[:, AI_ID]
        biased = StubModel(ai_bias = 3.0).predict_probs(texts)[:, AI_ID]
        self.assertTrue((biased > plain).all())

class TestLoadModel(unittest.TestCase):

    def test_stub(self):
        self.assertEqual(load_model('stub', None).name, 'stub')
        self.assertEqual(load_model(None, None).name, 'stub')
        self.assertEqual(load_model('STUB', None).name, 'stub')

    def test_stub_bias_from_env(self):
        with mock.patch.dict(os.environ, {'STUB_AI_BIAS': '2.5'}):
            self.assertEqual(load_model('stub', None).ai_bias, 2.5)

    def test_hf_needs_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'MODEL_CKPT'):
            load_model('hf', None)

    def test_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, 'Unknown DETECTOR_MODE'):
            load_model('gpt', None)

HF_CKPT = os.environ.get('MODEL_CKPT') or os.path.join(REPO_ROOT, 'models', 'e3_lora_r8_lr5e-4')

@unittest.skipUnless(os.environ.get('RUN_SLOW_TESTS') == '1' and os.path.isdir(HF_CKPT),
                     'set RUN_SLOW_TESTS=1 to load the real checkpoint')
class TestHFModelRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = load_model('hf', HF_CKPT)

    def test_predict_probs(self):
        probs = self.model.predict_probs(['A short essay about my holidays.', 'Another text.'])
        self.assertEqual(probs.shape, (2, N_CLASSES))
        np.testing.assert_allclose(probs.sum(axis = 1), 1.0, atol = 1e-5)

    def test_empty_input(self):
        self.assertEqual(self.model.predict_probs([]).shape, (0, N_CLASSES))

    def test_batching_matches_single(self):
        texts = [f'Sentence number {i} for batching.' for i in range(5)]
        batched = self.model.predict_probs(texts)
        single = np.vstack([self.model.predict_probs([t]) for t in texts])
        np.testing.assert_allclose(batched, single, atol = 1e-4)

# -------------------------------------------------------------------------
# Inference service
# -------------------------------------------------------------------------

class TestDetector(ChunkAssertions):

    def test_stub_end_to_end(self):
        text = _sentences(60)
        r = inf.Detector(mode = 'stub').detect(text)

        self.assertIn(r.estimated_cefr, HUMAN_LEVELS)
        self.assertTrue(0 <= r.overall_ai_pct <= 100)
        self.assertTrue(0 <= r.overall_confidence <= 1)

        chunks = chunker(text)
        self.assertEqual([(c.start_idx, c.end_idx, c.text) for c in r.chunks],
                         [(c.start_idx, c.end_idx, c.text) for c in chunks])
        for c in r.chunks:
            self.assertTrue(0 <= c.p_ai_raw <= 1)
            self.assertTrue(0 <= c.p_ai_calibrated <= 1)
            self.assertTrue(0 <= c.confidence <= 1)
            # Calibration maps the threshold to 0.5
            self.assertEqual(c.flagged, c.p_ai_calibrated >= 0.5)

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            inf.Detector(mode = 'stub').detect('   ')

    def test_wrong_model_shape(self):
        bad = FixedModel([0.5, 0.5])
        with self.assertRaisesRegex(RuntimeError, 'shape'):
            inf.Detector(model = bad).detect('Some text here.')

    def test_document_level_estimate(self):
        r = inf.Detector(model = FixedModel(_row('B2', 0.1))).detect(_sentences(30))
        self.assertEqual(r.estimated_cefr, 'B2')

    def test_uncalibrated_threshold(self):
        r = inf.Detector(model = FixedModel(_row('A2', 0.1))).detect(_sentences(30))
        for c in r.chunks:
            self.assertAlmostEqual(c.p_ai_raw, 0.1, places = 4)
            # p / tau * 0.5 with tau = 0.5
            self.assertAlmostEqual(c.p_ai_calibrated, 0.1, places = 4)
            self.assertFalse(c.flagged)
        self.assertEqual(r.overall_ai_pct, 0.0)

    def test_overall_pct_is_word_weighted(self):
        text = _sentences(20) + '\n\n' + 'robot ' + _sentences(10)
        r = inf.Detector(model = KeywordModel('robot')).detect(text)
        flagged = sum(len(c.text.split()) for c in r.chunks if c.flagged)
        total = sum(len(c.text.split()) for c in r.chunks)
        self.assertTrue(any(c.flagged for c in r.chunks) and not all(c.flagged for c in r.chunks))
        self.assertAlmostEqual(r.overall_ai_pct, round(100 * flagged / total, 2), places = 2)

    def test_info(self):
        info = inf.Detector(mode = 'stub').info
        self.assertEqual(info['model'], 'stub')
        self.assertEqual(info['calibration'], inf.CALIBRATION_SOURCE)
        self.assertEqual(set(info['thresholds']), set(HUMAN_LEVELS))
        self.assertEqual(info['max_words_per_chunk'], DEFAULT_MAX_WORDS)

    @unittest.skipUnless(os.path.isdir(TEST_DIR), 'backend/test_files not present')
    def test_real_file(self):
        r = inf.Detector(mode = 'stub').detect(_extracted(STACCATO_TXT))
        self.assertEqual(len(r.chunks), len(chunker(_extracted(STACCATO_TXT))))

class TestThresholds(unittest.TestCase):

    def _write(self, data) -> str:
        f = tempfile.NamedTemporaryFile('w', suffix = '.json', delete = False)
        json.dump(data, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_missing_file(self):
        det = inf.Detector(mode = 'stub', thresholds_path = 'does/not/exist.json')
        self.assertFalse(det.thresholds_loaded)
        self.assertEqual(det.thresholds, {l: inf.DEFAULT_THRESHOLD for l in HUMAN_LEVELS})

    def test_partial_file_filled_with_default(self):
        path = self._write({'B2': 0.05, 'native': 0.3})
        det = inf.Detector(mode = 'stub', thresholds_path = path)
        self.assertTrue(det.thresholds_loaded)
        self.assertEqual(det.thresholds['B2'], 0.05)
        self.assertEqual(det.thresholds['native'], 0.3)
        self.assertEqual(det.thresholds['A1'], inf.DEFAULT_THRESHOLD)

    def test_level_threshold_applied(self):
        path = self._write({'B2': 0.05})
        det = inf.Detector(model = FixedModel(_row('B2', 0.1)), thresholds_path = path)
        r = det.detect(_sentences(30))
        self.assertEqual(r.overall_ai_pct, 100.0)
        # 0.5 + 0.5 * (0.1 - 0.05) / 0.95
        for c in r.chunks:
            self.assertTrue(c.flagged)
            self.assertAlmostEqual(c.p_ai_calibrated, 0.5263, places = 4)

    def test_other_level_threshold_ignored(self):
        path = self._write({'B2': 0.05})
        det = inf.Detector(model = FixedModel(_row('C1', 0.1)), thresholds_path = path)
        self.assertEqual(det.detect(_sentences(30)).overall_ai_pct, 0.0)

class TestCalibrationFallbacks(unittest.TestCase):
    '''
    The backend fallbacks must behave exactly like the calibration package.
    '''

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        cls.probs = rng.dirichlet(np.ones(N_CLASSES), size = 40)
        cls.levels = [HUMAN_LEVELS[i % len(HUMAN_LEVELS)] for i in range(40)]
        cls.thresholds = {l: float(t) for l, t in zip(HUMAN_LEVELS, np.linspace(0.1, 0.9, len(HUMAN_LEVELS)))}

    def test_fallbacks_are_self_consistent(self):
        score, flagged = inf._fallback_apply_thresholds(self.probs, self.levels, self.thresholds)
        np.testing.assert_array_equal(flagged, score >= 0.5)
        conf = inf._fallback_confidence(score, self.probs)
        self.assertTrue(((conf >= 0) & (conf <= 1)).all())
        self.assertEqual(len(inf._fallback_estimate_cefr(self.probs)), 40)

    @unittest.skipUnless(_has_module('calibration'), 'calibration package not importable')
    def test_matches_calibration_package(self):
        from calibration.cefr_estimator import estimate_cefr
        from calibration.confidence import confidence
        from calibration.threshold_calibrator import apply_thresholds

        self.assertEqual(list(inf._fallback_estimate_cefr(self.probs)), list(estimate_cefr(self.probs)))

        score_fb, flag_fb = inf._fallback_apply_thresholds(self.probs, self.levels, self.thresholds)
        score, flag = apply_thresholds(self.probs, self.levels, self.thresholds)
        np.testing.assert_allclose(score_fb, score)
        np.testing.assert_array_equal(flag_fb, flag)

        np.testing.assert_allclose(inf._fallback_confidence(score, self.probs), confidence(score, self.probs))

# -------------------------------------------------------------------------
# API
# -------------------------------------------------------------------------

@unittest.skipUnless(_has_module('multipart') and (_has_module('httpx2') or _has_module('httpx')),
                     'API tests need python-multipart and httpx2 (or httpx)')
class TestApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from backend import api
        cls.api = api
        cls.ctx = TestClient(api.app)
        cls.client = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get('/health')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['status'], 'ok')
        self.assertEqual(r.json()['model'], 'stub')

    def test_request_id_header(self):
        self.assertEqual(self.client.get('/health', headers = {'x-request-id': 'abc'}).headers['x-request-id'], 'abc')
        self.assertTrue(self.client.get('/health').headers['x-request-id'])

    def test_detect_text(self):
        r = self.client.post('/detect/text', json = {'text': _sentences(40)})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(set(body), {'estimated_cefr', 'overall_ai_pct', 'overall_confidence', 'chunks'})
        self.assertEqual(set(body['chunks'][0]),
                         {'text', 'start_idx', 'end_idx', 'p_ai_raw', 'p_ai_calibrated', 'flagged', 'confidence'})

    def test_text_validation(self):
        cases = {
            'empty': ('   ', 400),
            'too short': ('Only five words here, really.', 400),
            'too long': ('word ' * (self.api.MAX_TEXT_CHARS // 5 + 10), 413),
        }
        for label, (text, status) in cases.items():
            with self.subTest(case = label):
                self.assertEqual(self.client.post('/detect/text', json = {'text': text}).status_code, status)

    def test_missing_text_field(self):
        self.assertEqual(self.client.post('/detect/text', json = {}).status_code, 422)

    def test_detect_file_txt(self):
        files = {'file': ('essay.txt', _sentences(40).encode(), 'text/plain')}
        self.assertEqual(self.client.post('/detect/file', files = files).status_code, 200)

    @unittest.skipUnless(os.path.isdir(TEST_DIR), 'backend/test_files not present')
    def test_detect_file_real(self):
        for name in (STACCATO_TXT, STACCATO_DOCX, SEAVER):
            with self.subTest(file = name):
                r = self.client.post('/detect/file', files = {'file': (name, _read(name))})
                self.assertEqual(r.status_code, 200)
                self.assertGreater(len(r.json()['chunks']), 1)

    def test_unsupported_file_type(self):
        r = self.client.post('/detect/file', files = {'file': ('essay.rtf', b'hello')})
        self.assertEqual(r.status_code, 415)

    def test_oversize_file(self):
        with mock.patch.object(self.api, 'MAX_UPLOAD_BYTES', 100):
            r = self.client.post('/detect/file', files = {'file': ('essay.txt', b'x' * 101)})
        self.assertEqual(r.status_code, 413)

    def test_corrupt_file(self):
        r = self.client.post('/detect/file', files = {'file': ('essay.pdf', b'%PDF-1.4 garbage')})
        self.assertEqual(r.status_code, 422)

    def test_extracted_text_too_short(self):
        r = self.client.post('/detect/file', files = {'file': ('essay.txt', b'Too short.')})
        self.assertEqual(r.status_code, 400)

if __name__ == '__main__':
    unittest.main()
