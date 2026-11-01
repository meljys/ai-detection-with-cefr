'''
Sentence splitting and preserving chunking

Each chunk keeps the indices of the sentences it covers, so the frontend can map a flagged chunk back to the sentences to highlight.
'''

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_MAX_WORDS = 150
# A chunk shorter than this is merged into a neighbouring chunk
MIN_CHUNK_WORDS = 40
# Hardcap on chunk length
ABS_MAX_WORDS = 300

@dataclass
class Chunk:
    start_idx: int
    sentences: list[str] = field(default_factory = list)

    @property
    def end_idx(self) -> int:
        return self.start_idx + len(self.sentences) - 1

    @property
    def sentence_indices(self) -> range:
        return range(self.start_idx, self.end_idx + 1)

    @property
    def text(self) -> str:
        return ' '.join(self.sentences)

    @property
    def word_count(self) -> int:
        return sum(_word_count(s) for s in self.sentences)

def _word_count(s: str) -> int:
    return len(s.split())

# -------------------------------------------------------------------------
# Sentence splitting
# -------------------------------------------------------------------------

def _normalise(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()

def _split_long_sentence(sentence: str, piece_words: int = DEFAULT_MAX_WORDS) -> list[str]:
    '''
    Break a sentence longer than ``ABS_MAX_WORDS`` into balanced pieces of at most ``piece_words``.
    Cuts after the last clause punctuation (, ; :) in the last quarter of each window,
    or at the window end if there is none.

    Pieces are sized to the chunk target rather than the hard cap,
    so a short neighbouring chunk can always be merged next to one.
    '''

    words = sentence.split()
    if len(words) <= ABS_MAX_WORDS:
        return [sentence]

    pieces = []
    while len(words) > piece_words:
        # Spread the remaining words evenly over the pieces still needed
        n_pieces = -(-len(words) // piece_words)
        target = -(-len(words) // n_pieces)

        cut = target
        # Prefer a clause boundary so pieces read naturally
        for i in range(target, (target * 3) // 4, -1):
            if words[i - 1][-1] in ',;:':
                cut = i
                break
        pieces.append(' '.join(words[:cut]))
        words = words[cut:]

    if words:
        pieces.append(' '.join(words))

    return pieces

def split_sentences(text: str, piece_words: int = DEFAULT_MAX_WORDS) -> list[str]:
    '''
    Split text into sentences, normalises whitespace and quotes.
    Paragraph breaks are treated as sentence ends.
    Sentences longer than ``ABS_MAX_WORDS`` are split into pieces of at most ``piece_words``,
    and each piece counts as its own sentence for indexing.

    Call with the same ``piece_words`` as the chunker's ``max_words``
    so sentence indices line up with the chunks.
    '''
    # Input validation
    if not text or not text.strip():
        return []

    import pysbd
    seg = pysbd.Segmenter(language = 'en', clean = False)
    sentences = []

    # Split on paragraphs
    for para in re.split(r'\n\s*\n', text):
        para = para.strip()
        if not para:
            continue
        pieces = seg.segment(para)
        for p in pieces:
            p = _normalise(p)
            if p:
                sentences.extend(_split_long_sentence(p, piece_words))

    return sentences

# -------------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------------

def chunker(text: str, max_words: int = DEFAULT_MAX_WORDS) -> list[Chunk]:
    '''
    Groups consecutive sentences into chunks of up to ``max_words``.

    ``max_words`` is capped at ``ABS_MAX_WORDS``.
    A single sentence longer than ``max_words`` is put in its own chunk;
    split_sentences already guarantees no sentence exceeds ``ABS_MAX_WORDS``.
    Any chunk shorter than ``MIN_CHUNK_WORDS`` is merged into a neighbouring chunk,
    as long as the merged chunk stays within ``ABS_MAX_WORDS``.
    '''

    # Input validation
    max_words = max(1, min(max_words, ABS_MAX_WORDS))

    chunks: list[Chunk] = []
    chunk: list[str] = []
    chunk_len = 0
    start_idx = 0

    sentences = split_sentences(text, max_words)
    for i, sentence in enumerate(sentences):
        word_count = _word_count(sentence)

        # Start a new chunk if this sentence would overflow the current one
        if chunk and chunk_len + word_count > max_words:
            chunks.append(Chunk(start_idx, chunk))
            chunk = []
            chunk_len = 0
            start_idx = i

        chunk.append(sentence)
        chunk_len += word_count

    if chunk:
        chunks.append(Chunk(start_idx, chunk))

    return _merge_short_chunks(chunks)

def _merge_short_chunks(chunks: list[Chunk]) -> list[Chunk]:
    '''
    Merge every chunk shorter than ``MIN_CHUNK_WORDS`` into the neighbour
    that gives the smaller merged chunk, without exceeding ``ABS_MAX_WORDS``.
    If neither neighbour has room, sentences are moved across the boundary instead.
    A short chunk is left alone only if both neighbours are single sentences too long to share.
    '''

    i = 0
    while i < len(chunks):
        curr = chunks[i]
        if len(chunks) == 1 or curr.word_count >= MIN_CHUNK_WORDS:
            i += 1
            continue

        # (merged size, index of the left chunk in the pair)
        options = []
        if i > 0:
            options.append((chunks[i - 1].word_count + curr.word_count, i - 1))
        if i + 1 < len(chunks):
            options.append((curr.word_count + chunks[i + 1].word_count, i))
        options = [o for o in options if o[0] <= ABS_MAX_WORDS]

        if not options:
            _borrow_sentences(chunks, i)
            i += 1
            continue

        _, left = min(options)
        merged = Chunk(chunks[left].start_idx, chunks[left].sentences + chunks[left + 1].sentences)
        chunks[left:left + 2] = [merged]
        # Re-check the merged chunk, it may still be short
        i = left

    return chunks

def _borrow_sentences(chunks: list[Chunk], i: int) -> None:
    '''
    Grow the short chunk at ``i`` by moving boundary sentences out of a neighbour.
    Only moves a sentence if the neighbour keeps at least ``MIN_CHUNK_WORDS``
    and the short chunk stays within ``ABS_MAX_WORDS``.
    Tries the left neighbour first, then the right.
    '''

    curr = chunks[i]

    # Take the last sentences of the left neighbour
    if i > 0:
        prev = chunks[i - 1]
        while curr.word_count < MIN_CHUNK_WORDS and len(prev.sentences) > 1:
            moved = _word_count(prev.sentences[-1])
            if prev.word_count - moved < MIN_CHUNK_WORDS or curr.word_count + moved > ABS_MAX_WORDS:
                break
            curr.sentences.insert(0, prev.sentences.pop())
            curr.start_idx -= 1

    # Take the first sentences of the right neighbour
    if i + 1 < len(chunks):
        nxt = chunks[i + 1]
        while curr.word_count < MIN_CHUNK_WORDS and len(nxt.sentences) > 1:
            moved = _word_count(nxt.sentences[0])
            if nxt.word_count - moved < MIN_CHUNK_WORDS or curr.word_count + moved > ABS_MAX_WORDS:
                break
            curr.sentences.append(nxt.sentences.pop(0))
            nxt.start_idx += 1