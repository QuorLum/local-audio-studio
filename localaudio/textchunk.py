"""Sentence-aware chunking for long narration scripts.

Kokoro handles at most ~510 phonemes per call, sounds best on 100-200 tokens
(one or two sentences), is weak on very short utterances and rushes on very long
ones - so sentences are bundled up to ``max_chars`` and never cut mid-sentence. This module splits a script into chunks that

* never cut inside a sentence (abbreviations, initials, decimals and
  ellipses are respected),
* stay below ``max_chars`` (long sentences are split at clause punctuation,
  then at word boundaries),
* carry the pause that should follow them (longer after paragraphs).

It is pure Python so it can be tested and reused without any model installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

DEFAULT_MAX_CHARS = 240  # ~100-250 phoneme tokens: Kokoro's 'goldilocks' range (VOICES.md)
DEFAULT_MIN_CHARS = 60
DEFAULT_SENTENCE_PAUSE = 0.18  # seconds
DEFAULT_PARAGRAPH_PAUSE = 0.60
DEFAULT_CLAUSE_PAUSE = 0.08

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co",
    "fig", "approx", "dept", "est", "mt", "gen", "col", "lt", "sgt", "capt", "rev", "hon",
    "ave", "blvd", "rd", "no", "vol", "pp", "ed", "al", "cf", "ca",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# A sentence terminator run, optional closing quotes/brackets, then whitespace.
_TERMINATOR = re.compile(r"""([.!?\u2026]+)(["'\u201d\u2019)\]]*)(\s+)""")
_NEXT_STARTS_SENTENCE = re.compile(r"""^["'\u201c\u2018(\[]*[A-Z0-9\u00c0-\u00dd\u0400-\u042f]""")
_CLAUSE_SPLIT = re.compile(r"(?<=[;:,\u2014\u2013])\s+|\s+(?=[\u2014\u2013]\s)")


@dataclass(frozen=True)
class Chunk:
    text: str
    pause_after: float  # seconds of silence to insert after this chunk
    paragraph_end: bool = False


def _last_word(before: str) -> str:
    m = re.search(r"([A-Za-z][A-Za-z.]*|\d+)$", before)
    return m.group(1) if m else ""


def _is_boundary(before: str, punct: str, rest: str) -> bool:
    """Decide whether the terminator that ends ``before`` really ends a sentence."""
    if rest and not _NEXT_STARTS_SENTENCE.match(rest):
        return False  # next word is lowercase: '...and then', '"Really?" she said'
    if punct == ".":
        word = _last_word(before)
        stripped = word.rstrip(".").lower()
        if stripped in _ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isalpha() and word.isupper():
            return False  # initials: 'J. K. Rowling'
        if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", word):
            return False  # 'e.g.', 'U.S.', 'a.m.'
        if re.fullmatch(r"\s*\d+", before):
            return False  # list marker '1. Introduction'
    return True


def split_sentences(paragraph: str) -> List[str]:
    """Split one paragraph into sentences."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    sentences: List[str] = []
    start = 0
    for m in _TERMINATOR.finditer(paragraph):
        before = paragraph[start : m.start(1)]
        rest = paragraph[m.end() :]
        if not _is_boundary(before, m.group(1), rest):
            continue
        sentences.append(paragraph[start : m.end(2)].strip())
        start = m.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _split_long(sentence: str, max_chars: int) -> List[str]:
    """Break an over-long sentence at clause punctuation, then at word edges."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: List[str] = []
    current = ""
    for clause in _CLAUSE_SPLIT.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        candidate = f"{current} {clause}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(clause) <= max_chars:
            current = clause
            continue
        # a single clause is still too long: wrap on spaces
        words, line = clause.split(), ""
        for word in words:
            cand = f"{line} {word}".strip()
            if len(cand) <= max_chars or not line:
                line = cand
            else:
                pieces.append(line)
                line = word
        current = line
    if current:
        pieces.append(current)
    return pieces


def _unwrap_lines(paragraph: str) -> str:
    """Decide whether single newlines are soft wraps or hard boundaries.

    Scripts/subtitles often put one sentence per line *without* punctuation; in
    that case every newline is a boundary (we add a full stop). Prose that is
    merely hard-wrapped is re-joined into one flowing paragraph.
    """
    lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return lines[0] if lines else ""
    terminal = sum(1 for ln in lines if re.search(r"[.!?\u2026:;\"'\u201d\u2019)]$", ln))
    if terminal / len(lines) >= 0.4:
        return " ".join(lines)
    return " ".join(ln if re.search(r"[.!?\u2026]$", ln) else ln + "." for ln in lines)


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    sentence_pause: float = DEFAULT_SENTENCE_PAUSE,
    paragraph_pause: float = DEFAULT_PARAGRAPH_PAUSE,
    clause_pause: float = DEFAULT_CLAUSE_PAUSE,
) -> List[Chunk]:
    """Split ``text`` into speakable chunks with pause lengths.

    The last chunk of the whole text gets ``pause_after == 0``.
    """
    if max_chars < 40:
        raise ValueError("max_chars must be at least 40")
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text).strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p_index, raw_paragraph in enumerate(paragraphs):
        paragraph = _unwrap_lines(raw_paragraph)
        sentences = split_sentences(paragraph)

        # pack sentences into chunks <= max_chars
        packed: List[str] = []
        current = ""
        for sentence in sentences:
            for piece in _split_long(sentence, max_chars):
                candidate = f"{current} {piece}".strip()
                if current and len(candidate) > max_chars:
                    packed.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            packed.append(current)

        # merge a tiny tail into its predecessor when it fits
        if len(packed) >= 2 and len(packed[-1]) < min_chars and len(packed[-2]) + 1 + len(packed[-1]) <= max_chars:
            packed[-2] = f"{packed[-2]} {packed[-1]}"
            packed.pop()

        for c_index, piece in enumerate(packed):
            last_in_paragraph = c_index == len(packed) - 1
            pause = paragraph_pause if last_in_paragraph else sentence_pause
            # a chunk that ends mid-sentence (clause split) only needs a breath
            if not last_in_paragraph and not re.search(r"[.!?\u2026][\"'\u201d\u2019)\]]*$", piece):
                pause = clause_pause
            chunks.append(Chunk(piece, pause, last_in_paragraph))

    if chunks:
        last = chunks[-1]
        chunks[-1] = Chunk(last.text, 0.0, last.paragraph_end)
    return chunks
