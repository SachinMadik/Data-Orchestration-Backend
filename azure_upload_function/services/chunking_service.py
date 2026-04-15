"""
chunking_service.py — Split document text into overlapping token chunks.

Strategy:
  - ~500 tokens per chunk  (~2000 chars, since avg English word ≈ 4 chars + space)
  - 50 token overlap       (~200 chars) to preserve context across boundaries
  - Split on sentence boundaries where possible to avoid mid-sentence cuts
"""

import re
import logging

# Approximate chars per token for English text
_CHARS_PER_TOKEN = 4
CHUNK_TOKENS     = 500
OVERLAP_TOKENS   = 50
CHUNK_SIZE       = CHUNK_TOKENS  * _CHARS_PER_TOKEN   # 2000 chars
OVERLAP_SIZE     = OVERLAP_TOKENS * _CHARS_PER_TOKEN  # 200 chars


def chunk_text(text: str, doc_id: str, filename: str) -> list[dict]:
    """
    Split text into overlapping chunks.

    Returns list of dicts:
      { chunk_id, doc_id, filename, chunk_index, text, token_estimate }
    """
    if not text or not text.strip():
        return []

    # Split into sentences for cleaner boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks   = []
    current  = []
    cur_len  = 0
    idx      = 0

    for sentence in sentences:
        s_len = len(sentence)

        # If a single sentence exceeds chunk size, hard-split it
        if s_len > CHUNK_SIZE:
            # Flush current buffer first
            if current:
                _flush(chunks, current, cur_len, idx, doc_id, filename)
                idx    += 1
                current = []
                cur_len = 0
            # Hard-split the long sentence
            for start in range(0, s_len, CHUNK_SIZE - OVERLAP_SIZE):
                piece = sentence[start:start + CHUNK_SIZE]
                chunks.append(_make_chunk(piece, idx, doc_id, filename))
                idx += 1
            continue

        # If adding this sentence exceeds chunk size → flush + start new with overlap
        if cur_len + s_len > CHUNK_SIZE and current:
            _flush(chunks, current, cur_len, idx, doc_id, filename)
            idx += 1
            # Overlap: carry last N chars from previous chunk
            overlap_text = " ".join(current)[-OVERLAP_SIZE:]
            current  = [overlap_text] if overlap_text.strip() else []
            cur_len  = len(overlap_text)

        current.append(sentence)
        cur_len += s_len + 1  # +1 for space

    # Flush remaining
    if current:
        _flush(chunks, current, cur_len, idx, doc_id, filename)

    logging.info("chunking: '%s' → %d chunks (chunk_size=%d, overlap=%d)",
                 filename, len(chunks), CHUNK_SIZE, OVERLAP_SIZE)
    return chunks


def _flush(chunks, current, cur_len, idx, doc_id, filename):
    text = " ".join(current).strip()
    if text:
        chunks.append(_make_chunk(text, idx, doc_id, filename))


def _make_chunk(text: str, idx: int, doc_id: str, filename: str) -> dict:
    return {
        "chunk_id":       f"{doc_id}_chunk_{idx}",
        "doc_id":         doc_id,
        "filename":       filename,
        "chunk_index":    idx,
        "text":           text,
        "token_estimate": len(text) // _CHARS_PER_TOKEN,
    }
