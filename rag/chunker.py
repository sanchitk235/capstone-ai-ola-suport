"""
rag/chunker.py - chunking stratagies for ola knowledge docs.
A: fixed size (200 char window, 40 overlap)
B: sentence split (2 sentences per chunk)
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    topic: str
    text: str
    strategy: str  # "fixed" | "sentence"


# strat A: fixed sliding window
def fixed_size_chunks(
    doc: dict,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """slides a char window across doc text with overlap"""
    text = doc["content"]
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}_fixed_{idx}",
                    doc_id=doc["doc_id"],
                    topic=doc["topic"],
                    text=chunk_text,
                    strategy="fixed",
                )
            )
            idx += 1
        start += step

    return chunks


# strat B: sentence grouping
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def sentence_chunks(
    doc: dict,
    sentences_per_chunk: int = 2,
) -> list[Chunk]:
    """breaks text into sentences and groups them together"""
    text = doc["content"]
    sentences = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]

    chunks: list[Chunk] = []
    idx = 0
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(group)
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}_sent_{idx}",
                    doc_id=doc["doc_id"],
                    topic=doc["topic"],
                    text=chunk_text,
                    strategy="sentence",
                )
            )
            idx += 1

    return chunks


# chunk helper
def chunk_all_documents(
    kb_documents: list[dict],
) -> tuple[list[Chunk], list[Chunk]]:
    """chunks all docs using both methods"""
    fixed: list[Chunk] = []
    sent: list[Chunk] = []
    for doc in kb_documents:
        fixed.extend(fixed_size_chunks(doc))
        sent.extend(sentence_chunks(doc))
    return fixed, sent
