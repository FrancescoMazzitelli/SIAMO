"""Split a paragraph into overlapping chunks for chunk-level retrieval."""

import re

from src.core.config import get_settings
from src.core.models.document import (
    ProcessedChunk,
    HierarchicalMetadata,
    Level,
)


class ParagraphChunker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.max_chars = self.settings.chunk_max_chars
        self.overlap = self.settings.chunk_overlap

    def chunk(
        self,
        paragraph_id: str,
        doc_id: str,
        paragraph_index: int,
        text: str,
        page_index: int = 0,
    ) -> list[ProcessedChunk]:
        """Chunk a paragraph; a short paragraph stays a single chunk."""
        if len(text) <= self.max_chars:
            return [
                ProcessedChunk(
                    text=text,
                    paragraph_id=paragraph_id,
                    doc_id=doc_id,
                    chunk_index=0,
                    metadata=HierarchicalMetadata(
                        doc_id=doc_id,
                        paragraph_id=paragraph_id,
                        level=Level.CHUNK,
                        paragraph_index=paragraph_index,
                        chunk_index=0,
                        total_chunks_in_paragraph=1,
                        page_index=page_index,
                    ),
                )
            ]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[ProcessedChunk] = []
        current = ""
        idx = 0

        def make_chunk(chunk_text: str, chunk_index: int) -> ProcessedChunk:
            return ProcessedChunk(
                text=chunk_text.strip(),
                paragraph_id=paragraph_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                metadata=HierarchicalMetadata(
                    doc_id=doc_id,
                    paragraph_id=paragraph_id,
                    level=Level.CHUNK,
                    paragraph_index=paragraph_index,
                    chunk_index=chunk_index,
                    page_index=page_index,
                ),
            )

        for sentence in sentences:
            if len(current) + len(sentence) > self.max_chars and current:
                chunks.append(make_chunk(current, idx))
                overlap_text = current[-self.overlap :] if len(current) > self.overlap else current
                current = overlap_text + " " + sentence
                idx += 1
            else:
                current = (current + " " + sentence).strip() if current else sentence

        if current.strip():
            chunks.append(make_chunk(current, idx))

        for chunk in chunks:
            chunk.metadata.total_chunks_in_paragraph = len(chunks)

        return chunks