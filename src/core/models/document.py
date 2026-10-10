from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
import uuid


class Level(str, Enum):
    DOCUMENT = "document"
    PARAGRAPH = "paragraph"
    CHUNK = "chunk"


class HierarchicalMetadata(BaseModel):
    doc_id: str
    paragraph_id: str = ""
    chunk_id: str = ""
    level: Level
    doc_source: str = ""
    doc_summary: str = ""
    paragraph_index: int = 0
    chunk_index: int = 0
    total_chunks_in_paragraph: int = 0
    page_index: int = 0


class ProcessedChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    paragraph_id: str
    doc_id: str
    chunk_index: int
    metadata: HierarchicalMetadata


class ProcessedParagraph(BaseModel):
    paragraph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    full_text: str
    chunks: list[ProcessedChunk] = Field(default_factory=list)
    doc_id: str
    paragraph_index: int
    page_index: int = 0


class ProcessedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    raw_text: str
    summary: str = ""
    paragraphs: list[ProcessedParagraph] = Field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(len(p.chunks) for p in self.paragraphs)

    @property
    def total_paragraphs(self) -> int:
        return len(self.paragraphs)
