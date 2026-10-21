"""Split document text into paragraphs (the mid level of the hierarchy).

Text is split on Markdown headings (#, ##, ...) and on blank lines. A heading
always starts a new paragraph, so a section and its content stay together instead
of being cut after the first short line. Runs of blank-line separated text that
have no heading are grouped until they reach min_paragraph_chars, and any
paragraph longer than max_paragraph_chars is split further (with a hard split for
text that has no sentence punctuation at all), so the parent paragraph recovered
by the retrieval always fits in the LLM context.

MarkdownSplitter.split() processes a whole list of pages at once, while
MarkdownSplitter.new_stream() returns a ParagraphStream that accepts one page at
a time and yields each paragraph as soon as it is closed, so the ingest can embed
and index paragraphs while the remaining pages are still being read.
"""

import re
from dataclasses import dataclass

_HEADING_LINE = re.compile(r"^\s*#{1,6}\s+\S")
_FENCE_LINE = re.compile(r"^\s*```\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class ParagraphSegment:
    """One extracted paragraph plus the page where it begins."""

    text: str
    page_index: int


class MarkdownSplitter:
    def __init__(self, min_paragraph_chars: int = 80, max_paragraph_chars: int = 1800) -> None:
        self.min_paragraph_chars = min_paragraph_chars
        self._min = max(min_paragraph_chars, 1)
        self._max = max(max_paragraph_chars, min_paragraph_chars + 1)

    def new_stream(self) -> "ParagraphStream":
        """Return an incremental builder that consumes pages one at a time."""
        return ParagraphStream(self)

    def split(self, page_markdowns: list[str]) -> list[ParagraphSegment]:
        stream = self.new_stream()
        segments: list[ParagraphSegment] = []
        for page_index, markdown in enumerate(page_markdowns):
            segments.extend(stream.feed(markdown, page_index))
        segments.extend(stream.close())
        return segments

    def _cap(self, text: str, page_index: int) -> list[ParagraphSegment]:
        if len(text) <= self._max:
            return [ParagraphSegment(text=text, page_index=page_index)]

        pieces: list[str] = []
        buffer = ""
        for sentence in _SENTENCE_END.split(text):
            if buffer and len(buffer) + len(sentence) + 1 > self._max:
                pieces.append(buffer.strip())
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            pieces.append(buffer.strip())

        result: list[str] = []
        for piece in pieces:
            while len(piece) > self._max:
                result.append(piece[: self._max].strip())
                piece = piece[self._max :].strip()
            if piece:
                result.append(piece)
        return [ParagraphSegment(text=piece, page_index=page_index) for piece in result]


class ParagraphStream:
    """Incremental paragraph builder driven by MarkdownSplitter.

    feed() consumes one page and returns the paragraphs completed by that page;
    close() flushes whatever is still open at the end of the document.
    """

    def __init__(self, splitter: MarkdownSplitter) -> None:
        self._splitter = splitter
        self._blocks: list[str] = []
        self._first_page = 0
        self._length = 0
        self._has_body = False
        self._first_is_heading = False

    def feed(self, markdown: str, page_index: int) -> list[ParagraphSegment]:
        ready: list[ParagraphSegment] = []
        for text, is_heading in self._iter_blocks(markdown):
            if self._should_flush(is_heading, len(text)):
                ready.extend(self._flush())
            if not self._blocks:
                self._first_is_heading = is_heading
                self._first_page = page_index
            self._blocks.append(text)
            self._length += len(text)
            if not is_heading:
                self._has_body = True
        return ready

    def close(self) -> list[ParagraphSegment]:
        return self._flush()

    def _should_flush(self, is_heading: bool, incoming: int) -> bool:
        if not self._blocks:
            return False
        if is_heading and self._has_body:
            return True
        if not self._first_is_heading and self._length >= self._splitter._min:
            return True
        if self._length + incoming > self._splitter._max:
            return True
        return False

    def _flush(self) -> list[ParagraphSegment]:
        if not self._blocks:
            return []
        text = "\n".join(self._blocks)
        page_index = self._first_page
        self._blocks = []
        self._length = 0
        self._has_body = False
        self._first_is_heading = False
        return self._splitter._cap(text, page_index)

    @staticmethod
    def _iter_blocks(markdown: str) -> list[tuple[str, bool]]:
        raw_blocks: list[str] = []
        current: list[str] = []
        for raw_line in markdown.split("\n"):
            line = raw_line.rstrip()
            if line.strip():
                current.append(line)
                continue
            if current:
                raw_blocks.append("\n".join(current))
                current = []
        if current:
            raw_blocks.append("\n".join(current))

        blocks: list[tuple[str, bool]] = []
        for raw_block in raw_blocks:
            lines = [line for line in raw_block.split("\n") if not _FENCE_LINE.match(line)]
            text = "\n".join(lines).strip()
            if text:
                blocks.append((text, bool(_HEADING_LINE.match(text))))
        return blocks
