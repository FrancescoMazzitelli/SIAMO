"""Split OCR Markdown into paragraphs, one per heading section.

Rule (as requested): text is accumulated until a heading of any level
(#, ##, ###, ####, #####, ######) is found. The heading closes the current
paragraph and opens a new one, heading included. Text before the first heading
belongs to the first paragraph; a paragraph may span page breaks.
"""

import re
from dataclasses import dataclass

_HEADING_LINE = re.compile(r"^\s*#{1,6}\s+\S")
_FENCE_LINE = re.compile(r"^\s*```\s*$")


@dataclass
class ParagraphSegment:
    """One extracted paragraph plus the page where it begins."""

    text: str
    page_index: int


class MarkdownSplitter:
    def __init__(self, min_paragraph_chars: int = 80) -> None:
        self.min_paragraph_chars = min_paragraph_chars

    def split(self, page_markdowns: list[str]) -> list[ParagraphSegment]:
        segments: list[ParagraphSegment] = []
        current: list[str] = []
        current_page = 0

        for page_index, markdown in enumerate(page_markdowns):
            for raw_line in markdown.split("\n"):
                if _HEADING_LINE.match(raw_line):
                    self._flush(segments, current, current_page)
                    current = [raw_line.strip()]
                    current_page = page_index
                else:
                    current.append(raw_line)

        self._flush(segments, current, current_page)

        cleaned: list[ParagraphSegment] = []
        for segment in segments:
            lines = [line for line in segment.text.split("\n") if not _FENCE_LINE.match(line)]
            text = "\n".join(lines).strip()
            if text:
                cleaned.append(ParagraphSegment(text=text, page_index=segment.page_index))

        return [p for p in cleaned if len(p.text) >= self.min_paragraph_chars]

    @staticmethod
    def _flush(segments: list[ParagraphSegment], current: list[str], page_index: int) -> None:
        if current:
            segments.append(ParagraphSegment(text="\n".join(current), page_index=page_index))