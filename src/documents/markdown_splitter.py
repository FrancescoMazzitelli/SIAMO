"""Split the Markdown produced by the OCR vision model into paragraphs.

Paragraph rule (as requested): text is accumulated, block after block, until a
new paragraph is encountered (blank line in the Markdown) or the end of the
document is reached.

Page breaks are handled with a single heuristic: a paragraph cut mid-sentence by
a page break is re-joined with the FIRST block of the following page, unless
that block starts a new structural element (heading / table / list item / code
block). Blocks separated by a blank line within the same page always stay
separate paragraphs. Markdown headings are attached to the paragraph that
follows them so that no heading context is lost.
"""

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"[.!?;:]\s*$")
_HEADING_START = re.compile(r"^#{1,6}\s+\S")
_STRUCTURE_START = re.compile(r"^(?:\|)|^(?:[-*+]\s+|\d+[.)]\s+)|(?:```)")
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
        """Split per-page Markdown into paragraphs.
        Args:
            page_markdowns: one Markdown string per document page, in page order.
        """
        paragraphs: list[ParagraphSegment] = []
        pending: ParagraphSegment | None = None
        pending_heading: str | None = None
        for page_index, markdown in enumerate(page_markdowns):
            first_content_of_page = True
            for raw_block, is_code in self._split_blocks(markdown):
                block = self._clean_block(raw_block)
                if not block:
                    continue
                if _HEADING_START.match(block):
                    lines = block.split("\n")
                    pending_heading = lines[0]
                    rest = "\n".join(lines[1:]).strip()
                    if not rest:
                        continue
                    block = rest
                if pending is None:
                    pending = self._make_segment(pending_heading, block, page_index)
                    pending_heading = None
                    first_content_of_page = False
                    continue
                if (
                    first_content_of_page
                    and not is_code
                    and self._continues_paragraph(pending.text, block)
                ):
                    pending.text += "\n\n" + block
                else:
                    paragraphs.append(pending)
                    pending = self._make_segment(pending_heading, block, page_index)
                    pending_heading = None
                first_content_of_page = False
        if pending is not None:
            paragraphs.append(pending)
        return [p for p in paragraphs if len(p.text) >= self.min_paragraph_chars]
    @staticmethod
    def _make_segment(heading: str | None, block: str, page_index: int) -> ParagraphSegment:
        text = f"{heading}\n{block}" if heading else block
        return ParagraphSegment(text=text, page_index=page_index)
    @staticmethod
    def _continues_paragraph(previous: str, current: str) -> bool:
        return (
            not _SENTENCE_END.search(previous)
            and not _STRUCTURE_START.match(current)
        )
    @staticmethod
    def _split_blocks(markdown: str) -> list[tuple[str, bool]]:
        """Split on blank lines, keeping fenced code blocks intact.
        Returns (block_text, is_fenced_code) pairs.
        """
        blocks: list[tuple[str, bool]] = []
        current: list[str] = []
        in_code = False
        for line in markdown.split("\n"):
            if line.strip().startswith("```"):
                in_code = not in_code
                current.append(line)
                continue
            if not in_code and not line.strip():
                if current:
                    blocks.append(("\n".join(current), False))
                    current = []
                continue
            current.append(line)
        if current:
            blocks.append(("\n".join(current), in_code))
        return blocks
    @staticmethod
    def _clean_block(block: str) -> str:
        """Strip code fences and whitespace noise."""
        lines = [line for line in block.split("\n") if not _FENCE_LINE.match(line)]
        return "\n".join(lines).strip()
