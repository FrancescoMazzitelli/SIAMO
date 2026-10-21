"""Strip running page headers/footers produced by OCR.

LightOnOCR returns the running head of a page as its first line (for example
"Chapter 1 FIRST AID 21" or "# International Medical Guide for Ships") and, in
some layouts, the page number as the last line ("229"). If they are kept, the
header is glued to the last paragraph of the previous page and both pollute the
embeddings and the retrieved context.

The first line is dropped when it matches a page-number/chapter pattern or when
the same line (ignoring the digits, which change with the page number) has
already been seen in the document. The first occurrence of an otherwise
unrecognised running head is kept, which leaves at most one stray line per
manual. The last line is dropped only when it is a bare page number, so real
trailing content (figure captions, sentences) is never removed.
"""

import re
from collections import Counter

_DIGITS = re.compile(r"\d+")
_LEADING_MARKERS = re.compile(r"^[#*\s]+")
_WHITESPACE = re.compile(r"\s+")

_BARE_NUMBER = re.compile(r"^\d{1,4}$")
_NUMBER_FIRST = re.compile(r"^\d{1,4}\s+\S")
_CHAPTER = re.compile(r"^chapter\s+[\d.]+", re.IGNORECASE)


def _core(text: str) -> str:
    return _LEADING_MARKERS.sub("", _WHITESPACE.sub(" ", text).strip())


def normalize_header(text: str) -> str:
    """Collapse a candidate line so page-number variants compare equal."""
    return _DIGITS.sub("#", _core(text).lower())


def looks_like_header(text: str) -> bool:
    core = _core(text)
    if not core:
        return True
    if _BARE_NUMBER.match(core):
        return True
    if _NUMBER_FIRST.match(core):
        return True
    if _CHAPTER.match(core):
        return True
    return False


def is_page_number(text: str) -> bool:
    return bool(_BARE_NUMBER.match(_core(text)))


class PageFurnitureFilter:
    """Per-document filter that drops running heads and numeric footers."""

    def __init__(self) -> None:
        self._seen: Counter[str] = Counter()
        self._removed = 0

    @property
    def removed(self) -> int:
        return self._removed

    def clean(self, markdown: str) -> str:
        lines = markdown.split("\n")
        self._strip_header(lines)
        self._strip_footer(lines)
        return "\n".join(lines)

    def _strip_header(self, lines: list[str]) -> None:
        index = next((i for i, line in enumerate(lines) if line.strip()), None)
        if index is None:
            return

        candidate = lines[index]
        key = normalize_header(candidate)
        is_furniture = self._seen[key] >= 1 or looks_like_header(candidate)
        self._seen[key] += 1
        if is_furniture:
            del lines[index]
            self._removed += 1

    def _strip_footer(self, lines: list[str]) -> None:
        index = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
        if index is None:
            return
        if is_page_number(lines[index]):
            del lines[index]
            self._removed += 1


def clean_pages(pages: list[str]) -> list[str]:
    """Apply PageFurnitureFilter to a whole list of pages (convenience/testing)."""
    filter_ = PageFurnitureFilter()
    return [filter_.clean(page) for page in pages]
