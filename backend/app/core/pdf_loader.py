"""PDF loading + text normalization.

Uses LangChain's :class:`PyPDFLoader` under the hood, then applies light
cleaning to the extracted text so the splitter produces cleaner chunks:

* de-hyphenate line-break word splits (``"docu-\\nment" -> "document"``)
* collapse runs of newlines / trailing whitespace

Every page becomes a :class:`DocumentPage` with 1-indexed page numbers and
the cleaned text.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class DocumentPage:
    """A single page extracted from a PDF."""
    source: str        # basename of the PDF, e.g. "c06584210.pdf"
    page: int          # 1-indexed page number
    text: str          # cleaned page text (may be empty)


_HYPHEN_LINEBREAK = re.compile(r"-\n(?=\w)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _TRAILING_SPACES.sub("\n", text)
    return text.strip()


def load_pdf(path: str) -> List[DocumentPage]:
    """Load one PDF and return its cleaned pages.

    Parameters
    ----------
    path:
        Path to the PDF file on disk.

    Returns
    -------
    List[DocumentPage]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Import lazily so unit tests can monkeypatch without importing LangChain
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(path)
    docs = loader.load()

    source = os.path.basename(path)
    out: List[DocumentPage] = []
    for i, d in enumerate(docs, start=1):
        # LangChain sets ``metadata['page']`` starting at 0; prefer that when present.
        page_num = d.metadata.get("page")
        page_num = int(page_num) + 1 if page_num is not None else i
        out.append(DocumentPage(source=source, page=page_num, text=_clean(d.page_content or "")))
    return out


def load_pdfs(paths: Iterable[str]) -> List[DocumentPage]:
    """Load multiple PDFs; results are concatenated in the given order."""
    pages: List[DocumentPage] = []
    for p in paths:
        pages.extend(load_pdf(p))
    return pages
