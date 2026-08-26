"""Deterministic, local-only text extraction for synchronized sources."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar, Literal


class ExtractionError(RuntimeError):
    """Raised when a synchronized source cannot be converted to text."""


class _VisibleTextParser(HTMLParser):
    _HIDDEN: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "noscript", "svg", "template"}
    )
    _BREAKS: ClassVar[frozenset[str]] = frozenset(
        {
            "article",
            "blockquote",
            "br",
            "div",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "li",
            "main",
            "p",
            "section",
            "table",
            "td",
            "th",
            "tr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in self._HIDDEN:
            self._hidden_depth += 1
        elif self._hidden_depth == 0 and tag in self._BREAKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._HIDDEN and self._hidden_depth:
            self._hidden_depth -= 1
        elif self._hidden_depth == 0 and tag in self._BREAKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0:
            self.parts.append(data)


class _ArticleTextParser(HTMLParser):
    """Capture semantic article regions without page chrome or related cards."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self._frames: list[list[str]] = []
        self._candidates: list[tuple[int, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in _VisibleTextParser._HIDDEN:
            self._hidden_depth += 1
            return
        if self._hidden_depth:
            return
        if tag == "article":
            self._frames.append([])
        if tag in _VisibleTextParser._BREAKS:
            for parts in self._frames:
                parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VisibleTextParser._HIDDEN:
            if self._hidden_depth:
                self._hidden_depth -= 1
            return
        if self._hidden_depth:
            return
        if tag in _VisibleTextParser._BREAKS:
            for parts in self._frames:
                parts.append("\n")
        if tag == "article" and self._frames:
            depth = len(self._frames) - 1
            self._candidates.append((depth, self._frames.pop()))

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0:
            for parts in self._frames:
                parts.append(data)

    def article_text(self) -> str:
        candidates = [
            *self._candidates,
            *((index, parts) for index, parts in enumerate(self._frames)),
        ]
        if not candidates:
            raise ExtractionError("HTML source has no semantic <article> region")
        normalized = [(depth, _normalize_text("".join(parts))) for depth, parts in candidates]
        nonempty = [(depth, text) for depth, text in normalized if text]
        if not nonempty:
            raise ExtractionError("semantic <article> regions contain no visible text")
        deepest = max(depth for depth, _text in nonempty)
        return max(
            (text for depth, text in nonempty if depth == deepest),
            key=len,
        )


ContentScope = Literal["document", "article"]


def extract_text(
    path: Path,
    media_type: str,
    *,
    content_scope: ContentScope = "document",
) -> str:
    if content_scope not in {"document", "article"}:
        raise ExtractionError(f"unsupported content scope: {content_scope!r}")
    if media_type == "application/pdf":
        if content_scope != "document":
            raise ExtractionError("PDF extraction supports only document scope")
        return _extract_pdf(path)
    try:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ExtractionError(f"cannot decode {path} as UTF-8: {exc}") from exc
    if media_type == "text/html":
        parser = _ArticleTextParser() if content_scope == "article" else _VisibleTextParser()
        try:
            parser.feed(text)
            parser.close()
        except Exception as exc:  # HTMLParser can surface malformed character refs.
            raise ExtractionError(f"cannot parse HTML {path}: {exc}") from exc
        text = (
            parser.article_text()
            if isinstance(parser, _ArticleTextParser)
            else "".join(parser.parts)
        )
    elif content_scope != "document":
        raise ExtractionError("article content scope requires an HTML source")
    return _normalize_text(text)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific.
        raise ExtractionError("PDF extraction requires the 'research' dependency profile") from exc
    try:
        reader = PdfReader(path, strict=False)
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise ExtractionError(f"cannot extract PDF {path}: {exc}") from exc
    return _normalize_text("\n\n\f\n\n".join(pages))


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized + ("\n" if normalized else "")
