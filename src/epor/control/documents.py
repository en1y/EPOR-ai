"""Read-only index of the project's own tracked Markdown documentation.

This is deliberately not a filesystem browser.  The index is built by globbing
a fixed pair of first-party locations, and a request is answered by an exact
slug lookup inside that index, so no caller-supplied value is ever joined onto
a path.  Traversal has nothing to traverse.

Ignored third-party research downloads under ``research/raw/`` are excluded on
purpose.  Rendering unreviewed external text as console content would make it a
new instruction surface; the research ledger links those sources at their
canonical URL instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_DOCUMENT_BYTES = 512 * 1024


class DocumentTooLargeError(ValueError):
    """Raised when a tracked document exceeds the console read limit."""


@dataclass(frozen=True, slots=True)
class Document:
    """One tracked Markdown file the console is allowed to render."""

    slug: str
    title: str
    group: str
    path: Path


def _title(path: Path) -> str:
    """Return the first Markdown heading, falling back to the file name."""

    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        pass
    return path.stem.replace("-", " ").replace("_", " ")


def build_index(project_root: Path) -> dict[str, Document]:
    """Map slug to document for every tracked Markdown file under the root."""

    root = project_root.expanduser().resolve()
    paths = sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
    index: dict[str, Document] = {}
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parent = relative.parent.as_posix()
        index[relative.as_posix()] = Document(
            slug=relative.as_posix(),
            title=_title(path),
            group="" if parent == "." else parent,
            path=path,
        )
    return index


def read_document(document: Document) -> str:
    """Read one indexed document, refusing anything past the read limit."""

    if document.path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise DocumentTooLargeError(f"{document.slug} exceeds the console read limit")
    return document.path.read_text(encoding="utf-8")
