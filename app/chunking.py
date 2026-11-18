"""Two deliberately simple chunkers for a controlled first comparison."""

from dataclasses import dataclass
import re

from app.documents import Section


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section_id: str
    title: str
    url: str
    text: str


def _chunks(section: Section, parts: list[str], strategy: str) -> list[Chunk]:
    return [Chunk(
        chunk_id=f"{section.section_id}:{strategy}:{i:04d}",
        section_id=section.section_id,
        title=section.title,
        url=section.url,
        text=part,
    ) for i, part in enumerate(parts) if part.strip()]


def fixed_chunks(section: Section, size: int = 300, overlap: int = 50) -> list[Chunk]:
    """Whitespace tokens, not model tokens. Named explicitly to avoid false precision."""
    if not (0 <= overlap < size):
        raise ValueError("overlap must be nonnegative and smaller than size")
    words = section.text.split()
    step = size - overlap
    return _chunks(section, [" ".join(words[i:i + size]) for i in range(0, len(words), step)], "fixed")


def paragraph_chunks(section: Section, size: int = 300) -> list[Chunk]:
    """Keep paragraph boundaries when possible; split long paragraphs on words."""
    if size <= 0:
        raise ValueError("size must be positive")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
    output: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            output.append(" ".join(pending))
            pending.clear()

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > size:
            flush()
            output.extend(" ".join(words[i:i + size]) for i in range(0, len(words), size))
        elif len(pending) + len(words) > size:
            flush()
            pending.extend(words)
        else:
            pending.extend(words)
    flush()
    return _chunks(section, output, "paragraph")
