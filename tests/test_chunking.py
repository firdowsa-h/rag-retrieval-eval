import pytest

from app.chunking import fixed_chunks, paragraph_chunks
from app.documents import Section


@pytest.fixture
def section():
    return Section("pg18:test", "Test", "https://example.org/test", "one two three four\n\nfive six seven eight")


def test_fixed_chunks_are_traceable(section):
    chunks = fixed_chunks(section, size=4, overlap=1)
    assert len(chunks) >= 2
    assert all(chunk.section_id == section.section_id for chunk in chunks)
    assert chunks[0].chunk_id == "pg18:test:fixed:0000"


def test_paragraph_chunks_keep_source(section):
    chunks = paragraph_chunks(section, size=4)
    assert len(chunks) == 2
    assert chunks[1].text == "five six seven eight"
    assert chunks[0].url == section.url


def test_bad_overlap_rejected(section):
    with pytest.raises(ValueError):
        fixed_chunks(section, size=4, overlap=4)
