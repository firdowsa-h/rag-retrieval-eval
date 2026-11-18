from app.documents import Section, load_sections, save_sections


def test_section_round_trip(tmp_path):
    source = [Section("pg18:example", "A title", "https://example.org", "Some text")]
    destination = tmp_path / "sections.jsonl"
    save_sections(source, destination)
    assert load_sections(destination) == source
