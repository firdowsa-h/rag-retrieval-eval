from app.fetch_docs import parse_page


def test_parser_extracts_main_content():
    html = "<html><body><div id='docContent'><h1>Indexes</h1><p>" + ("B-tree indexes can support searches. " * 8) + "</p></div></body></html>"
    result = parse_page(html, "https://example.org", "indexes.html")
    assert result.section_id == "pg18:indexes.html"
    assert "B-tree" in result.text
