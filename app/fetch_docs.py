"""Fetch a fixed list of PostgreSQL 18 manual pages.

The allowlist is intentional: it is easier to audit and reproduce than a crawler.
Keep the output snapshot for later experiments.
"""

from pathlib import Path
import argparse
import re

from bs4 import BeautifulSoup
import httpx

from app.documents import Section, save_sections

DOC_PAGES = [
    "tutorial-sql-intro.html",
    "tutorial-table.html",
    "tutorial-populate.html",
    "tutorial-select.html",
    "tutorial-join.html",
    "tutorial-agg.html",
    "ddl-basics.html",
    "ddl-constraints.html",
    "ddl-default.html",
    "ddl-generated-columns.html",
    "ddl-schemas.html",
    "ddl-priv.html",
    "queries-overview.html",
    "queries-table-expressions.html",
    "queries-select-lists.html",
    "queries-order.html",
    "queries-limit.html",
    "queries-with.html",
    "indexes-intro.html",
    "indexes-types.html",
    "indexes-multicolumn.html",
    "indexes-unique.html",
    "indexes-expressional.html",
    "indexes-partial.html",
    "textsearch-intro.html",
    "textsearch-tables.html",
    "transaction-iso.html",
    "explicit-locking.html",
    "routine-vacuuming.html",
    "performance-tips.html",
]


def parse_page(html: str, url: str, slug: str) -> Section:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("#docContent") or soup.select_one(".sect1")
    if main is None:
        raise ValueError(f"No documentation body found at {url}; inspect the page structure")
    for tag in main.select(".navheader, .navfooter, .toc, script, style"):
        tag.decompose()
    title_tag = main.select_one("h1, h2") or soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else slug
    text = re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True))
    if len(text) < 150:
        raise ValueError(f"Unexpectedly short page at {url} ({len(text)} characters)")
    return Section(section_id=f"pg18:{slug}", title=title, url=url, text=text)


def fetch_all() -> list[Section]:
    sections = []
    with httpx.Client(timeout=30, follow_redirects=True,
                      headers={"User-Agent": "pg-docs-rag-evaluation/0.1 (learning project)"}) as client:
        for slug in DOC_PAGES:
            url = f"https://www.postgresql.org/docs/18/{slug}"
            response = client.get(url)
            response.raise_for_status()
            sections.append(parse_page(response.text, url, slug))
            print(f"Fetched {slug}")
    return sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/postgres18_sections.jsonl"))
    args = parser.parse_args()
    sections = fetch_all()
    save_sections(sections, args.output)
    print(f"Saved {len(sections)} documents to {args.output}")


if __name__ == "__main__":
    main()
