from pathlib import Path

from recall import chunker


def test_build_chunk_basic() -> None:
    chunk = chunker.build_chunk(
        "Alice lives in Seattle and works at Acme Corp",
        "identity.md",
        "identity.md",
    )
    assert chunk is not None
    assert "Alice lives in Seattle" in chunk.content
    assert chunk.source_file == "identity.md"


def test_build_chunk_filters_short() -> None:
    chunk = chunker.build_chunk("too short", "test", "test")
    assert chunk is None


def test_build_chunk_adds_section_path() -> None:
    chunk = chunker.build_chunk(
        "A fact long enough to pass the minimum character threshold",
        "section > subsection",
        "test.md",
    )
    assert chunk is not None
    assert chunk.content.startswith("section > subsection\n\n")


def test_list_supported_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "c.pdf").write_bytes(b"fake pdf")
    (tmp_path / "d.txt").write_text("ignored")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "e.md").write_text("nested")

    files = chunker.list_supported_files(tmp_path)
    names = [f.name for f in files]
    assert "a.md" in names
    assert "b.json" in names
    assert "c.pdf" in names
    assert "e.md" in names
    assert "d.txt" not in names


def test_read_file_text_markdown(tmp_path: Path) -> None:
    md = tmp_path / "test.md"
    md.write_text("# Title\n\nSome content here.\n")

    text = chunker.read_file_text(md)
    assert "Title" in text
    assert "Some content" in text


def test_read_file_text_json(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_text('{"key": "value"}')

    text = chunker.read_file_text(f)
    assert '"key"' in text


def test_read_file_text_pdf(tmp_path: Path) -> None:
    import pymupdf

    path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF content for testing")
    doc.save(str(path))
    doc.close()

    text = chunker.read_file_text(path)
    assert "PDF content" in text


def test_chunk_hashes_are_unique() -> None:
    c1 = chunker.build_chunk("First fact about something important and meaningful enough", "s", "f")
    c2 = chunker.build_chunk("Second fact about something entirely different from the first", "s", "f")
    assert c1 is not None and c2 is not None
    assert c1.chunk_hash != c2.chunk_hash


def test_parse_fact_date_full() -> None:
    assert chunker.parse_fact_date("[2025-07-11] Alice was born") == "2025-07-11"


def test_parse_fact_date_month() -> None:
    assert chunker.parse_fact_date("[2024-08] Shopping trip") == "2024-08"


def test_parse_fact_date_year() -> None:
    assert chunker.parse_fact_date("[2025] Something happened this year") == "2025"


def test_parse_fact_date_quarter() -> None:
    assert chunker.parse_fact_date("[2024-Q2] OKR review") == "2024-Q2"


def test_parse_fact_date_half() -> None:
    assert chunker.parse_fact_date("[2025-H2] Second half plans") == "2025-H2"


def test_parse_fact_date_range() -> None:
    assert chunker.parse_fact_date("[2024-11 to 2025-02] Winter project") == "2024-11 to 2025-02"


def test_parse_fact_date_day_range() -> None:
    assert chunker.parse_fact_date("[2023-11-11 to 2023-11-13] Weekend trip") == "2023-11-11 to 2023-11-13"


def test_parse_fact_date_none() -> None:
    assert chunker.parse_fact_date("No date here") is None


def test_parse_fact_date_undated() -> None:
    assert chunker.parse_fact_date("[Undated] Some fact") is None


def test_parse_fact_date_category_only() -> None:
    assert chunker.parse_fact_date("[Career - Acme Corp] Some fact") is None


def test_parse_fact_date_with_leading_whitespace() -> None:
    assert chunker.parse_fact_date("  [2025-01-15] Trimmed") == "2025-01-15"


def test_build_chunk_extracts_fact_date() -> None:
    chunk = chunker.build_chunk(
        "[2025-07-11] [Personal] Alice Johnson was born on July 11 2000",
        "test.md",
        "test.md",
    )
    assert chunk is not None
    assert chunk.fact_date == "2025-07-11"


def test_build_chunk_no_date() -> None:
    chunk = chunker.build_chunk(
        "Alice lives in Seattle and works at Acme Corp as a senior SWE",
        "test.md",
        "test.md",
    )
    assert chunk is not None
    assert chunk.fact_date is None
