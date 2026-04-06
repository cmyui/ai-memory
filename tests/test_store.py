import numpy as np
import pytest
import pytest_asyncio

from recall import settings as settings_module
from recall import store


@pytest_asyncio.fixture
async def db(tmp_path):
    settings_module.settings.db_path = tmp_path / "test.db"
    database = await store.get_database()
    yield database
    await database.disconnect()


def _make_embedding() -> np.ndarray:
    rng = np.random.default_rng(42)
    vec = rng.random(settings_module.settings.embedding_dimensions, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.mark.asyncio
async def test_upsert_and_get_chunks(db) -> None:
    emb = _make_embedding()
    chunks = [
        {
            "content": "Test content",
            "section_path": "Section A",
            "chunk_hash": "hash1",
            "embedding": emb,
        },
    ]

    count = await store.upsert_chunks(db, chunks=chunks, source_file="test.md", file_hash="filehash1")
    assert count == 1

    all_chunks = await store.get_all_chunks(db)
    assert len(all_chunks) == 1
    assert all_chunks[0].content == "Test content"
    assert all_chunks[0].source_type == "ingest"
    assert all_chunks[0].source_file == "test.md"
    assert all_chunks[0].fact_date is None


@pytest.mark.asyncio
async def test_upsert_with_source_type(db) -> None:
    emb = _make_embedding()
    chunks = [
        {
            "content": "Dream-generated fact about someone",
            "section_path": "Peer Card: Test",
            "chunk_hash": "hash_dream",
            "embedding": emb,
        },
    ]

    await store.upsert_chunks(
        db, chunks=chunks, source_file="peer-card-facts/Test", file_hash="fh", source_type="dream",
    )

    all_chunks = await store.get_all_chunks(db)
    assert len(all_chunks) == 1
    assert all_chunks[0].source_type == "dream"


@pytest.mark.asyncio
async def test_upsert_with_fact_date(db) -> None:
    emb = _make_embedding()
    chunks = [
        {
            "content": "[2025-07-11] Alice was born on July 11, 2000",
            "section_path": "S",
            "chunk_hash": "h1",
            "embedding": emb,
            "fact_date": "2025-07-11",
        },
    ]

    await store.upsert_chunks(db, chunks=chunks, source_file="test.md", file_hash="fh")

    all_chunks = await store.get_all_chunks(db)
    assert len(all_chunks) == 1
    assert all_chunks[0].fact_date == "2025-07-11"


@pytest.mark.asyncio
async def test_upsert_replaces_old_chunks(db) -> None:
    emb = _make_embedding()

    await store.upsert_chunks(
        db,
        chunks=[{"content": "Old", "section_path": "S", "chunk_hash": "old", "embedding": emb}],
        source_file="test.md",
        file_hash="hash_v1",
    )

    await store.upsert_chunks(
        db,
        chunks=[{"content": "New", "section_path": "S", "chunk_hash": "new", "embedding": emb}],
        source_file="test.md",
        file_hash="hash_v2",
    )

    all_chunks = await store.get_all_chunks(db)
    assert len(all_chunks) == 1
    assert all_chunks[0].content == "New"


@pytest.mark.asyncio
async def test_insert_chunk_defaults(db) -> None:
    emb = _make_embedding()
    row_id = await store.insert_chunk(db, content="A remembered fact", chunk_hash="remhash", embedding=emb)
    assert row_id > 0

    all_chunks = await store.get_all_chunks(db)
    assert len(all_chunks) == 1
    assert all_chunks[0].source_type == "ingest"
    assert all_chunks[0].source_file is None


@pytest.mark.asyncio
async def test_insert_chunk_with_source_type(db) -> None:
    emb = _make_embedding()
    row_id = await store.insert_chunk(
        db,
        content="AI remembered this fact about Alice",
        chunk_hash="ai_hash",
        embedding=emb,
        source_type="remember:ai",
        fact_date="2026-04-05",
    )
    assert row_id > 0

    all_chunks = await store.get_all_chunks(db)
    assert len(all_chunks) == 1
    assert all_chunks[0].source_type == "remember:ai"
    assert all_chunks[0].fact_date == "2026-04-05"


@pytest.mark.asyncio
async def test_get_all_chunks_exclude_source_types(db) -> None:
    emb = _make_embedding()

    await store.upsert_chunks(
        db,
        chunks=[{"content": "Ingested fact", "section_path": "S", "chunk_hash": "f1", "embedding": emb}],
        source_file="test.md",
        file_hash="fh1",
    )
    await store.insert_chunk(
        db, content="Dream-generated peer card fact", chunk_hash="d1", embedding=emb, source_type="dream",
    )
    await store.insert_chunk(
        db, content="AI remembered fact about something", chunk_hash="r1", embedding=emb, source_type="remember:ai",
    )

    # All chunks
    all_chunks = await store.get_all_chunks(db)
    assert len(all_chunks) == 3

    # Exclude dream
    no_dream = await store.get_all_chunks(db, exclude_source_types={"dream"})
    assert len(no_dream) == 2
    assert all(c.source_type != "dream" for c in no_dream)

    # Exclude multiple
    only_ingest = await store.get_all_chunks(db, exclude_source_types={"dream", "remember:ai"})
    assert len(only_ingest) == 1
    assert only_ingest[0].source_type == "ingest"


@pytest.mark.asyncio
async def test_get_file_hash(db) -> None:
    emb = _make_embedding()
    await store.upsert_chunks(
        db,
        chunks=[{"content": "C", "section_path": "S", "chunk_hash": "h1", "embedding": emb}],
        source_file="test.md",
        file_hash="abc123",
    )

    assert await store.get_file_hash(db, "test.md") == "abc123"
    assert await store.get_file_hash(db, "nonexistent.md") is None


@pytest.mark.asyncio
async def test_delete_by_id(db) -> None:
    emb = _make_embedding()
    row_id = await store.insert_chunk(db, content="To delete this memory", chunk_hash="del1", embedding=emb)

    assert await store.delete_by_id(db, row_id) is True
    assert await store.delete_by_id(db, row_id) is False
    assert await store.get_all_chunks(db) == []


@pytest.mark.asyncio
async def test_delete_by_source(db) -> None:
    emb = _make_embedding()
    await store.upsert_chunks(
        db,
        chunks=[{"content": "C", "section_path": "S", "chunk_hash": "h1", "embedding": emb}],
        source_file="to_delete.md",
        file_hash="fh",
    )

    count = await store.delete_by_source(db, "to_delete.md")
    assert count == 1
    assert await store.get_all_chunks(db) == []
    assert await store.get_file_hash(db, "to_delete.md") is None


@pytest.mark.asyncio
async def test_bm25_search(db) -> None:
    emb = _make_embedding()
    await store.upsert_chunks(
        db,
        chunks=[
            {"content": "Alice loves engineering principles", "section_path": "S", "chunk_hash": "h1", "embedding": emb},
            {"content": "Alice enjoys painting landscapes", "section_path": "S", "chunk_hash": "h2", "embedding": emb},
            {"content": "Engineering is a broad discipline", "section_path": "S", "chunk_hash": "h3", "embedding": emb},
        ],
        source_file="test.md",
        file_hash="fh",
    )

    results = await store.bm25_search(db, "engineering principles")
    assert len(results) >= 2
    # Both chunks containing "engineering" should appear
    all_chunks = await store.get_all_chunks(db)
    engineering_ids = {c.id for c in all_chunks if "engineering" in c.content.lower()}
    assert engineering_ids.issubset(set(results))


@pytest.mark.asyncio
async def test_bm25_fts_kept_in_sync_on_delete(db) -> None:
    emb = _make_embedding()
    row_id = await store.insert_chunk(
        db, content="Temporary engineering note to delete", chunk_hash="t1", embedding=emb,
    )

    # Should find it
    results = await store.bm25_search(db, "engineering")
    assert row_id in results

    # Delete and verify FTS is cleaned up
    await store.delete_by_id(db, row_id)
    results = await store.bm25_search(db, "engineering")
    assert row_id not in results


@pytest.mark.asyncio
async def test_get_stats(db) -> None:
    emb = _make_embedding()
    await store.upsert_chunks(
        db,
        chunks=[{"content": "C", "section_path": "S", "chunk_hash": "h1", "embedding": emb}],
        source_file="test.md",
        file_hash="fh",
    )

    stats = await store.get_stats(db)
    assert stats["total_chunks"] == 1
    assert len(stats["sources"]) == 1
    assert stats["sources"][0]["file"] == "test.md"
