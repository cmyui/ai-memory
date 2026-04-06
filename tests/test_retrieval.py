import time

import numpy as np
import pytest
import pytest_asyncio

from recall import retrieval
from recall import settings as settings_module
from recall import store


@pytest_asyncio.fixture
async def db(tmp_path):
    settings_module.settings.db_path = tmp_path / "test.db"
    database = await store.get_database()
    yield database
    await database.disconnect()


def _make_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(settings_module.settings.embedding_dimensions, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.mark.asyncio
async def test_query_returns_results(db, monkeypatch) -> None:
    emb = _make_embedding(seed=1)
    await store.upsert_chunks(
        db,
        chunks=[{"content": "I like pizza", "section_path": "Food", "chunk_hash": "h1", "embedding": emb}],
        source_file="test.md",
        file_hash="fh",
    )

    from recall import embeddings
    monkeypatch.setattr(embeddings, "embed_query", lambda _text: emb)

    results, elapsed_ms = await retrieval.query(db, "pizza")
    assert len(results) == 1
    assert results[0].content == "I like pizza"
    assert results[0].source_type == "ingest"
    assert results[0].fact_date is None
    assert elapsed_ms >= 0


@pytest.mark.asyncio
async def test_query_includes_source_type_and_fact_date(db, monkeypatch) -> None:
    emb = _make_embedding(seed=1)
    await store.insert_chunk(
        db,
        content="[2026-04-05] Alice likes sushi a lot",
        chunk_hash="h1",
        embedding=emb,
        source_type="remember:ai",
        fact_date="2026-04-05",
    )

    from recall import embeddings
    monkeypatch.setattr(embeddings, "embed_query", lambda _text: emb)

    results, _ = await retrieval.query(db, "sushi")
    assert len(results) == 1
    assert results[0].source_type == "remember:ai"
    assert results[0].fact_date == "2026-04-05"


@pytest.mark.asyncio
async def test_query_respects_top_k(db, monkeypatch) -> None:
    emb = _make_embedding(seed=1)

    for i in range(10):
        await store.upsert_chunks(
            db,
            chunks=[{"content": f"Chunk {i}", "section_path": "S", "chunk_hash": f"h{i}", "embedding": emb}],
            source_file=f"file{i}.md",
            file_hash=f"fh{i}",
        )

    from recall import embeddings
    monkeypatch.setattr(embeddings, "embed_query", lambda _text: emb)

    results, _ = await retrieval.query(db, "test", top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_query_empty_db(db, monkeypatch) -> None:
    from recall import embeddings
    monkeypatch.setattr(embeddings, "embed_query", lambda _text: _make_embedding())

    results, _ = await retrieval.query(db, "anything")
    assert results == []


@pytest.mark.asyncio
async def test_hybrid_retrieval_combines_bm25_and_dense(db, monkeypatch) -> None:
    """BM25-only match and dense-only match should both appear in results."""
    emb_a = _make_embedding(seed=1)
    emb_b = _make_embedding(seed=2)

    # Chunk A: matches "engineering" via BM25 keywords
    await store.upsert_chunks(
        db,
        chunks=[{"content": "Engineering principles for software design", "section_path": "S", "chunk_hash": "h1", "embedding": emb_a}],
        source_file="eng.md",
        file_hash="fh1",
    )
    # Chunk B: will match via dense similarity (same embedding as query)
    await store.upsert_chunks(
        db,
        chunks=[{"content": "Totally unrelated topic about cooking", "section_path": "S", "chunk_hash": "h2", "embedding": emb_b}],
        source_file="cook.md",
        file_hash="fh2",
    )

    from recall import embeddings
    # Query embedding matches chunk B perfectly (dense match), but query text matches chunk A (BM25)
    monkeypatch.setattr(embeddings, "embed_query", lambda _text: emb_b)

    results, _ = await retrieval.query(db, "engineering principles", top_k=2)

    contents = [r.content for r in results]
    # Both should appear: one via BM25, one via dense
    assert any("Engineering" in c for c in contents)
    assert any("cooking" in c for c in contents)


def test_format_results_empty() -> None:
    assert retrieval.format_results([], 10.0) == "No results found."


def test_format_results_output() -> None:
    results = [
        retrieval.Result(
            id=1,
            content="A fact about something interesting",
            source_file="test.md",
            section_path="Section A",
            score=0.95,
            cosine_sim=0.85,
            created_at=time.time(),
            source_type="ingest",
            fact_date="2025-01-15",
        ),
        retrieval.Result(
            id=2,
            content="Another remembered thing worth noting",
            source_file=None,
            section_path=None,
            score=0.75,
            cosine_sim=0.72,
            created_at=time.time(),
            source_type="remember:ai",
            fact_date=None,
        ),
    ]

    output = retrieval.format_results(results, 42.5)
    assert "[test.md]" in output
    assert "[remembered]" in output
    assert "2 results from 2 sources (42ms)" in output
