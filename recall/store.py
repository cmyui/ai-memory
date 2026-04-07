import re
import time
from dataclasses import dataclass
from typing import Any

import databases
import numpy as np

from recall import settings as settings_module

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    source_file TEXT,
    section_path TEXT,
    chunk_hash TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'ingest',
    fact_date TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);

CREATE TABLE IF NOT EXISTS ingest_log (
    source_file TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    ingested_at REAL NOT NULL,
    chunk_count INTEGER NOT NULL
);
"""

FTS_SCHEMA_SQL = "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content)"

_FTS_SPECIAL = re.compile(r"[^\w\s]", re.UNICODE)


async def get_database() -> databases.Database:
    db_path = settings_module.settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = databases.Database(f"sqlite:///{db_path}")
    await db.connect()

    # WAL mode: allows concurrent readers + better write concurrency across processes
    await db.execute(query="PRAGMA journal_mode=WAL")

    for statement in SCHEMA_SQL.strip().split(";"):
        statement = statement.strip()
        if statement:
            await db.execute(query=statement)

    await db.execute(query=FTS_SCHEMA_SQL)

    # Auto-populate FTS if chunks exist but FTS is empty (migration)
    chunks_count = await db.fetch_one(query="SELECT COUNT(*) as c FROM chunks")
    fts_count = await db.fetch_one(query="SELECT COUNT(*) as c FROM chunks_fts")
    assert chunks_count is not None and fts_count is not None
    if chunks_count._mapping["c"] > 0 and fts_count._mapping["c"] == 0:
        await db.execute(
            query="INSERT INTO chunks_fts(rowid, content) SELECT id, content FROM chunks",
        )

    # Migration: add source_type and fact_date columns if missing
    cols = await db.fetch_all(query="PRAGMA table_info(chunks)")
    col_names = {row._mapping["name"] for row in cols}
    if "source_type" not in col_names:
        await db.execute(
            query="ALTER TABLE chunks ADD COLUMN source_type TEXT NOT NULL DEFAULT 'ingest'",
        )
        await db.execute(
            query="CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type)",
        )
    if "fact_date" not in col_names:
        await db.execute(query="ALTER TABLE chunks ADD COLUMN fact_date TEXT")

    return db


async def upsert_chunks(
    db: databases.Database,
    *,
    chunks: list[dict[str, Any]],
    source_file: str,
    file_hash: str,
    source_type: str = "ingest",
) -> int:
    """Delete old chunks for source_file, insert new ones, update ingest_log."""
    now = time.time()

    async with db.transaction():
        await db.execute(
            query="DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE source_file = :source_file)",
            values={"source_file": source_file},
        )
        await db.execute(
            query="DELETE FROM chunks WHERE source_file = :source_file",
            values={"source_file": source_file},
        )

        for chunk in chunks:
            row_id: int = await db.execute(
                query="""\
                    INSERT OR REPLACE INTO chunks
                    (content, source_file, section_path, chunk_hash, embedding, created_at, source_type, fact_date)
                    VALUES (:content, :source_file, :section_path, :chunk_hash, :embedding, :created_at, :source_type, :fact_date)
                """,
                values={
                    "content": chunk["content"],
                    "source_file": source_file,
                    "section_path": chunk["section_path"],
                    "chunk_hash": chunk["chunk_hash"],
                    "embedding": chunk["embedding"].tobytes(),
                    "created_at": now,
                    "source_type": source_type,
                    "fact_date": chunk.get("fact_date"),
                },
            )
            await db.execute(
                query="INSERT INTO chunks_fts(rowid, content) VALUES (:rowid, :content)",
                values={"rowid": row_id, "content": chunk["content"]},
            )

        await db.execute(
            query="""\
                INSERT OR REPLACE INTO ingest_log
                (source_file, file_hash, ingested_at, chunk_count)
                VALUES (:source_file, :file_hash, :ingested_at, :chunk_count)
            """,
            values={
                "source_file": source_file,
                "file_hash": file_hash,
                "ingested_at": now,
                "chunk_count": len(chunks),
            },
        )

    return len(chunks)


async def insert_chunk(
    db: databases.Database,
    *,
    content: str,
    chunk_hash: str,
    embedding: np.ndarray,
    source_file: str | None = None,
    source_type: str = "ingest",
    fact_date: str | None = None,
) -> int:
    """Insert a single chunk (e.g. from `recall remember`). Returns the row id."""
    now = time.time()
    row_id: int = await db.execute(
        query="""\
            INSERT OR REPLACE INTO chunks
            (content, source_file, section_path, chunk_hash, embedding, created_at, source_type, fact_date)
            VALUES (:content, :source_file, NULL, :chunk_hash, :embedding, :created_at, :source_type, :fact_date)
        """,
        values={
            "content": content,
            "source_file": source_file,
            "chunk_hash": chunk_hash,
            "embedding": embedding.tobytes(),
            "created_at": now,
            "source_type": source_type,
            "fact_date": fact_date,
        },
    )
    await db.execute(
        query="INSERT INTO chunks_fts(rowid, content) VALUES (:rowid, :content)",
        values={"rowid": row_id, "content": content},
    )
    return row_id


@dataclass
class ChunkRow:
    id: int
    content: str
    source_file: str | None
    section_path: str | None
    embedding: np.ndarray
    created_at: float
    source_type: str
    fact_date: str | None


async def get_all_chunks(
    db: databases.Database,
    *,
    exclude_source_types: set[str] | None = None,
) -> list[ChunkRow]:
    """Load all chunks (with embeddings) for similarity search."""
    dims = settings_module.settings.embedding_dimensions

    if exclude_source_types:
        placeholders = ", ".join(f":ex{i}" for i in range(len(exclude_source_types)))
        query_str = f"SELECT id, content, source_file, section_path, embedding, created_at, source_type, fact_date FROM chunks WHERE source_type NOT IN ({placeholders})"
        values = {f"ex{i}": st for i, st in enumerate(exclude_source_types)}
        rows = await db.fetch_all(query=query_str, values=values)
    else:
        rows = await db.fetch_all(
            query="SELECT id, content, source_file, section_path, embedding, created_at, source_type, fact_date FROM chunks",
        )

    return [
        ChunkRow(
            id=row._mapping["id"],
            content=row._mapping["content"],
            source_file=row._mapping["source_file"],
            section_path=row._mapping["section_path"],
            embedding=np.frombuffer(row._mapping["embedding"], dtype=np.float32).reshape(dims),
            created_at=row._mapping["created_at"],
            source_type=row._mapping["source_type"],
            fact_date=row._mapping["fact_date"],
        )
        for row in rows
    ]


def _sanitize_fts_query(query: str) -> str:
    """Convert natural language query into FTS5 OR query."""
    cleaned = _FTS_SPECIAL.sub(" ", query)
    words = [w for w in cleaned.split() if len(w) >= 2]
    if not words:
        return ""
    return " OR ".join(f'"{w}"' for w in words)


async def bm25_search(
    db: databases.Database,
    query: str,
    *,
    top_k: int = 20,
) -> list[int]:
    """Return chunk IDs ordered by BM25 relevance."""
    fts_query = _sanitize_fts_query(query)
    if not fts_query:
        return []

    try:
        rows = await db.fetch_all(
            query="""\
                SELECT rowid as id FROM chunks_fts
                WHERE chunks_fts MATCH :query
                ORDER BY rank
                LIMIT :top_k
            """,
            values={"query": fts_query, "top_k": top_k},
        )
        return [row._mapping["id"] for row in rows]
    except Exception:
        return []


async def bm25_search_scored(
    db: databases.Database,
    query: str,
    *,
    top_k: int = 20,
) -> dict[int, float]:
    """Return chunk IDs with BM25 scores (higher = more relevant)."""
    fts_query = _sanitize_fts_query(query)
    if not fts_query:
        return {}

    try:
        rows = await db.fetch_all(
            query="""\
                SELECT rowid as id, rank as bm25_rank FROM chunks_fts
                WHERE chunks_fts MATCH :query
                ORDER BY rank
                LIMIT :top_k
            """,
            values={"query": fts_query, "top_k": top_k},
        )
        # FTS5 rank is negative (lower = better). Negate so higher = better.
        return {row._mapping["id"]: -row._mapping["bm25_rank"] for row in rows}
    except Exception:
        return {}


async def get_file_hash(db: databases.Database, source_file: str) -> str | None:
    row = await db.fetch_one(
        query="SELECT file_hash FROM ingest_log WHERE source_file = :source_file",
        values={"source_file": source_file},
    )
    return row._mapping["file_hash"] if row else None


async def delete_by_id(db: databases.Database, chunk_id: int) -> bool:
    row = await db.fetch_one(
        query="SELECT id FROM chunks WHERE id = :id",
        values={"id": chunk_id},
    )
    if row is None:
        return False
    await db.execute(
        query="DELETE FROM chunks_fts WHERE rowid = :id",
        values={"id": chunk_id},
    )
    await db.execute(
        query="DELETE FROM chunks WHERE id = :id",
        values={"id": chunk_id},
    )
    return True


async def delete_by_source(db: databases.Database, source_file: str) -> int:
    rows = await db.fetch_all(
        query="SELECT id FROM chunks WHERE source_file = :source_file",
        values={"source_file": source_file},
    )
    count = len(rows)
    if count > 0:
        await db.execute(
            query="DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE source_file = :source_file)",
            values={"source_file": source_file},
        )
        await db.execute(
            query="DELETE FROM chunks WHERE source_file = :source_file",
            values={"source_file": source_file},
        )
    await db.execute(
        query="DELETE FROM ingest_log WHERE source_file = :source_file",
        values={"source_file": source_file},
    )
    return count


async def get_stats(db: databases.Database) -> dict[str, Any]:
    total_row = await db.fetch_one(query="SELECT COUNT(*) as c FROM chunks")
    assert total_row is not None
    total: int = total_row._mapping["c"]

    sources = await db.fetch_all(
        query="SELECT source_file, chunk_count FROM ingest_log ORDER BY source_file",
    )

    db_path = settings_module.settings.db_path
    db_size = db_path.stat().st_size if db_path.exists() else 0

    return {
        "total_chunks": total,
        "sources": [{"file": row._mapping["source_file"], "chunks": row._mapping["chunk_count"]} for row in sources],
        "db_size_bytes": db_size,
    }
