import argparse
import asyncio
import datetime
import hashlib
import sys
from pathlib import Path

import databases

from recall import chunker
from recall import dreamer
from recall import embeddings
from recall import extractor
from recall import llm
from recall import retrieval
from recall import settings as settings_module
from recall import store


async def _embed_and_store(
    db: databases.Database,
    chunks: list[chunker.Chunk],
    source_file: str,
    file_hash: str,
) -> int:
    """Embed chunks in batches and store them."""
    embed_batch_size = 1000
    all_chunk_dicts: list[dict] = []
    for batch_start in range(0, len(chunks), embed_batch_size):
        batch = chunks[batch_start : batch_start + embed_batch_size]
        batch_embeddings = embeddings.embed_texts([c.content for c in batch])
        all_chunk_dicts.extend(
            {
                "content": c.content,
                "section_path": c.section_path,
                "chunk_hash": c.chunk_hash,
                "embedding": emb,
                "fact_date": c.fact_date,
            }
            for c, emb in zip(batch, batch_embeddings)
        )

    return await store.upsert_chunks(
        db,
        chunks=all_chunk_dicts,
        source_file=source_file,
        file_hash=file_hash,
    )


async def _ingest_one_file(
    file_path: Path,
    directory: Path,
    db: databases.Database,
    llm_sem: asyncio.Semaphore,
    db_lock: asyncio.Lock,
) -> tuple[str, int]:
    """Ingest a single file. Returns (source_file, chunk_count) or ("", 0) if skipped."""

    source_file = str(file_path.relative_to(directory))
    file_content = file_path.read_bytes()
    file_hash = hashlib.sha256(file_content).hexdigest()

    async with db_lock:
        existing_hash = await store.get_file_hash(db, source_file)
    if existing_hash == file_hash:
        return "", 0

    raw_text = chunker.read_file_text(file_path)
    if not raw_text.strip():
        return "", 0

    # Split into pages at line boundaries
    file_context = f"[Source: {source_file}]\n\n"
    max_page = extractor.get_max_batch_chars()
    if len(raw_text) <= max_page:
        pages = [file_context + raw_text]
    else:
        pages = []
        lines = raw_text.split("\n")
        current_page: list[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > max_page and current_page:
                pages.append(file_context + "\n".join(current_page))
                current_page = [line]
                current_len = line_len
            else:
                current_page.append(line)
                current_len += line_len
        if current_page:
            pages.append(file_context + "\n".join(current_page))

    print(f"  [start] {source_file} ({len(pages)} pages)...", flush=True)

    # LLM extraction — parallel batches within this file, gated by semaphore
    batches = _batch_by_chars(pages, max_page)

    async def _extract_one(batch_texts: list[str]) -> list[str]:
        async with llm_sem:
            return await extractor.extract_facts(batch_texts)

    if len(batches) == 1:
        all_facts = await _extract_one(batches[0])
    else:
        batch_results = await asyncio.gather(*[_extract_one(b) for b in batches])
        all_facts = [fact for result in batch_results for fact in result]

    # Embed and store — serialize DB writes to avoid SQLite locking
    all_chunks = [
        c for fact in all_facts
        if (c := chunker.build_chunk(fact, source_file, source_file)) is not None
    ]
    if not all_chunks:
        # Record in ingest_log so we skip this file next run
        async with db_lock:
            await store.upsert_chunks(db, chunks=[], source_file=source_file, file_hash=file_hash)
        print(f"  [skip]  {source_file} → 0 facts", flush=True)
        return "", 0

    async with db_lock:
        await _embed_and_store(db, all_chunks, source_file, file_hash)
    print(f"  [done]  {source_file} → {len(all_chunks)} facts", flush=True)
    return source_file, len(all_chunks)


def _batch_by_chars(texts: list[str], max_chars: int) -> list[list[str]]:
    """Split texts into batches that fit within max_chars."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for text in texts:
        if current_len + len(text) > max_chars and current:
            batches.append(current)
            current = [text]
            current_len = len(text)
        else:
            current.append(text)
            current_len += len(text)
    if current:
        batches.append(current)
    return batches


async def cmd_ingest(args: argparse.Namespace) -> None:
    for hint in getattr(args, "hint", []):
        extractor.add_hint(hint)
    if getattr(args, "model", None):
        llm.set_model(args.model)
    if getattr(args, "max_emails", None):
        settings_module.settings.max_emails = args.max_emails

    directory = Path(args.path).expanduser().resolve()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    db = await store.get_database()
    try:
        all_files = chunker.list_supported_files(directory)

        # Sort by file size
        smallest_first = getattr(args, "smallest_first", False)
        all_files.sort(key=lambda f: f.stat().st_size, reverse=not smallest_first)
        top_n = getattr(args, "top_n", None)
        if top_n is not None:
            all_files = all_files[:top_n]

        # Filter out excluded patterns
        exclude_patterns: list[str] = getattr(args, "exclude", [])
        if exclude_patterns:
            def _excluded(fp: Path) -> bool:
                rel = str(fp.relative_to(directory))
                return any(pat in rel for pat in exclude_patterns)
            before = len(all_files)
            all_files = [f for f in all_files if not _excluded(f)]
            excluded_count = before - len(all_files)
            if excluded_count:
                print(f"Excluded {excluded_count} files matching: {', '.join(exclude_patterns)}", flush=True)

        concurrency = getattr(args, "concurrency", 1)
        num_files = len(all_files)
        print(f"Processing {num_files} files (concurrency={concurrency})...", flush=True)

        llm_sem = asyncio.Semaphore(concurrency)
        db_lock = asyncio.Lock()

        # Always process files sequentially; -j controls batch parallelism within each file
        total_chunks = 0
        skipped = 0
        ingested = 0

        for i, file_path in enumerate(all_files, 1):
            source_file, count = await _ingest_one_file(file_path, directory, db, llm_sem, db_lock)
            if count == 0:
                skipped += 1
            else:
                total_chunks += count
                ingested += 1

        print(f"Ingested {total_chunks} chunks from {ingested} files ({skipped} unchanged, skipped)")
    finally:
        await db.disconnect()


async def cmd_remember(args: argparse.Namespace) -> None:
    text = args.text

    # Auto-prepend today's date if no date bracket present
    if not chunker.parse_fact_date(text):
        text = f"[{datetime.date.today().isoformat()}] {text}"

    fact_date = chunker.parse_fact_date(text)
    source_type = "remember:human" if args.human else "remember:ai"

    chunk_hash = hashlib.sha256(text.encode()).hexdigest()
    embedding = embeddings.embed_query(text)

    db = await store.get_database()
    try:
        row_id = await store.insert_chunk(
            db,
            content=text,
            chunk_hash=chunk_hash,
            embedding=embedding,
            source_type=source_type,
            fact_date=fact_date,
        )
        print(f"Saved memory (id={row_id}, type={source_type})")
    finally:
        await db.disconnect()


async def cmd_query(args: argparse.Namespace) -> None:
    db = await store.get_database()
    try:
        min_conf = getattr(args, "min_confidence", None)
        if min_conf is not None:
            retrieval.MIN_CONFIDENCE = min_conf

        results, elapsed_ms = await retrieval.query(
            db,
            args.query,
            top_k=args.top_k,
        )
        print(retrieval.format_results(results, elapsed_ms))
    finally:
        await db.disconnect()


async def cmd_forget(args: argparse.Namespace) -> None:
    db = await store.get_database()
    try:
        if args.id is not None:
            deleted = await store.delete_by_id(db, args.id)
            if deleted:
                print(f"Deleted memory id={args.id}")
            else:
                print(f"No memory found with id={args.id}", file=sys.stderr)
                sys.exit(1)
        elif args.source is not None:
            count = await store.delete_by_source(db, args.source)
            print(f"Deleted {count} memories from {args.source}")
        else:
            print("Specify --id or --source", file=sys.stderr)
            sys.exit(1)
    finally:
        await db.disconnect()


async def cmd_dream(args: argparse.Namespace) -> None:
    db = await store.get_database()
    try:
        await dreamer.run_dream(db, dry_run=getattr(args, "dry_run", False))
    finally:
        await db.disconnect()


async def cmd_stats(_args: argparse.Namespace) -> None:
    db = await store.get_database()
    try:
        s = await store.get_stats(db)
        db_bytes = s["db_size_bytes"]
        if db_bytes >= 1 << 30:
            db_size_str = f"{db_bytes / (1 << 30):.1f} GB"
        elif db_bytes >= 1 << 20:
            db_size_str = f"{db_bytes / (1 << 20):.1f} MB"
        else:
            db_size_str = f"{db_bytes / (1 << 10):.1f} KB"

        print(f"Total memories: {s['total_chunks']}")
        print(f"DB size:        {db_size_str}")
        if s["sources"]:
            print("\nIngested files:")
            sorted_sources = sorted(s["sources"], key=lambda x: x["chunks"], reverse=True)
            for src in sorted_sources:
                print(f"  {src['file']} ({src['chunks']} memories)")
    finally:
        await db.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recall", description="Semantic memory for Claude Code")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--use-claude-cli", action="store_true", help="Use Claude CLI (Opus) instead of OpenAI for LLM calls")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest files from a directory")
    p_ingest.add_argument("path", help="Directory to ingest")
    p_ingest.add_argument("--top-n", type=int, help="Only ingest the top N files by size")
    p_ingest.add_argument("--hint", action="append", default=[], help="Context hints for the LLM extractor (can be repeated)")
    p_ingest.add_argument("--model", default=None, help="Override extraction model (e.g. gpt-4.1-mini)")
    p_ingest.add_argument("--max-emails", type=int, default=None, help="Max emails to keep from mbox files (sorted by length)")
    p_ingest.add_argument("--exclude", action="append", default=[], help="Skip files whose path contains this substring (repeatable)")
    p_ingest.add_argument("--concurrency", "-j", type=int, default=1, help="Number of parallel LLM extractions (default: 1)")
    p_ingest.add_argument("--smallest-first", action="store_true", help="Process smallest files first (race to checkpoints)")

    # remember
    p_remember = subparsers.add_parser("remember", help="Save a memory")
    p_remember.add_argument("text", help="The fact or observation to remember")
    p_remember.add_argument("--human", action="store_true", help="Mark as human-authored (default is AI-authored)")

    # query
    p_query = subparsers.add_parser("query", help="Semantic search over memory")
    p_query.add_argument("query", help="Search query")
    p_query.add_argument("--top-k", type=int, default=50, help="Number of results")
    p_query.add_argument("--min-confidence", type=float, default=None, help="Override minimum cosine similarity threshold")

    # forget
    p_forget = subparsers.add_parser("forget", help="Delete memories")
    p_forget.add_argument("--id", type=int, help="Delete specific chunk by ID")
    p_forget.add_argument("--source", help="Delete all chunks from a source file")

    # dream
    p_dream = subparsers.add_parser("dream", help="Run dream cycle (consolidate, deduplicate, build peer cards)")
    p_dream.add_argument("--dry-run", action="store_true", help="Show what would happen without writing to DB")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start web UI")
    p_serve.add_argument("--port", type=int, default=8765, help="Port to serve on")

    # stats
    subparsers.add_parser("stats", help="Show memory statistics")

    return parser


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn  # optional dependency, only needed for serve command

    from recall.web.app import app  # optional dependency (fastapi)

    print(f"Starting recall web UI on http://localhost:{args.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


COMMAND_HANDLERS = {
    "ingest": cmd_ingest,
    "remember": cmd_remember,
    "query": cmd_query,
    "forget": cmd_forget,
    "dream": cmd_dream,
    "stats": cmd_stats,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.debug:
        settings_module.settings.debug = True
    if args.use_claude_cli:
        llm.set_use_claude_cli(True)
    if args.command == "serve":
        cmd_serve(args)
    else:
        handler = COMMAND_HANDLERS[args.command]
        asyncio.run(handler(args))
