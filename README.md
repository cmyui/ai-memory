# recall

Semantic memory system for Claude Code. Stores, retrieves, and consolidates personal knowledge using hybrid search (BM25 + dense embeddings) and LLM-powered fact extraction.

## How it works

```
Files → Chunker → LLM Extraction → Embeddings → SQLite
                                                    ↓
                              Query → Hybrid Search (BM25 + Dense → RRF)
                                                    ↓
                                              Ranked Results

Dream Cycle:
All Memories → Entity Extraction → Peer Card Synthesis → Store
```

1. **Ingest** — reads files (markdown, JSON, PDF, mbox, docx, csv), sends content to an LLM to extract self-contained facts, embeds them, and stores in SQLite.
2. **Query** — embeds the query, runs BM25 keyword search + dense cosine similarity, fuses rankings via Reciprocal Rank Fusion (RRF).
3. **Dream** — scans all memories to identify people, builds biographical peer cards per entity, stores as searchable memories. Excludes its own output to prevent feedback loops.

## Setup

```bash
pip install -e .
```

Requires either:
- An OpenAI API key (`OPENAI_API_KEY`) for GPT-based extraction (default)
- The [Claude CLI](https://claude.ai/code) installed and authenticated, with `--use-claude-cli` flag for Opus-based extraction

## Usage

### Ingest files

```bash
# Basic ingestion (uses GPT-4.1-nano by default)
recall ingest ~/documents/

# With Claude Opus (higher quality, uses your CLI subscription)
recall --use-claude-cli ingest ~/documents/

# Parallel batch extraction within each file (4 concurrent LLM calls)
recall --use-claude-cli ingest ~/documents/ -j 4

# Process smallest files first (checkpoint quickly)
recall --use-claude-cli ingest ~/documents/ -j 4 --smallest-first

# Skip certain directories
recall --use-claude-cli ingest ~/documents/ --exclude standups --exclude drafts

# With context hints for the LLM
recall ingest ~/discord-exports/ --hint "Discord DMs involving Alice Johnson (ally_j)"

# Override extraction model
recall ingest ~/notes/ --model gpt-4.1-mini
```

### Remember facts directly

```bash
# AI-authored (default, e.g. called by Claude Code)
recall remember "Alice prefers sushi over pizza"

# Human-authored
recall remember --human "My bank account number is 12345"
```

Auto-prepends today's date if no `[YYYY-MM-DD]` prefix is present.

### Search

```bash
recall query "who is Bob?"
recall query "production migration practices" --top-k 10
recall query "Alice's birthday" --min-confidence 0.5
```

### Forget

```bash
recall forget --id 42
recall forget --source "old-file.md"
```

### Dream cycle

```bash
recall dream              # Build/rebuild peer cards
recall dream --dry-run    # Preview without writing
```

### Other

```bash
recall stats              # Show memory count, DB size, sources
recall serve              # Start web UI on localhost:8765
```

## Architecture

| Module | Purpose |
|---|---|
| `cli.py` | CLI entry point, argument parsing, ingestion orchestration |
| `llm.py` | LLM backend abstraction (OpenAI API or Claude CLI subprocess) |
| `extractor.py` | Fact extraction prompts and batching |
| `embeddings.py` | Sentence-transformer embedding (BAAI/bge-small-en-v1.5) |
| `retrieval.py` | Hybrid BM25 + dense search with RRF fusion |
| `store.py` | SQLite storage, schema, migrations, CRUD |
| `chunker.py` | File discovery, format parsing, fact date extraction |
| `dreamer.py` | Dream cycle — entity extraction and peer card generation |
| `parsers.py` | Email body extraction helper |
| `web/app.py` | FastAPI web UI for browsing/editing memories |

## Schema

```sql
chunks (
    id INTEGER PRIMARY KEY,
    content TEXT,           -- the extracted fact
    source_file TEXT,       -- e.g. "identity.md" or "peer-card/Bob"
    section_path TEXT,      -- hierarchical context
    chunk_hash TEXT UNIQUE, -- SHA256(source_file + content)
    embedding BLOB,         -- float32 vector (384 dims)
    created_at REAL,        -- ingestion timestamp
    source_type TEXT,       -- 'ingest', 'remember:human', 'remember:ai', 'dream'
    fact_date TEXT          -- extracted from [YYYY-MM-DD] prefix, nullable
)
```

## Supported formats

- **Markdown** (`.md`) — raw text
- **JSON** (`.json`) — auto-detects Discord and Slack export formats
- **PDF** (`.pdf`) — text extraction via PyMuPDF
- **Email** (`.mbox`) — filters automated/calendar emails, sorts by substance
- **Word** (`.docx`) — XML paragraph extraction
- **CSV** (`.csv`) — raw text

## Configuration

Settings via `pydantic-settings` in `recall/settings.py`:

| Setting | Default | Description |
|---|---|---|
| `db_path` | `~/.local/share/recall/memory.db` | SQLite database location |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `embedding_dimensions` | `384` | Must match embedding model |
| `debug` | `True` | Verbose output |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
