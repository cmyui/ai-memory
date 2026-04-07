#!/usr/bin/env python3
"""
Recall MCP Server — semantic memory for Claude Code
=====================================================
Install: claude mcp add recall -- python -m recall.mcp_server

Tools:
  recall_search    — hybrid BM25 + dense search with cross-encoder reranking
  recall_remember  — store a new memory (auto-dates, defaults to AI-authored)
  recall_forget    — delete a memory by ID
  recall_stats     — memory count, DB size, source breakdown
"""

import asyncio
import datetime
import hashlib
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger("recall_mcp")

_db = None


async def _get_db():
    global _db
    if _db is None:
        from recall import store

        _db = await store.get_database()
    return _db


async def tool_search(query: str, top_k: int = 25, min_confidence: float = 0.60) -> str:
    """Semantic search over memory."""
    from recall import retrieval

    db = await _get_db()
    old_conf = retrieval.MIN_CONFIDENCE
    retrieval.MIN_CONFIDENCE = min_confidence

    results, elapsed_ms = await retrieval.query(db, query, top_k=top_k)

    retrieval.MIN_CONFIDENCE = old_conf
    return retrieval.format_results(results, elapsed_ms)


async def tool_remember(text: str, human: bool = False) -> str:
    """Store a new memory. Auto-prepends today's date if missing."""
    from recall import chunker
    from recall import embeddings
    from recall import store

    if not chunker.parse_fact_date(text):
        text = f"[{datetime.date.today().isoformat()}] {text}"

    fact_date = chunker.parse_fact_date(text)
    source_type = "remember:human" if human else "remember:ai"
    chunk_hash = hashlib.sha256(text.encode()).hexdigest()
    embedding = embeddings.embed_query(text)

    db = await _get_db()
    row_id = await store.insert_chunk(
        db,
        content=text,
        chunk_hash=chunk_hash,
        embedding=embedding,
        source_type=source_type,
        fact_date=fact_date,
    )
    return f"Saved memory (id={row_id}, type={source_type})"


async def tool_forget(memory_id: int) -> str:
    """Delete a memory by ID."""
    from recall import store

    db = await _get_db()
    deleted = await store.delete_by_id(db, memory_id)
    if deleted:
        return f"Deleted memory id={memory_id}"
    return f"No memory found with id={memory_id}"


async def tool_stats() -> str:
    """Show memory count, DB size, and source breakdown."""
    from recall import store
    from recall import settings as settings_module

    db = await _get_db()
    s = await store.get_stats(db)

    db_bytes = s["db_size_bytes"]
    if db_bytes >= 1 << 30:
        db_size_str = f"{db_bytes / (1 << 30):.1f} GB"
    elif db_bytes >= 1 << 20:
        db_size_str = f"{db_bytes / (1 << 20):.1f} MB"
    else:
        db_size_str = f"{db_bytes / (1 << 10):.1f} KB"

    lines = [
        f"Total memories: {s['total_chunks']}",
        f"DB size: {db_size_str}",
    ]
    if s["sources"]:
        sorted_sources = sorted(s["sources"], key=lambda x: x["chunks"], reverse=True)
        lines.append(f"\nTop sources ({len(sorted_sources)} total):")
        for src in sorted_sources[:20]:
            lines.append(f"  {src['file']} ({src['chunks']} memories)")
        if len(sorted_sources) > 20:
            lines.append(f"  ... +{len(sorted_sources) - 20} more")

    return "\n".join(lines)


TOOLS = {
    "recall_search": {
        "description": "Search semantic memory. Returns facts ranked by relevance. Use this to look up anything about the user: biographical facts, decisions, plans, opinions, relationships, projects, finances, life events, preferences, habits, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 25)",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum cosine similarity threshold (default: 0.60)",
                },
            },
            "required": ["query"],
        },
        "handler": tool_search,
    },
    "recall_remember": {
        "description": "Store a new fact in memory. Auto-prepends today's date if no [YYYY-MM-DD] prefix. Use this when the user reveals new information worth remembering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The fact to remember. Should be self-contained and name the subject explicitly.",
                },
                "human": {
                    "type": "boolean",
                    "description": "Set true if the user is explicitly asking to remember something (default: false, AI-authored)",
                },
            },
            "required": ["text"],
        },
        "handler": tool_remember,
    },
    "recall_forget": {
        "description": "Delete a memory by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "The memory ID to delete",
                },
            },
            "required": ["memory_id"],
        },
        "handler": tool_forget,
    },
    "recall_stats": {
        "description": "Show memory database statistics: total count, DB size, top sources.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
        "handler": tool_stats,
    },
}


def handle_request(request: dict) -> dict | None:
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "recall", "version": "1.0.0"},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": name,
                        "description": t["description"],
                        "inputSchema": t["inputSchema"],
                    }
                    for name, t in TOOLS.items()
                ]
            },
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        try:
            result = asyncio.run(TOOLS[tool_name]["handler"](**tool_args))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": result}]},
            }
        except Exception as e:
            logger.error(f"Tool error in {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main() -> None:
    logger.info("Recall MCP Server starting...")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
