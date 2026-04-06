"""Web UI for managing recall memories."""

import hashlib
from pathlib import Path

import databases
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from recall import embeddings
from recall import retrieval
from recall import store

app = FastAPI(title="recall", debug=True)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_db: databases.Database | None = None


async def get_db() -> databases.Database:
    global _db
    if _db is None:
        _db = await store.get_database()
    return _db


@app.on_event("shutdown")
async def shutdown() -> None:
    if _db is not None:
        await _db.disconnect()


def _render(request: Request, template: str, context: dict) -> HTMLResponse:
    context["request"] = request
    return templates.TemplateResponse(request, template, context)


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    q: str = "",
    page: int = 1,
    per_page: int = 50,
) -> HTMLResponse:
    db = await get_db()

    if q:
        results, elapsed_ms = await retrieval.query(db, q, top_k=per_page * 2)
        memories = [
            {
                "id": r.id,
                "content": r.content,
                "source_file": r.source_file or "remembered",
                "score": f"{r.cosine_sim:.3f}",
            }
            for r in results
        ]
        total = len(memories)
        start = (page - 1) * per_page
        memories = memories[start : start + per_page]
    else:
        total_row = await db.fetch_one(query="SELECT COUNT(*) as c FROM chunks")
        assert total_row is not None
        total = total_row._mapping["c"]

        rows = await db.fetch_all(
            query="SELECT id, content, source_file, created_at FROM chunks ORDER BY created_at DESC LIMIT :limit OFFSET :offset",
            values={"limit": per_page, "offset": (page - 1) * per_page},
        )
        memories = [
            {
                "id": row._mapping["id"],
                "content": row._mapping["content"],
                "source_file": row._mapping["source_file"] or "remembered",
                "score": None,
            }
            for row in rows
        ]

    total_pages = max(1, (total + per_page - 1) // per_page)

    return _render(request, "index.html", {
        "memories": memories,
        "q": q,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    })


@app.get("/memory/{memory_id}", response_class=HTMLResponse)
async def view_memory(request: Request, memory_id: int) -> HTMLResponse:
    db = await get_db()
    row = await db.fetch_one(
        query="SELECT id, content, source_file, section_path, created_at FROM chunks WHERE id = :id",
        values={"id": memory_id},
    )
    if row is None:
        return HTMLResponse("Memory not found", status_code=404)

    memory = {
        "id": row._mapping["id"],
        "content": row._mapping["content"],
        "source_file": row._mapping["source_file"],
        "section_path": row._mapping["section_path"],
        "created_at": row._mapping["created_at"],
    }
    return _render(request, "edit.html", {"memory": memory})


@app.post("/memory/{memory_id}/update")
async def update_memory(request: Request, memory_id: int) -> RedirectResponse:
    form = await request.form()
    new_content = str(form.get("content", ""))

    if not new_content.strip():
        return RedirectResponse(url="/", status_code=303)

    db = await get_db()
    new_embedding = embeddings.embed_query(new_content)
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()

    await db.execute(
        query="UPDATE chunks SET content = :content, chunk_hash = :hash, embedding = :embedding WHERE id = :id",
        values={
            "content": new_content,
            "hash": new_hash,
            "embedding": new_embedding.tobytes(),
            "id": memory_id,
        },
    )
    await db.execute(query="DELETE FROM chunks_fts WHERE rowid = :id", values={"id": memory_id})
    await db.execute(
        query="INSERT INTO chunks_fts(rowid, content) VALUES (:id, :content)",
        values={"id": memory_id, "content": new_content},
    )

    return RedirectResponse(url=f"/memory/{memory_id}", status_code=303)


@app.post("/memory/{memory_id}/delete")
async def delete_memory(memory_id: int) -> RedirectResponse:
    db = await get_db()
    await store.delete_by_id(db, memory_id)
    return RedirectResponse(url="/", status_code=303)


@app.get("/create", response_class=HTMLResponse)
async def create_form(request: Request) -> HTMLResponse:
    return _render(request, "create.html", {})


@app.post("/create")
async def create_memory(request: Request) -> RedirectResponse:
    form = await request.form()
    content = str(form.get("content", ""))

    if not content.strip():
        return RedirectResponse(url="/create", status_code=303)

    db = await get_db()
    embedding = embeddings.embed_query(content)
    chunk_hash = hashlib.sha256(content.encode()).hexdigest()

    row_id = await store.insert_chunk(
        db, content=content, chunk_hash=chunk_hash, embedding=embedding
    )

    return RedirectResponse(url=f"/memory/{row_id}", status_code=303)
