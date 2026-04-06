import time
from dataclasses import dataclass

import databases
import numpy as np

from recall import embeddings
from recall import store

# Reciprocal Rank Fusion constant (standard value from Cormack et al. 2009)
RRF_K = 60

# Minimum cosine similarity for the top result to be considered relevant
MIN_CONFIDENCE = 0.60


@dataclass
class Result:
    id: int
    content: str
    source_file: str | None
    section_path: str | None
    score: float
    cosine_sim: float
    created_at: float
    source_type: str
    fact_date: str | None


async def query(
    db: databases.Database,
    query_text: str,
    *,
    top_k: int = 5,
) -> tuple[list[Result], float]:
    """Hybrid retrieval: BM25 + dense embeddings combined via RRF."""
    t0 = time.time()

    query_embedding = embeddings.embed_query(query_text)
    all_chunks = await store.get_all_chunks(db)

    if not all_chunks:
        return [], (time.time() - t0) * 1000

    chunk_by_id = {c.id: c for c in all_chunks}

    # Dense ranking: cosine similarity
    candidate_count = top_k * 3
    chunk_embeddings = np.stack([c.embedding for c in all_chunks])
    similarities = chunk_embeddings @ query_embedding  # already normalized

    cosine_by_id = {c.id: float(sim) for c, sim in zip(all_chunks, similarities)}

    dense_order = np.argsort(similarities)[::-1][:candidate_count]
    dense_ranking = {
        all_chunks[idx].id: rank + 1 for rank, idx in enumerate(dense_order)
    }

    # BM25 ranking
    bm25_ids = await store.bm25_search(db, query_text, top_k=candidate_count)
    bm25_ranking = {chunk_id: rank + 1 for rank, chunk_id in enumerate(bm25_ids)}

    # Reciprocal Rank Fusion
    all_candidate_ids = set(dense_ranking) | set(bm25_ranking)
    rrf_scores: dict[int, float] = {}
    for chunk_id in all_candidate_ids:
        score = 0.0
        if chunk_id in dense_ranking:
            score += 1.0 / (RRF_K + dense_ranking[chunk_id])
        if chunk_id in bm25_ranking:
            score += 1.0 / (RRF_K + bm25_ranking[chunk_id])
        rrf_scores[chunk_id] = score

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

    results = [
        Result(
            id=cid,
            content=chunk_by_id[cid].content,
            source_file=chunk_by_id[cid].source_file,
            section_path=chunk_by_id[cid].section_path,
            score=rrf_scores[cid],
            cosine_sim=cosine_by_id.get(cid, 0.0),
            created_at=chunk_by_id[cid].created_at,
            source_type=chunk_by_id[cid].source_type,
            fact_date=chunk_by_id[cid].fact_date,
        )
        for cid in sorted_ids
        if cid in chunk_by_id
    ]

    elapsed_ms = (time.time() - t0) * 1000
    return results, elapsed_ms


def format_results(results: list[Result], elapsed_ms: float) -> str:
    """Format results for CLI output."""
    if not results:
        return "No results found."

    # Filter out results below confidence threshold
    results = [r for r in results if r.cosine_sim >= MIN_CONFIDENCE]
    if not results:
        return "No relevant memories found for this query. Do not guess or infer — the information is not in the memory system."

    lines: list[str] = []
    sources: set[str] = set()

    for r in results:
        source = r.source_file or "remembered"
        sources.add(source)
        # Strip the source_file prefix from content if duplicated
        content = r.content
        prefix = f"{source}\n\n"
        if content.startswith(prefix):
            content = content[len(prefix) :]
        lines.append(f"[{source}] (score: {r.cosine_sim:.3f})")
        lines.append(content)
        lines.append("")

    lines.append("---")
    lines.append(
        f"{len(results)} results from {len(sources)} sources ({elapsed_ms:.0f}ms)"
    )

    # Machine-readable footer for consuming agents
    result_ids = ",".join(str(r.id) for r in results)
    lines.append(f"[result-id: {result_ids}]")

    return "\n".join(lines)
