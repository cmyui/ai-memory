#!/usr/bin/env python3
"""LongMemEval benchmark for recall.

Reproduces the same methodology as mempalace's benchmark:
- 500 human-written questions from the LongMemEval dataset
- Each question has ground-truth session IDs
- Measures R@k (Recall at k) — is the correct session in top-k results?

Uses a temporary DB per question — does NOT touch the main recall database.

Usage:
    python benchmarks/longmemeval_bench.py /path/to/longmemeval_s_cleaned.json
    python benchmarks/longmemeval_bench.py /path/to/data.json --limit 50  # quick test
"""

import argparse
import asyncio
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import databases
import numpy as np

from recall import embeddings
from recall import retrieval
from recall import settings as settings_module
from recall import store


def dcg(relevances: list[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg(rankings: list[int], correct_ids: set[str], corpus_ids: list[str], k: int) -> float:
    relevances = [1.0 if corpus_ids[idx] in correct_ids else 0.0 for idx in rankings[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def evaluate_retrieval(
    rankings: list[int],
    correct_ids: set[str],
    corpus_ids: list[str],
    k: int,
) -> tuple[float, float, float]:
    """Returns (recall_any, recall_all, ndcg_score)."""
    top_k_ids = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(cid in top_k_ids for cid in correct_ids))
    recall_all = float(all(cid in top_k_ids for cid in correct_ids))
    ndcg_score = ndcg(rankings, correct_ids, corpus_ids, k)
    return recall_any, recall_all, ndcg_score


async def ingest_sessions(
    db: databases.Database,
    sessions: list[list[dict]],
    session_ids: list[str],
) -> None:
    """Ingest conversation sessions into recall's chunk store."""
    for session, sess_id in zip(sessions, session_ids):
        user_turns = [t["content"] for t in session if t["role"] == "user"]
        if not user_turns:
            continue

        doc = "\n".join(user_turns)
        chunk_hash = hashlib.sha256(f"{sess_id}:{doc}".encode()).hexdigest()
        embedding = embeddings.embed_query(doc)

        await store.insert_chunk(
            db,
            content=doc,
            chunk_hash=chunk_hash,
            embedding=embedding,
            source_file=sess_id,
        )


async def query_and_rank(
    db: databases.Database,
    question: str,
    corpus_ids: list[str],
    top_k: int = 50,
) -> list[int]:
    """Query recall and return rankings as indices into corpus_ids."""
    results, _ = await retrieval.query(db, question, top_k=top_k)

    rankings = []
    seen: set[str] = set()
    for r in results:
        sess_id = r.source_file
        if sess_id is None or sess_id in seen:
            continue
        seen.add(sess_id)
        try:
            rankings.append(corpus_ids.index(sess_id))
        except ValueError:
            continue

    return rankings


async def run_benchmark(data_path: str, limit: int | None = None) -> None:
    with open(data_path) as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    total = len(dataset)
    print(f"Running LongMemEval benchmark on {total} questions...")
    print(f"Embedding model: {settings_module.settings.embedding_model}")
    print(f"Retrieval: hybrid BM25 + dense (RRF fusion)")
    print()

    k_values = [1, 3, 5, 10, 30, 50]
    metrics: dict[str, list[float]] = {}
    for k in k_values:
        metrics[f"recall_any@{k}"] = []
        metrics[f"ndcg@{k}"] = []

    type_metrics: dict[str, dict[str, list[float]]] = {}

    tmp_dir = Path(tempfile.mkdtemp())
    original_db_path = settings_module.settings.db_path

    t0 = time.time()

    try:
        for qi, entry in enumerate(dataset):
            question = entry["question"]
            q_type = entry["question_type"]
            correct_session_ids = set(entry["answer_session_ids"])

            sessions = entry["haystack_sessions"]
            session_ids = entry["haystack_session_ids"]

            # Fresh temp DB per question (matches mempalace's fresh collection)
            bench_db_path = tmp_dir / f"bench_{qi}.db"
            settings_module.settings.db_path = bench_db_path

            db = await store.get_database()

            try:
                await ingest_sessions(db, sessions, session_ids)
                rankings = await query_and_rank(db, question, session_ids, top_k=50)

                for k in k_values:
                    r_any, r_all, ndcg_score = evaluate_retrieval(
                        rankings, correct_session_ids, session_ids, k,
                    )
                    metrics[f"recall_any@{k}"].append(r_any)
                    metrics[f"ndcg@{k}"].append(ndcg_score)

                    if q_type not in type_metrics:
                        type_metrics[q_type] = {
                            f"recall_any@{k2}": [] for k2 in k_values
                        }
                    type_metrics[q_type][f"recall_any@{k}"].append(r_any)

            finally:
                await db.disconnect()

            # Clean up temp DB
            bench_db_path.unlink(missing_ok=True)
            Path(f"{bench_db_path}-wal").unlink(missing_ok=True)
            Path(f"{bench_db_path}-shm").unlink(missing_ok=True)

            if (qi + 1) % 10 == 0 or qi == total - 1:
                elapsed = time.time() - t0
                r5 = sum(metrics["recall_any@5"]) / len(metrics["recall_any@5"]) * 100
                print(
                    f"  [{qi+1}/{total}] R@5: {r5:.1f}% "
                    f"({elapsed:.0f}s, {elapsed/(qi+1):.1f}s/q)",
                    flush=True,
                )

    finally:
        settings_module.settings.db_path = original_db_path
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"RECALL — LongMemEval Benchmark Results ({total} questions)")
    print(f"Embedding: {settings_module.settings.embedding_model}")
    print(f"Retrieval: hybrid BM25 + dense (RRF fusion)")
    print(f"Time: {elapsed:.0f}s ({elapsed/total:.1f}s/question)")
    print("=" * 60)

    print()
    print("Overall:")
    for k in k_values:
        r_any = sum(metrics[f"recall_any@{k}"]) / len(metrics[f"recall_any@{k}"]) * 100
        ndcg_avg = sum(metrics[f"ndcg@{k}"]) / len(metrics[f"ndcg@{k}"]) * 100
        marker = " <<<" if k == 5 else ""
        print(f"  R@{k:<3} {r_any:5.1f}%   NDCG@{k:<3} {ndcg_avg:5.1f}%{marker}")

    print()
    print("By question type:")
    for q_type, tmetrics in sorted(type_metrics.items()):
        count = len(tmetrics["recall_any@5"])
        r5 = sum(tmetrics["recall_any@5"]) / count * 100
        r10 = sum(tmetrics["recall_any@10"]) / count * 100
        print(f"  {q_type:<30} R@5: {r5:5.1f}%  R@10: {r10:5.1f}%  (n={count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for recall")
    parser.add_argument("data", help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--limit", type=int, help="Only run first N questions")
    args = parser.parse_args()

    settings_module.settings.debug = False
    asyncio.run(run_benchmark(args.data, args.limit))


if __name__ == "__main__":
    main()
