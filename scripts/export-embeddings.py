#!/usr/bin/env python3
"""Export recall embeddings for visualization in TensorFlow Embedding Projector.

Usage:
    python scripts/export-embeddings.py [output_dir]

Output dir defaults to ~/Desktop. Produces two files:
    recall-embeddings.tsv  — one row per vector (tab-separated dimensions)
    recall-metadata.tsv    — labels and sources for each point

Then open https://projector.tensorflow.org and load both files.
"""

import asyncio
import re
import sys
from pathlib import Path

from recall import store


def sanitize(text: str) -> str:
    return re.sub(r"[\t\n\r]+", " ", text).strip()


async def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop"
    output_dir.mkdir(parents=True, exist_ok=True)

    db = await store.get_database()
    chunks = await store.get_all_chunks(db)

    emb_path = output_dir / "recall-embeddings.tsv"
    meta_path = output_dir / "recall-metadata.tsv"

    with open(emb_path, "w") as f:
        for c in chunks:
            f.write("\t".join(str(x) for x in c.embedding) + "\n")

    with open(meta_path, "w") as f:
        f.write("label\tsource\n")
        for c in chunks:
            source = sanitize(c.source_file or "remembered")
            content = c.content
            prefix = f"{c.source_file}\n\n"
            if content.startswith(prefix):
                content = content[len(prefix):]
            label = sanitize(content[:150])
            f.write(f"{label}\t{source}\n")

    print(f"Exported {len(chunks)} vectors")
    print(f"  {emb_path}")
    print(f"  {meta_path}")
    print(f"\nOpen https://projector.tensorflow.org → Load → upload both files")

    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
