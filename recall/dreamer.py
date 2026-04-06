"""Dream cycle — consolidates, deduplicates, and builds peer cards from memories.

Run via `recall dream`. Processes all memories in the database and:
1. Identifies entities (people) mentioned across memories
2. For each entity, gathers all related facts
3. Consolidates into a peer card (40-fact biographical summary)
4. Resolves aliases and temporal supersession
5. Stores peer cards as searchable memories
"""

import hashlib

import databases

from recall import embeddings
from recall import llm
from recall import settings as settings_module
from recall import store

DREAM_MODEL = "gpt-4.1-mini"

MAX_PEER_CARD_FACTS = 40

ENTITY_EXTRACTION_PROMPT = """\
You are analyzing a memory database. Given a batch of memories, identify all \
distinct people/entities mentioned. For each person, list all known aliases.

Return one entity per line in this format:
NAME | alias1, alias2, alias3

Rules:
- Merge aliases that clearly refer to the same person (e.g. "ally_j" and "alice.johnson99" and "Alice Johnson")
- Include Discord handles, real names, nicknames
- Only include people, not organizations or places
- If nothing found, return NONE

Example output:
Alice Johnson | ally_j, alice.johnson99, Alice Marie Johnson
Bob Chen | None
Charlie | Charlie Rivera, charlie_dev
"""

PEER_CARD_PROMPT = """\
You are building a biographical peer card for {entity_name}. A peer card is a \
concise, searchable summary of everything known about this person — like a \
dossier that an AI agent can quickly reference.

Given the memories below about {entity_name}, create a peer card with up to \
{max_facts} key facts. Prioritize:
1. Identity (full name, aliases, handles, age, location)
2. Relationship to the user (how they met, nature of relationship)
3. Biographical facts (job, education, family, background)
4. Personality traits and values
5. Key life events and timeline
6. Current status and plans

Rules:
- ONLY include facts where {entity_name} is the primary subject. Discard facts \
where {entity_name} is merely mentioned but the fact is primarily about someone else.
- Each fact must be self-contained — name the person explicitly
- Include dates where known
- Resolve contradictions — keep the most recent/accurate version
- For temporal facts (plans, locations), note the date to indicate currency
- Merge duplicate facts into the best version
- One fact per line, no numbering

The known aliases for {entity_name} are: {known_aliases}. \
If aliases exist, the FIRST line should be:
ALIASES: name1, name2, name3
Only include aliases that belong to {entity_name} — not aliases of other people \
mentioned in the same memories.
"""

CONSOLIDATION_PROMPT = """\
You are cleaning up a memory database. Given these memories, identify and fix:

1. **Duplicates**: Facts that say the same thing in different words → keep the best version
2. **Superseded facts**: Plans/intentions that were later updated → mark the old ones for deletion
3. **Contradictions**: Facts that conflict → keep the most recent/reliable one

For each action, output one line:
KEEP: <the fact to keep, possibly improved/merged>
DELETE: <the fact to remove, quoted exactly as provided>

If no cleanup needed, return NONE.

Rules:
- Only act on clear duplicates/contradictions — don't delete facts that are merely similar
- When merging, preserve all unique details from both versions
- Preserve dates and specifics
"""


def _debug(msg: str) -> None:
    if settings_module.settings.debug:
        print(f"  [dream] {msg}", flush=True)


async def _llm_call(prompt: str, content: str) -> str:
    return await llm.call(prompt, content, default_model=DREAM_MODEL)


async def extract_entities(db: databases.Database) -> list[dict[str, list[str] | str]]:
    """Scan all memories and identify distinct people/entities."""
    all_chunks = await store.get_all_chunks(db, exclude_source_types={"dream"})

    _debug(f"scanning {len(all_chunks)} memories for entities (excluding dream-generated)")

    # Sample memories to find entities (don't send all 1400+ to the LLM)
    sample_size = min(200, len(all_chunks))
    # Take evenly spaced samples
    step = max(1, len(all_chunks) // sample_size)
    sampled = [all_chunks[i] for i in range(0, len(all_chunks), step)][:sample_size]

    content = "\n".join(c.content[:200] for c in sampled)
    response = await _llm_call(ENTITY_EXTRACTION_PROMPT, content)

    entities = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        if "|" in line:
            name, aliases_str = line.split("|", 1)
            aliases = [a.strip() for a in aliases_str.split(",") if a.strip() and a.strip().lower() != "none"]
            entities.append({"name": name.strip(), "aliases": aliases})
        else:
            entities.append({"name": line.strip(), "aliases": []})

    _debug(f"found {len(entities)} entities")
    for e in entities:
        alias_str = ", ".join(e["aliases"]) if e["aliases"] else "no aliases"
        _debug(f"  {e['name']} ({alias_str})")

    return entities


async def build_peer_card(
    db: databases.Database,
    entity: dict[str, list[str] | str],
) -> list[str]:
    """Gather all facts about an entity and build a peer card."""
    name = entity["name"]
    search_terms = [name] + entity.get("aliases", [])

    # Search for all memories mentioning this entity
    all_facts: list[str] = []
    seen_ids: set[int] = set()

    for term in search_terms:
        # Use BM25 for keyword matching
        chunk_ids = await store.bm25_search(db, term, top_k=100)
        for cid in chunk_ids:
            if cid not in seen_ids:
                seen_ids.add(cid)

    # Fetch the actual content for matched chunks (excluding dream-generated)
    all_chunks = await store.get_all_chunks(db, exclude_source_types={"dream"})
    chunk_by_id = {c.id: c for c in all_chunks}
    for cid in seen_ids:
        if cid in chunk_by_id:
            all_facts.append(chunk_by_id[cid].content)

    if not all_facts:
        _debug(f"  no facts found for {name}")
        return []

    _debug(f"  found {len(all_facts)} facts about {name}")

    # Build the peer card
    facts_text = "\n\n".join(all_facts[:100])  # Cap at 100 facts to fit context
    aliases = entity.get("aliases", [])
    known_aliases_str = ", ".join(aliases) if aliases else "none known"
    prompt = PEER_CARD_PROMPT.format(
        entity_name=name,
        known_aliases=known_aliases_str,
        max_facts=MAX_PEER_CARD_FACTS,
    )
    response = await _llm_call(prompt, facts_text)

    card_facts = [line.strip() for line in response.split("\n") if line.strip()]
    _debug(f"  peer card: {len(card_facts)} facts")
    for fact in card_facts[:5]:
        _debug(f"    • {fact}")
    if len(card_facts) > 5:
        _debug(f"    ... +{len(card_facts) - 5} more")

    return card_facts


async def run_dream(db: databases.Database, *, dry_run: bool = False) -> dict[str, int]:
    """Run the full dream cycle."""
    if dry_run:
        print("Starting dream cycle (DRY RUN — no writes)...", flush=True)
    else:
        print("Starting dream cycle...", flush=True)

    # Step 1: Extract entities
    print("\n1. Extracting entities...", flush=True)
    entities = await extract_entities(db)

    if not entities:
        print("No entities found. Nothing to dream about.")
        return {"entities": 0, "cards_built": 0}

    # Step 2: Build peer cards for top entities
    print(f"\n2. Building peer cards for {len(entities)} entities...", flush=True)
    cards_built = 0

    for entity in entities:
        name = entity["name"]
        print(f"\n  Building card for: {name}...", flush=True)

        card_facts = await build_peer_card(db, entity)
        if not card_facts:
            continue

        # Print the full card
        print(f"  Peer card for {name} ({len(card_facts)} facts):", flush=True)
        for fact in card_facts:
            print(f"    • {fact}", flush=True)

        if dry_run:
            cards_built += 1
            print(f"  [dry-run] would store {len(card_facts)} facts for {name}", flush=True)
            continue

        # Store peer card as memories
        source_file = f"peer-card/{name}"

        # Delete old peer card if exists
        await store.delete_by_source(db, source_file)

        # Build card header with aliases
        card_content = "\n".join(card_facts)
        card_hash = hashlib.sha256(f"{source_file}:{card_content}".encode()).hexdigest()

        # Embed and store the full card as one memory
        embedding = embeddings.embed_query(card_content)
        await store.insert_chunk(
            db,
            content=f"[Peer Card: {name}]\n{card_content}",
            chunk_hash=card_hash,
            embedding=embedding,
            source_file=source_file,
            source_type="dream",
        )

        # Also store individual facts for granular retrieval
        fact_embeddings = embeddings.embed_texts(card_facts)
        chunk_dicts = []
        for fact, emb in zip(card_facts, fact_embeddings):
            fact_hash = hashlib.sha256(f"{source_file}:{fact}".encode()).hexdigest()
            chunk_dicts.append({
                "content": fact,
                "section_path": f"Peer Card: {name}",
                "chunk_hash": fact_hash,
                "embedding": emb,
            })

        await store.upsert_chunks(
            db,
            chunks=chunk_dicts,
            source_file=f"peer-card-facts/{name}",
            file_hash=card_hash,
            source_type="dream",
        )

        cards_built += 1
        print(f"  ✓ {name}: {len(card_facts)} facts stored", flush=True)

    print(f"\nDream cycle complete: {len(entities)} entities, {cards_built} cards built.", flush=True)
    return {"entities": len(entities), "cards_built": cards_built}
