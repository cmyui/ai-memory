"""LLM-based fact extraction from raw content.

Sends batches of raw text to an LLM and returns clean,
self-contained factual statements suitable for embedding.
"""

import collections.abc

from recall import llm
from recall import settings as settings_module

# Maximum characters to send in a single LLM call
MAX_BATCH_CHARS_OPENAI = 100_000
MAX_BATCH_CHARS_CLAUDE = 500_000


def get_max_batch_chars() -> int:
    return MAX_BATCH_CHARS_CLAUDE if llm._use_claude_cli else MAX_BATCH_CHARS_OPENAI


SYSTEM_PROMPT = """\
You are building a searchable memory database. You are processing one page of a \
larger dataset. Your job is to distill this page into useful knowledge that can be \
found via semantic search later.

{hints}

Synthesize, don't transcribe, with the exception of direct quotes. \
If you are processing a conversation between people/agents, DO NOT produce play-by-play \
summaries like "A said X", "B replied Y", "C sent a GIF". Instead, synthesize the \
*conclusions* and *outcomes*: decisions made, opinions held, plans formed, facts revealed. \
Preserve each participant's distinct perspectives and notable opinions — these are facts \
about those people worth keeping in their own right.

For curated/structured content (tables, lists, notes): extract each meaningful \
entry as its own fact. This data was already human-curated — most of it is worth keeping.

For all content: focus on knowledge the user or an agent might want to look up later — \
biographical facts, decisions, plans, opinions, relationships, projects, finances, principles, \
life events, preferences, habits, agreements and obligations, etc.

Output rules:
- Each statement must be FULLY self-contained — a reader with zero context must \
understand it. Include: which person, which dates, which project, which company, which city, etc.
- NEVER use pronouns (he, she, they, it, his, her) — always use the person's name. \
This is critical: "She passed the driving test" is WRONG, "Alice Johnson passed the driving test" is RIGHT. \
Every single sentence must name the subject explicitly.
- Prefix each fact with a date AND optional category tag. Format: [YYYY-MM-DD] or \
[YYYY-MM] or [YYYY] for the date, and optionally [Category - Subcategory] for topical \
grouping. Examples: "[2024-08] [Shopping - Electronics] Alice Johnson purchased...", \
"[2025-01] [Travel - Italy] Alice Johnson visited...", "[Habits - Rock Climbing] Alice Johnson...". \
The category helps retrieval — use natural categories like Travel, Shopping, Health, \
Habits, Career, Relationships, Finance, Engineering, etc.
- Include full names, locations, amounts, and handles when available
- Preserve usernames/handles as-is — do not resolve aliases
- For plans/intentions, phrase as "as of [date], X plans to..." because later pages may update plans
- Try to keep facts isolated — "one fact per line", without numbering or bullets
- Skip generic boilerplate that applies to everyone (airline baggage policies, check-in \
procedures, generic privacy notices). DO keep terms specific to individuals \
(employment terms, non-compete clauses, salary, lease conditions)
- If nothing worth extracting exists, return NONE: followed by a brief reason why \
(e.g. "NONE: casual chat with no substantive content"). This helps us debug extraction quality.\
"""

# Stored hints from --hint flags, injected into the system prompt
_hints: list[str] = []


def add_hint(hint: str) -> None:
    _hints.append(hint)


def _build_system_prompt() -> str:
    if _hints:
        hints_block = "User-provided context for this dataset:\n" + "\n".join(
            f"- {h}" for h in _hints
        )
    else:
        hints_block = ""
    return SYSTEM_PROMPT.format(hints=hints_block)


def _debug(msg: str) -> None:
    if settings_module.settings.debug:
        print(f"      [debug] {msg}", flush=True)


async def extract_facts(texts: list[str]) -> list[str]:
    """Send a batch of raw texts to the LLM and return extracted facts."""
    combined = "\n\n---\n\n".join(texts)

    _debug(f"sending {len(combined)} chars")
    _debug(f"first 200 chars: {combined[:200]!r}")

    content = await llm.call(_build_system_prompt(), combined)
    if not content:
        return []

    _debug(f"response length: {len(content)} chars")
    _debug(f"response preview: {content[:300]!r}")

    facts = [line.strip() for line in content.split("\n") if line.strip()]
    filtered = []
    for f in facts:
        if f.upper().startswith("NONE"):
            _debug(f"NONE reason: {f}")
        else:
            filtered.append(f)
    _debug(f"{len(facts)} lines \u2192 {len(filtered)} facts after filtering")
    return filtered


async def extract_facts_batched(texts: list[str]) -> list[str]:
    """Process a large list of texts in batches by character count."""
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_chars = 0

    for text in texts:
        text_len = len(text)
        if current_chars + text_len > get_max_batch_chars() and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_chars = text_len
        else:
            current_batch.append(text)
            current_chars += text_len

    if current_batch:
        batches.append(current_batch)

    all_facts: list[str] = []
    for i, batch in enumerate(batches, 1):
        facts = await extract_facts(batch)
        all_facts.extend(facts)

        print(
            f"    extracted {len(facts)} facts from batch {i}/{len(batches)}",
            flush=True,
        )
        for fact in facts:
            print(f"      • {fact}", flush=True)

    return all_facts


async def extract_facts_streaming(
    texts: list[str],
) -> collections.abc.AsyncIterator[list[str]]:
    """Like extract_facts_batched, but yields facts per batch for incremental saving."""
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_chars = 0

    for text in texts:
        text_len = len(text)
        if current_chars + text_len > get_max_batch_chars() and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_chars = text_len
        else:
            current_batch.append(text)
            current_chars += text_len

    if current_batch:
        batches.append(current_batch)

    for i, batch in enumerate(batches, 1):
        facts = await extract_facts(batch)

        print(
            f"    extracted {len(facts)} facts from batch {i}/{len(batches)}",
            flush=True,
        )
        for fact in facts:
            print(f"      • {fact}", flush=True)

        yield facts
