"""File discovery and text extraction for ingestion."""

import hashlib
import json
import mailbox
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from recall import parsers
from recall import settings as settings_module

MIN_CHUNK_CHARS = 40

# Matches: [2024-01-15], [2024-01], [2024], [2024-Q2], [2024-H2],
# [2024-01 to 2024-03], [2024-01-01 to 2024-01-15]
_FACT_DATE_RE = re.compile(r"^\[(\d{4}(?:-(?:Q[1-4]|H[12]|\d{2}(?:-\d{2})?))?(?:\s+to\s+\d{4}(?:-\d{2}(?:-\d{2})?)?)?)\]")

SUPPORTED_EXTENSIONS = {".md", ".json", ".pdf", ".mbox", ".docx", ".csv"}
IGNORED_FILENAMES = {"package-lock.json", "package.json", "node_modules", ".DS_Store"}


def _debug_print(msg: str) -> None:
    if settings_module.settings.debug:
        print(f"  [chunker] {msg}", flush=True)


@dataclass
class Chunk:
    content: str
    source_file: str
    section_path: str
    chunk_hash: str
    fact_date: str | None


def _compute_hash(text: str, source_file: str = "") -> str:
    return hashlib.sha256(f"{source_file}:{text}".encode()).hexdigest()


def parse_fact_date(text: str) -> str | None:
    """Extract a date from a [YYYY-MM-DD] prefix, if present."""
    match = _FACT_DATE_RE.match(text.strip())
    return match.group(1) if match else None


def build_chunk(statement: str, section_path: str, source_file: str) -> Chunk | None:
    """Build a chunk from a statement. Returns None if too short."""
    statement = statement.strip()
    if len(statement) < MIN_CHUNK_CHARS:
        return None

    fact_date = parse_fact_date(statement)

    if section_path:
        content = f"{section_path}\n\n{statement}"
    else:
        content = statement

    return Chunk(
        content=content,
        source_file=source_file,
        section_path=section_path or "(top-level)",
        chunk_hash=_compute_hash(content, source_file),
        fact_date=fact_date,
    )


def list_supported_files(directory: Path) -> list[Path]:
    """Find all supported files in a directory."""
    return sorted(
        f
        for f in directory.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.name not in IGNORED_FILENAMES
    )


def read_file_text(path: Path) -> str:
    """Read a file's text content, handling different formats."""
    suffix = path.suffix.lower()

    if suffix == ".md":
        return path.read_text(encoding="utf-8")

    elif suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ""

        # Discord export: convert messages to readable chat
        if isinstance(data, dict) and "guild" in data and "messages" in data:
            channel = data.get("channel", {}).get("name", "unknown")
            lines: list[str] = []
            for msg in data.get("messages", []):
                content = msg.get("content", "").strip()
                if not content:
                    continue
                author = msg.get("author", {})
                if author.get("isBot", False):
                    continue
                name = author.get("nickname") or author.get("name", "unknown")
                ts = msg.get("timestamp", "")[:10]
                lines.append(f"[{ts}] {name}: {content}")
            return f"Discord DM: {channel}\n\n" + "\n".join(lines)

        # Slack export: convert messages to readable chat
        if isinstance(data, dict) and "exported_at" in data and "messages" in data:
            users = {
                uid: info.get("real_name", uid)
                for uid, info in data.get("users", {}).items()
            }
            lines = []
            for msg in data.get("messages", []):
                text = msg.get("text", "").strip()
                if not text:
                    continue
                user_id = msg.get("user", "")
                if user_id == "USLACKBOT":
                    continue
                name = users.get(user_id, user_id)
                date = msg.get("date", "")[:10]
                lines.append(f"[{date}] {name}: {text}")
            return "Slack DM\n\n" + "\n".join(lines)

        # Other JSON: return as-is
        return path.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        doc = pymupdf.open(str(path))
        text = "\n\n".join(
            page.get_text().strip() for page in doc if page.get_text().strip()
        )
        doc.close()
        return text

    elif suffix == ".mbox":
        # Automated/bulk sender patterns to skip
        skip_senders = {
            # Generic automated
            "noreply",
            "no-reply",
            "no_reply",
            "notifications",
            "mailer-daemon",
            "donotreply",
            "newsletter",
            "marketing",
            "billing@",
            "receipts@",
            "updates@",
            "shipment-tracking",
            "auto-confirm",
            "alerts@",
            "digest@",
        }

        # Calendar invite subject patterns to skip
        calendar_patterns = {
            "invitation:",
            "updated invitation:",
            "accepted:",
            "declined:",
            "tentatively accepted:",
            "canceled event:",
            "cancelled event:",
        }

        mbox = mailbox.mbox(str(path))
        candidates: list[tuple[int, str]] = []  # (body_length, formatted_email)
        skipped = 0

        for msg in mbox:
            sender = (msg.get("from", "") or "").lower()

            # Skip automated emails
            if any(kw in sender for kw in skip_senders):
                skipped += 1
                continue

            subject = msg.get("subject", "(no subject)")

            # Skip calendar invites
            if any(subject.lower().startswith(p) for p in calendar_patterns):
                skipped += 1
                continue

            date = msg.get("date", "")
            body = parsers._extract_email_body(msg)

            if not body or len(body.strip()) < 50:
                skipped += 1
                continue

            # Truncate long emails
            if len(body) > 1000:
                body = body[:1000] + "..."

            formatted = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{body}"
            candidates.append((len(body), formatted))

        # Sort by body length descending (longest = most substantive first)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Apply max_emails limit
        max_emails = settings_module.settings.max_emails
        if max_emails is not None:
            candidates = candidates[:max_emails]

        _debug_print(f"mbox: kept {len(candidates)} emails, skipped {skipped}")
        return "\n\n---\n\n".join(text for _, text in candidates)

    elif suffix == ".docx":
        with zipfile.ZipFile(str(path)) as z:
            xml_content = z.read("word/document.xml")
        root = ET.fromstring(xml_content)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return "\n".join(
            "".join(node.text or "" for node in para.findall(".//w:t", ns))
            for para in root.findall(".//w:p", ns)
        )

    elif suffix == ".csv":
        return path.read_text(encoding="utf-8")

    return ""
