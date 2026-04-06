import json
import textwrap
from pathlib import Path

from recall import parsers


def test_extract_email_body_plain() -> None:
    import mailbox

    mbox_content = textwrap.dedent("""\
        From sender@example.com Mon Jan 15 10:00:00 2024
        From: sender@example.com
        Subject: Test
        Content-Type: text/plain; charset="utf-8"

        Hello, this is a test email body.
    """)
    path = "/tmp/test_parsers_email.mbox"
    with open(path, "wb") as f:
        f.write(mbox_content.encode())

    mbox = mailbox.mbox(path)
    for msg in mbox:
        body = parsers._extract_email_body(msg)
        assert "test email body" in body
