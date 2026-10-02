"""Keep the layers apart: the web app and CLI go through repositories and services."""

import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "vouch"
# `vouch db check` runs raw SQL on purpose (server version, pgvector, migration); it reads no data.
QUERY = re.compile(r"\bselect\(|\bsession\.(get|scalars?|execute|add|delete)\(")


def test_web_and_cli_do_not_query_the_database():
    files = [*(SRC / "web").glob("*.py"), SRC / "cli.py"]
    offenders = [
        f"{f.relative_to(SRC)}:{n}: {line.strip()}"
        for f in files
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
        if QUERY.search(line)
    ]
    assert offenders == []
