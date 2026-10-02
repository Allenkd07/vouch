"""Keep the layers apart: the web app and CLI go through repositories and services."""

import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "vouch"
# `vouch db check` runs raw SQL on purpose (server version, pgvector, migration); it reads no data.
QUERY = re.compile(r"\bselect\(|\bsession\.(get|scalars?|execute|add|delete|commit|rollback)\(")


def test_web_and_cli_neither_query_nor_commit():
    files = [*(SRC / "web").glob("*.py"), SRC / "cli.py"]
    offenders = [
        f"{f.relative_to(SRC)}:{n}: {line.strip()}"
        for f in files
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
        if QUERY.search(line)
    ]
    assert offenders == []


def test_only_the_composition_root_reads_settings():
    # db.py builds the engine from settings; everything else gets them through vouch.bootstrap.
    allowed = {"bootstrap.py", "config.py", "db.py"}
    offenders = [
        str(f.relative_to(SRC))
        for f in SRC.rglob("*.py")
        if f.name not in allowed and "get_settings" in f.read_text(encoding="utf-8")
    ]
    assert offenders == []
