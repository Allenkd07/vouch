import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from vouch.config import get_settings


@pytest.fixture
def session():
    """Session on the Docker Postgres inside a transaction that is always rolled back.

    Needs `docker compose up -d` and `alembic upgrade head`; skips otherwise."""
    engine = create_engine(get_settings().database_url)
    try:
        conn = engine.connect()
        conn.execute(text("SELECT 1 FROM jobs LIMIT 1"))
        conn.rollback()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not available: {e}")
    trans = conn.begin()
    s = Session(bind=conn, join_transaction_mode="create_savepoint")
    yield s
    s.close()
    trans.rollback()
    conn.close()
