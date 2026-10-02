from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session


@contextmanager
def transaction(session: Session) -> Iterator[Session]:
    """Commit what the block wrote, or roll all of it back if the block raises.

    Services wrap their writes in this, so callers never commit or roll back themselves."""
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
