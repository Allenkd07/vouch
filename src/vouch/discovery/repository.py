"""Reading and writing job embeddings and match scores."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from vouch.db import JobEmbedding, Match
from vouch.discovery.fit import Fit


def embeddings_for(session: Session, job_ids: list[int]) -> dict[int, JobEmbedding]:
    rows = session.scalars(select(JobEmbedding).where(JobEmbedding.job_id.in_(job_ids)))
    return {e.job_id: e for e in rows}


def save_embeddings(session: Session, rows: list[JobEmbedding]) -> None:
    session.add_all(rows)
    session.flush()


def similarities(
    session: Session, model: str, job_ids: list[int], vector: list[float]
) -> dict[int, float]:
    """Cosine similarity of each job's stored `model` embedding to `vector`, computed in
    Postgres by pgvector (vectors are unit length)."""
    rows = session.execute(
        select(JobEmbedding.job_id, 1 - JobEmbedding.embedding.cosine_distance(vector)).where(
            JobEmbedding.model == model, JobEmbedding.job_id.in_(job_ids)
        )
    ).all()
    return {job_id: float(sim) for job_id, sim in rows}


def stored_similarities(session: Session, job_ids: list[int]) -> dict[int, float]:
    """The similarity each job got in the last run that computed one."""
    rows = session.execute(
        select(Match.job_id, Match.similarity).where(
            Match.job_id.in_(job_ids), Match.similarity.is_not(None)
        )
    ).all()
    return {job_id: sim for job_id, sim in rows}


def get_match(session: Session, job_id: int) -> Match | None:
    return session.get(Match, job_id)


def upsert_match(session: Session, job_id: int, fit: Fit | None, similarity: float | None) -> None:
    values = {
        "job_id": job_id,
        "score": fit.score if fit else None,
        "similarity": similarity,
        "fit": fit.model_dump() if fit else None,
        "updated_at": datetime.now(UTC),
    }
    stmt = insert(Match).values(values)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Match.job_id],
            set_={k: stmt.excluded[k] for k in ("score", "similarity", "fit", "updated_at")},
        )
    )
