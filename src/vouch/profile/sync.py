import hashlib
import json
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from vouch.db import ProfileItem, ProfileSnapshot
from vouch.profile.schema import Profile


@dataclass
class SyncResult:
    snapshot_id: int
    new_snapshot: bool
    items: int
    removed: int


def profile_hash(profile: Profile) -> str:
    canonical = json.dumps(profile.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def sync_profile(session: Session, profile: Profile) -> SyncResult:
    """Store a snapshot (if the profile changed) and make profile_items match it exactly."""
    digest = profile_hash(profile)
    snapshot = session.scalar(select(ProfileSnapshot).where(ProfileSnapshot.content_hash == digest))
    new_snapshot = snapshot is None
    if snapshot is None:
        snapshot = ProfileSnapshot(content_hash=digest, content=profile.model_dump(mode="json"))
        session.add(snapshot)
        session.flush()

    rows = [
        {
            "id": b.id,
            "section": section,
            "text": b.text,
            "facts": b.facts.model_dump(),
            "tags": b.tags,
            "snapshot_id": snapshot.id,
        }
        for section, b in profile.iter_bullets()
    ]
    if rows:
        stmt = insert(ProfileItem).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[ProfileItem.id],
            set_={c: stmt.excluded[c] for c in ("section", "text", "facts", "tags", "snapshot_id")},
        )
        session.execute(stmt)

    removed = session.execute(
        delete(ProfileItem).where(ProfileItem.id.not_in([r["id"] for r in rows] or [""]))
    ).rowcount
    session.commit()
    return SyncResult(snapshot.id, new_snapshot, len(rows), removed)
