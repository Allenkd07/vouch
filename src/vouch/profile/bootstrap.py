"""One-time: turn existing resumes into a draft master profile for manual review."""

from pathlib import Path

from vouch.llm import LLM
from vouch.profile.schema import Profile

SYSTEM = """You convert resumes into a structured master profile.
Rules:
- Copy facts only. Never invent, infer or embellish tools, numbers, titles, dates or scope.
- If several resumes are given, merge them: one entry per job/project, keep every distinct bullet,
  and drop only exact or near-exact duplicates (keep the more specific wording).
- facts.tools: only technologies literally named in that bullet.
- facts.metrics: numbers exactly as written in that bullet (e.g. "40%", "2M records/day").
- facts.scope: team size, users, ownership stated in that bullet (e.g. "led team of 3").
- ids: short lowercase kebab-case, unique across the whole profile. Bullet ids are prefixed
  with their job/project id, e.g. "acme-1", "acme-2".
- Dates as YYYY-MM when the month is known, else YYYY; end is null for a current role.
- tags: 1-3 broad areas per bullet, e.g. backend, frontend, ml, data, devops, leadership."""


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    if suffix == ".docx":
        import docx

        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"unsupported file type: {path.name} (use .pdf, .docx, .txt or .md)")


def bootstrap_profile(paths: list[Path], llm: LLM) -> Profile:
    parts = [f"=== Resume: {p.name} ===\n{read_document(p).strip()}" for p in paths]
    prompt = "Build the master profile from these resumes.\n\n" + "\n\n".join(parts)
    return llm.generate_json(prompt, Profile, system=SYSTEM)
