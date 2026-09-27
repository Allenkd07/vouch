# Vouch

A job search tool that tailors your resume to each job using only claims your profile can vouch
for. Every rewritten line is checked against the facts it came from; anything unsupported falls
back to your original wording. It also finds and ranks open jobs at companies you choose, and
tracks your applications. Voice mock interviews are on the roadmap.

**Status:** Stages 0–4 done (setup, master profile, job intake, truthful resume tailoring, web UI,
job discovery and ranking).

## Setup

Requires Python 3.12+, Docker, and [uv](https://docs.astral.sh/uv/) (`pip install --user uv`;
if `uv` isn't on PATH, use `python -m uv`).

```bash
uv sync                          # install dependencies into .venv
cp .env.example .env             # then set GEMINI_API_KEY
docker compose up -d             # Postgres 16 + pgvector on localhost:5433
uv run alembic upgrade head      # create tables
uv run vouch db check               # -> postgres 16.x, pgvector 0.x, migration 0001
uv run vouch llm check              # -> one Gemini generation + one embedding
```

## Master profile

`profile/profile.yaml` is the source of truth for everything a tailored resume may claim.
It is git-ignored (personal data). See `profile/profile.example.yaml` for the format.

```bash
# 1. Put your current resumes (.pdf/.docx/.txt/.md) in resumes/, then draft the profile:
uv run vouch profile bootstrap resumes/*
# 2. Review profile/profile.yaml by hand: fix wording, fill in facts, add missing work.
uv run vouch profile validate       # schema errors + quality warnings
# 3. Store it in the database:
uv run vouch profile sync
```

Each bullet lists its `facts` (tools, metrics, scope). Tailoring may rephrase a bullet but may
never claim anything outside those facts, so the more complete the facts, the better the tailoring.

## Jobs

```bash
uv run vouch job add <job-url>                # Workday, Greenhouse, Lever, Ashby, or any page with JSON-LD
uv run vouch job add -f posting.txt --company X --title Y   # pasted description
uv run vouch job list
uv run vouch job show <id> [--description]
```

`job add` stores the posting and extracts must-haves, nice-to-haves and ATS keywords. Every
requirement must quote the posting (unmatched quotes are flagged) and keywords must appear
verbatim, so hallucinated requirements are caught. Re-adding an unchanged posting reuses the
stored analysis.

## Tailored resume

```bash
uv run vouch tailor <job-id>          # -> output/job-<id>-.../v<n>/ : PDF, DOCX, report.md
uv run vouch tailor <job-id> --pages 2
```

Pipeline (`src/vouch/tailoring/`): map each requirement to evidence bullets (the whole
profile goes in one prompt) → select bullets (every evidenced must-have first, then by relevance)
→ rewrite them in the posting's language → verify each rewrite (no tool or number absent from
the source bullet; an LLM judge flags overclaims such as "CI/CD" when only CI is evidenced) →
retry rejected items once with the reasons → fall back to the original wording if still
rejected → render with Typst, dropping the weakest bullets until the page limit holds.

`report.md` shows requirement coverage (strong / partial / education / skills-only / gap), ATS
keyword coverage before and after, and every change with its original. Each run is stored in
`resume_versions` with the profile snapshot it came from. Read the report before sending.

Each tailoring run makes ~3–5 Gemini calls. The free tier allows ~20 requests/day per model; on
a quota (429) or overload (503) error, retry later or set `LLM_MODEL` to another model.

## Job discovery

```bash
cp profile/search.example.yaml profile/search.yaml   # companies + filters (git-ignored)
uv run vouch discover --check      # fetch and filter only: no database, no Gemini
uv run vouch discover              # fetch, store, rank, extract requirements for the top few
uv run vouch matches               # ranked list + skills your best matches ask for
```

To find more companies to watch, list names (one per line) and probe them. It tries the usual
board spellings on Greenhouse, Lever and Ashby and prints the ones with openings in India, with
sample titles so a same-named company can be spotted, plus lines to paste into `search.yaml`:

```bash
uv run vouch companies probe --file data/india_companies.txt
uv run vouch companies probe "Pine Labs" Razorpay
```

A funnel from free to expensive, so the Gemini free tier is enough:

1. **Fetch** every open job from the companies' public boards (Greenhouse, Lever, Ashby,
   Workday). Free.
2. **Filter** by title and location rules from `search.yaml`. Free.
3. **Rank by similarity**: embed new or changed postings and the profile (pgvector cosine
   similarity). Unchanged postings are never re-embedded.
4. **Extract requirements** for the `analyze_per_run` most similar unanalysed jobs (1 Gemini
   request each); the rest wait for later runs, or use **Read requirements** on the job page.
5. **Score fit** without an LLM: technical requirements that name concrete technologies are
   matched against the profile's skills and bullet facts, with a penalty for seniority gaps.
   The page says how many requirements the score is based on; the rest are checked properly
   when you tailor a resume.

Postings not seen on their board for 3 days drop out of the ranking.

## Web UI

```bash
uv run vouch web                      # http://localhost:8000
```

- **Matches:** discovered jobs ranked by fit, what you match and what's missing, the skills your
  best matches ask for most, and a "Find new jobs" button.
- **Jobs:** add a job by link or pasted text; see requirements met per job; set a status.
- **Job page:** requirements with how the latest resume covers each; tailor a new resume
  (runs in the background, about a minute).
- **Resume review:** words the tailoring introduced are highlighted, rejected rewrites are
  marked with the reason. Per bullet choose tailored / original / your own wording, or leave it
  out; saving rebuilds the PDF and DOCX. Approving attaches the version to the application.
- **Applications:** jobs grouped by status, with notes.
- **Profile:** read-only view of `profile.yaml` with where each bullet came from.

## Tests

```bash
uv run pytest                    # DB tests skip if Postgres isn't running
uv run ruff check .
```
