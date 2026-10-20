from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# What an LLM is used for; each role can run on a different model.
Role = Literal["tailoring", "extraction", "judge"]
Provider = Literal["gemini", "vertex", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://jobmatcher:jobmatcher@localhost:5433/jobmatcher"

    llm_provider: Provider = "gemini"
    gemini_api_key: str = ""
    # Vertex AI (Google Cloud): signs in with Application Default Credentials and bills the project.
    gcp_project: str = ""
    gcp_location: str = "global"
    llm_model: str = "gemini-3.8-flash"  # resume tailoring (evidence, rewrite, honesty check)
    # Requirement extraction runs far more often (every discovered job), so it uses a cheaper
    # model with its own daily quota.
    extraction_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-2"
    # The tailoring eval's faithfulness judge: a different, stronger model than the one that
    # writes and checks rewrites, so it doesn't share its blind spots. Only evals use it.
    judge_provider: Provider | None = None  # None: same as llm_provider
    judge_model: str = "gemini-3.1-pro-preview"

    def model_for(self, role: Role) -> str:
        return {
            "tailoring": self.llm_model,
            "extraction": self.extraction_model,
            "judge": self.judge_model,
        }[role]

    def provider_for(self, role: Role) -> Provider:
        return (self.judge_provider if role == "judge" else None) or self.llm_provider


@lru_cache
def get_settings() -> Settings:
    """Settings from the environment and .env. Call only from vouch.bootstrap (and db.py)."""
    return Settings()
