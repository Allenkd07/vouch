from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# What an LLM is used for; each role can run on a different model.
Role = Literal["tailoring", "extraction"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://jobmatcher:jobmatcher@localhost:5433/jobmatcher"

    llm_provider: Literal["gemini", "fake"] = "gemini"
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"  # resume tailoring (evidence, rewrite, honesty check)
    # Requirement extraction runs far more often (every discovered job), so it uses a cheaper
    # model with its own daily quota.
    extraction_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-2"

    def model_for(self, role: Role) -> str:
        return {"tailoring": self.llm_model, "extraction": self.extraction_model}[role]


@lru_cache
def get_settings() -> Settings:
    """Settings from the environment and .env. Call only from vouch.bootstrap (and db.py)."""
    return Settings()
