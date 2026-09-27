from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://jobmatcher:jobmatcher@localhost:5433/jobmatcher"

    llm_provider: Literal["gemini", "fake"] = "gemini"
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
