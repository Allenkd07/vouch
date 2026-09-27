"""LLM access behind one small interface, so tests and offline work use a fake."""

import json
from collections.abc import Callable
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from vouch.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def generate_json(self, prompt: str, schema: type[T], *, system: str | None = None) -> T: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _describe(error, model: str) -> str:
    """Short, actionable message for the common Gemini API failures."""
    code = getattr(error, "code", None)
    if code == 429:
        return (
            f"Gemini quota exhausted for {model} (the free tier allows ~20 requests/day per "
            "model). Wait for the daily reset, set LLM_MODEL to another model, or enable billing."
        )
    if code == 503:
        return f"{model} is overloaded right now (503). Try again in a few minutes."
    return f"Gemini API error {code}: {getattr(error, 'message', error)}"


def _schema_prompt(prompt: str, schema: type[BaseModel]) -> str:
    return (
        f"{prompt}\n\n"
        "Respond with a single JSON object that validates against this JSON schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )


class GeminiLLM:
    """Gemini via google-genai.

    The JSON schema goes in the prompt and the reply is validated with Pydantic, instead of
    using Gemini's response_schema (which rejects some Pydantic features, e.g. defaults).
    One retry feeds the validation error back to the model.
    """

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        from google import genai

        self._genai = genai
        # Retry rate limits (429) and "model overloaded" (503) with exponential backoff.
        retry = genai.types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=30.0)
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=genai.types.HttpOptions(retry_options=retry),
        )
        self._model = settings.llm_model
        self._embedding_model = settings.embedding_model

    def generate_json(self, prompt: str, schema: type[T], *, system: str | None = None) -> T:
        types = self._genai.types
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = _schema_prompt(prompt, schema)
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except self._genai.errors.APIError as e:
                raise LLMError(_describe(e, self._model)) from e
            try:
                return schema.model_validate_json(response.text or "")
            except ValidationError as e:
                last_error = e
                contents = (
                    f"{_schema_prompt(prompt, schema)}\n\n"
                    f"Your previous reply was invalid:\n{response.text}\n\n"
                    f"Validation errors:\n{e}\n\nReturn corrected JSON only."
                )
        raise LLMError(f"Model output did not match {schema.__name__}: {last_error}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.models.embed_content(model=self._embedding_model, contents=texts)
        return [list(e.values) for e in result.embeddings]


class FakeLLM:
    """Deterministic stand-in. `responder` maps (prompt, schema) to a model instance."""

    def __init__(self, responder: Callable[[str, type[BaseModel]], BaseModel] | None = None):
        self._responder = responder
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, schema: type[T], *, system: str | None = None) -> T:
        self.prompts.append(prompt)
        if self._responder is None:
            raise LLMError("FakeLLM has no responder configured")
        return schema.model_validate(self._responder(prompt, schema))

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Stable pseudo-embedding: character histogram, enough for plumbing tests.
        out = []
        for t in texts:
            v = [0.0] * 16
            for ch in t.lower():
                v[ord(ch) % 16] += 1.0
            out.append(v)
        return out


def get_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return FakeLLM()
    return GeminiLLM(settings)
