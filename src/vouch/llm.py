"""LLM access behind one small interface, so tests and offline work use a fake."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from vouch.config import Role, Settings

T = TypeVar("T", bound=BaseModel)
TIMEOUT_MS = 240_000  # per request; a Pro judge call with thinking takes a minute or two
# How much a Gemini 3 model reasons before answering; None leaves the model's default (high).
Thinking = Literal["minimal", "low", "medium", "high"]

# USD per 1M tokens (input, output) on Vertex AI's global endpoint, prompts up to 200k tokens,
# as of 2026-10; thinking tokens are billed as output. Check
# https://cloud.google.com/vertex-ai/generative-ai/pricing when changing models; a model missing
# here is reported in tokens only.
PRICES: dict[str, tuple[float, float]] = {
    # Billed at this rate; the advertised 0.75 / 3.75 (until 2026-12-31) is a 50% credit issued
    # the next month, so spend caps and budgets see the full price.
    "gemini-3.8-flash": (1.50, 7.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}
CACHED_INPUT = 0.1  # implicitly cached prompt tokens cost a tenth of the input price


@dataclass
class Usage:
    """Tokens one model used in this process."""

    model: str
    requests: int = 0
    input: int = 0  # includes the cached tokens
    cached: int = 0
    output: int = 0
    thinking: int = 0

    def add(self, metadata) -> None:
        self.requests += 1
        if metadata:
            self.input += metadata.prompt_token_count or 0
            self.cached += metadata.cached_content_token_count or 0
            self.output += metadata.candidates_token_count or 0
            self.thinking += metadata.thoughts_token_count or 0

    @property
    def cost(self) -> float | None:
        """Estimated USD, or None when the model's price isn't known."""
        if self.model not in PRICES:
            return None
        per_input, per_output = PRICES[self.model]
        input_cost = (self.input - self.cached + self.cached * CACHED_INPUT) * per_input
        return (input_cost + (self.output + self.thinking) * per_output) / 1e6

    def __str__(self) -> str:
        cost = f", ~${self.cost:.2f}" if self.cost is not None else ", price unknown"
        return (
            f"{self.model}: {self.requests} requests, {self.input:,} in ({self.cached:,} cached), "
            f"{self.output:,} out + {self.thinking:,} thinking{cost}"
        )


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        thinking: Thinking | None = None,
    ) -> T: ...

    def embed(self, texts: list[str], dimensions: int | None = None) -> list[list[float]]: ...


def _describe(error, model: str, *, vertex: bool = False) -> str:
    """Short, actionable message for the common Gemini API failures."""
    code = getattr(error, "code", None)
    if code == 429 and vertex:
        return (
            f"Vertex AI quota exhausted for {model} (429: {getattr(error, 'message', error)}). "
            "Usually a per-minute limit, tightest on preview models: try again in a few "
            "minutes, or check IAM & Admin > Quotas in the Cloud console."
        )
    if code == 429:
        return (
            f"Gemini quota exhausted for {model} (the free tier allows ~20 requests/day per "
            "model). Wait for the daily reset, set LLM_MODEL to another model, or enable billing."
        )
    if code == 503:
        return f"{model} is overloaded right now (503). Try again in a few minutes."
    return f"Gemini API error {code}: {getattr(error, 'message', error)}"


def _unit(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else vector


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

    _vertex = False

    def __init__(self, settings: Settings, model: str, *, vertex: bool = False):
        if vertex and not settings.gcp_project:
            raise LLMError("GCP_PROJECT is not set (see .env.example)")
        if not vertex and not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        from google import genai

        self._genai = genai
        # Retry rate limits (429) and "model overloaded" (503) with exponential backoff.
        retry = genai.types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=30.0)
        # Without a timeout one stalled response hangs the caller for good (seen on Vertex).
        http_options = genai.types.HttpOptions(retry_options=retry, timeout=TIMEOUT_MS)
        if vertex:
            self._client = genai.Client(
                vertexai=True,
                project=settings.gcp_project,
                location=settings.gcp_location,
                http_options=http_options,
            )
        else:
            self._client = genai.Client(api_key=settings.gemini_api_key, http_options=http_options)
        self._model = model
        self._vertex = vertex
        self._embedding_model = settings.embedding_model
        self.usage = Usage(model)

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        thinking: Thinking | None = None,
    ) -> T:
        types = self._genai.types
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=thinking) if thinking else None,
        )
        contents = _schema_prompt(prompt, schema)
        last_error: Exception | None = None
        for _ in range(2):
            response = self._generate(contents, config)
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

    def _generate(self, contents: str, config):
        """One request; a timed-out one is tried once more (HttpRetryOptions only covers HTTP
        error statuses)."""
        for attempt in range(2):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
                self.usage.add(getattr(response, "usage_metadata", None))
                return response
            except self._genai.errors.APIError as e:
                raise LLMError(_describe(e, self._model, vertex=self._vertex)) from e
            except httpx.TimeoutException as e:
                if attempt:
                    raise LLMError(
                        f"{self._model} did not answer within {TIMEOUT_MS // 1000}s, twice"
                    ) from e

    def embed(self, texts: list[str], dimensions: int | None = None) -> list[list[float]]:
        """Unit-length vectors, 100 texts per request. `dimensions` truncates (the model is
        trained so shorter prefixes still work) and the result is re-normalised."""
        config = self._genai.types.EmbedContentConfig(output_dimensionality=dimensions)

        def call(contents) -> list[list[float]]:
            try:
                result = self._client.models.embed_content(
                    model=self._embedding_model, contents=contents, config=config
                )
            except self._genai.errors.APIError as e:
                raise LLMError(_describe(e, self._embedding_model, vertex=self._vertex)) from e
            return [_unit(list(e.values)) for e in result.embeddings]

        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            got = call(batch)
            if len(got) != len(batch):
                # Some (multimodal) models embed a list as one combined input; go one by one.
                got = [call(text)[0] for text in batch]
            vectors += got
        return vectors


class FakeLLM:
    """Deterministic stand-in. `responder` maps (prompt, schema) to a model instance."""

    def __init__(self, responder: Callable[[str, type[BaseModel]], BaseModel] | None = None):
        self._responder = responder
        self.prompts: list[str] = []

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        thinking: Thinking | None = None,
    ) -> T:
        self.prompts.append(prompt)
        if self._responder is None:
            raise LLMError("FakeLLM has no responder configured")
        return schema.model_validate(self._responder(prompt, schema))

    def embed(self, texts: list[str], dimensions: int | None = None) -> list[list[float]]:
        # Stable pseudo-embedding: character histogram, enough for plumbing tests.
        out = []
        for t in texts:
            size = dimensions or 16
            v = [0.0] * size
            for ch in t.lower():
                v[ord(ch) % size] += 1.0
            out.append(_unit(v))
        return out


# Settings.provider_for(role) -> how to build that provider's LLM for a model.
PROVIDERS: dict[str, Callable[[Settings, str], LLM]] = {
    "gemini": GeminiLLM,
    "vertex": lambda settings, model: GeminiLLM(settings, model, vertex=True),
    "fake": lambda settings, model: FakeLLM(),
}


def get_llm(settings: Settings, role: Role) -> LLM:
    """The provider configured for `role`, on the model chosen for it."""
    return PROVIDERS[settings.provider_for(role)](settings, settings.model_for(role))
