"""Shared LLM provider helpers for synthesis + evaluation scripts.

Mirrors the minimal subset of `pilot/run.py` that other modules need:
`load_env()` to read `.env` at project root, and `make_caller(provider)` to
get a `(call_fn, default_model)` pair for the four supported providers
(openai, openrouter, vllm, anthropic).

Two model classes go through the same OpenAI-compatible path with opposite
reasoning settings:
  - Non-reasoning models (gpt-4.1-mini, qwen3.5, etc.): explicitly pass
    `reasoning.enabled=False`. The OpenRouter gateway (`api.ssunlp.co.kr`)
    defaults to reasoning-mode for the qwen3.5 family, and without this flag
    the internal thinking consumes the entire `max_tokens` budget and returns
    empty content with no error.
  - Reasoning models (gpt-5*, o-series): the gateway rejects
    `reasoning.enabled=False` with a 400 ("Reasoning is mandatory for this
    endpoint and cannot be disabled."), doesn't accept `temperature != 1`,
    and uses `max_completion_tokens` rather than `max_tokens`. The caller is
    routed automatically based on `is_reasoning_model(model)`.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Reasoning models: reasoning.enabled cannot be turned off on the gateway, and
# the chat-completions call must use `max_completion_tokens` rather than
# `max_tokens`. Two sub-families differ on whether they accept `temperature`:
# - "strict" (gpt-5, o-series): gateway-managed; rejects temperature != 1
# - "open-weight" (gpt-oss 20b/120b): accepts temperature normally
STRICT_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
OPEN_REASONING_PREFIXES = ("gpt-oss",)
REASONING_MODEL_PREFIXES = STRICT_REASONING_PREFIXES + OPEN_REASONING_PREFIXES


def is_reasoning_model(model: str) -> bool:
    """True for any reasoning-mandatory model (strict or open-weight).

    The OpenRouter routing prefix (e.g. `openai/`) is stripped before matching.
    """
    name = model.lower().split("/")[-1]
    return any(name.startswith(p) for p in REASONING_MODEL_PREFIXES)


def is_strict_reasoning_model(model: str) -> bool:
    """True for reasoning models that ALSO reject `temperature != 1` (gpt-5, o-series)."""
    name = model.lower().split("/")[-1]
    return any(name.startswith(p) for p in STRICT_REASONING_PREFIXES)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def _openai_compatible(base_url: str | None, api_key_env: str, default_model: str):
    from openai import OpenAI
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set in environment / .env")
    client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)

    def call(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if is_reasoning_model(model):
            kwargs = dict(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                extra_body={"reasoning": {"enabled": True}},
            )
            # Open-weight reasoning models (gpt-oss) accept temperature; strict
            # ones (gpt-5, o-series) reject anything != 1 on the gateway.
            if not is_strict_reasoning_model(model):
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"reasoning": {"enabled": False}},
            )
        return resp.choices[0].message.content or ""

    return call, default_model


def _anthropic():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in environment / .env")
    client = anthropic.Anthropic(api_key=api_key)

    def call(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    return call, "claude-opus-4-7"


def make_caller(provider: str):
    if provider == "openai":
        return _openai_compatible(None, "OPENAI_API_KEY", "gpt-4o-mini")
    if provider == "openrouter":
        return _openai_compatible(
            os.environ.get("OPENROUTER_BASE_URL"),
            "OPENROUTER_API_KEY",
            os.environ.get("OPENROUTER_MODEL_NAME", "qwen/qwen3.5-9b"),
        )
    if provider == "vllm":
        return _openai_compatible(
            os.environ.get("VLLM_BASE_URL"),
            "VLLM_API_KEY",
            os.environ.get("VLLM_MODEL_NAME", "qwen/qwen3.5-9b"),
        )
    if provider == "anthropic":
        return _anthropic()
    raise ValueError(f"unknown provider: {provider}")
