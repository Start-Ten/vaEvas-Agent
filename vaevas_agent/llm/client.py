"""LLM client — unified interface for Anthropic Messages API and OpenAI Chat Completions / Responses API.

Supports:
  - Anthropic (native):    provider="anthropic"
  - Anthropic-compatible:   provider="anthropic-compatible"  (DashScope, self-hosted, proxies)
  - OpenAI (native):        provider="openai"
  - OpenAI-compatible:      provider="openai-compatible"     (Azure, vLLM, ollama, local)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

from ..config import LLMConfig


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    elapsed_ms: float


class LLMError(Exception):
    """Raised when LLM API returns an error."""


def _resolve_api_key(config: LLMConfig) -> str:
    """Resolve API key from config or environment."""
    if config.api_key_env:
        env_name = config.api_key_env
    else:
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "anthropic-compatible": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openai-compatible": "OPENAI_API_KEY",
        }
        env_name = env_map.get(config.provider, "")

    api_key = os.environ.get(env_name, "")
    if not api_key:
        raise LLMError(
            f"{env_name} environment variable is not set. "
            f"Set it via 'export {env_name}=<your-key>' or configure api_key_env."
        )
    return api_key


# ─── Anthropic (native + compatible) ─────────────────────────

def _call_anthropic(
    config: LLMConfig,
    system: str,
    user: str,
    temperature: float,
) -> LLMResponse:
    try:
        import anthropic
    except ImportError:
        raise LLMError("anthropic package not installed. Run: pip install anthropic")

    api_key = _resolve_api_key(config)
    kwargs = {"api_key": api_key, "timeout": float(config.timeout)}

    # Custom base_url for compatible providers (e.g., DashScope, proxies)
    if config.base_url:
        kwargs["base_url"] = config.base_url

    client = anthropic.Anthropic(**kwargs)

    t0 = time.perf_counter()
    message = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=temperature,
        top_p=config.top_p,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    elapsed = (time.perf_counter() - t0) * 1000

    # Collect text from TextBlock and ThinkingBlock content blocks.
    # Some providers (DeepSeek) return ThinkingBlock for chain-of-thought
    # followed by TextBlock with the actual response.
    text = ""
    thinking = ""
    for block in message.content:
        block_type = getattr(block, "type", "text")
        if block_type == "text" and hasattr(block, "text"):
            text += block.text
        elif block_type == "thinking" and hasattr(block, "thinking"):
            thinking += block.thinking

    # If only thinking blocks were returned (model hit max_tokens before
    # producing the final response), include a truncated thinking preview.
    if not text and thinking:
        text = f"[thinking: {thinking[:300]}...]"

    return LLMResponse(
        text=text,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        finish_reason=message.stop_reason or "unknown",
        elapsed_ms=elapsed,
    )


# ─── OpenAI (native + compatible) ────────────────────────────

def _call_openai(
    config: LLMConfig,
    system: str,
    user: str,
    temperature: float,
    use_responses_api: bool = False,
) -> LLMResponse:
    try:
        import openai
    except ImportError:
        raise LLMError("openai package not installed. Run: pip install openai")

    api_key = _resolve_api_key(config)
    kwargs = {"api_key": api_key, "timeout": float(config.timeout)}

    # Custom base_url for compatible providers (e.g., Azure, vLLM, ollama)
    if config.base_url:
        kwargs["base_url"] = config.base_url

    client = openai.OpenAI(**kwargs)

    t0 = time.perf_counter()

    if use_responses_api and not config.base_url:
        # Responses API only works with native OpenAI (no custom base_url)
        response = client.responses.create(
            model=config.model,
            instructions=system,
            input=user,
            temperature=temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        text = response.output_text or ""
        usage = response.usage or openai.types.responses.ResponseUsage(
            input_tokens=0, output_tokens=0, total_tokens=0
        )
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        finish_reason = getattr(response, "status", "unknown")
    else:
        # Chat Completions API (works with native + compatible providers)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        finish_reason = response.choices[0].finish_reason or "unknown"

    elapsed = (time.perf_counter() - t0) * 1000

    return LLMResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        elapsed_ms=elapsed,
    )


# ─── Public API ──────────────────────────────────────────────

def call_llm(
    config: LLMConfig,
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_responses_api: bool = False,
) -> LLMResponse:
    """Call the LLM with system + user messages. Returns LLMResponse.

    Supports four provider modes:
      - anthropic:            native Anthropic Messages API
      - anthropic-compatible: Anthropic Messages API at custom base_url
      - openai:               native OpenAI Chat Completions API
      - openai-compatible:    OpenAI Chat Completions at custom base_url

    Args:
        config: LLMConfig with provider, model, base_url, etc.
        system: System prompt (instructions).
        user: User prompt.
        temperature: Override config temperature.
        max_tokens: Override config max_tokens.
        use_responses_api: For native OpenAI only, use Responses API.
    """
    temp = temperature if temperature is not None else config.temperature
    tokens = max_tokens if max_tokens is not None else config.max_tokens

    provider = config.provider

    if provider in ("anthropic", "anthropic-compatible"):
        # Temporarily override model/temperature for the call
        saved_model = config.model
        if config.model != config.model:  # false — just preserving pattern
            pass
        return _call_anthropic(config, system, user, temp)
    elif provider in ("openai", "openai-compatible"):
        return _call_openai(config, system, user, temp, use_responses_api)
    else:
        raise LLMError(f"Unknown provider: {provider}. "
                       f"Expected: anthropic, anthropic-compatible, openai, openai-compatible")
