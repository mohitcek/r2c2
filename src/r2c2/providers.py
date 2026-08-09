"""Provider runners that normalise "cached input tokens" across four APIs.

Four providers, four different names for the same fact. Every runner returns a
CallResult with comparable token counts, whatever the provider called them. SDKs
are imported lazily inside each runner so the core package installs (and the cost
math runs) without any of them.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CallResult:
    provider: str
    model: str
    call: int
    prompt_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    answer: str


@dataclass
class ModelRun:
    provider: str
    model: str
    calls: list[CallResult] = field(default_factory=list)
    error: str | None = None


def field_of(obj, *names):
    """First present attribute/key, including pydantic extras."""
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
        if isinstance(obj, dict) and obj.get(name) is not None:
            return obj[name]
        extra = getattr(obj, "model_extra", None) or {}
        if extra.get(name) is not None:
            return extra[name]
    return None


def run_openai(model, context, question, call, *, base_url=None, api_key=None) -> CallResult:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI()
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]

    def call_api(stream):
        kwargs = {"model": model, "messages": messages}
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return client.chat.completions.create(**kwargs)

    try:
        resp = call_api(stream=False)
        usage = resp.usage
        answer = resp.choices[0].message.content or ""
    except Exception as exc:
        # Qwen3.6-Plus rejects non-streaming outright. Usage only rides the final
        # chunk, and only if you asked for it.
        if "streaming_required" not in str(exc):
            raise
        chunks, usage = [], None
        for chunk in call_api(stream=True):
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        answer = "".join(chunks)
        if usage is None:
            raise RuntimeError("streamed response carried no usage block") from exc

    # Together nests cached_tokens under prompt_tokens_details on reasoning models
    # and puts it flat on the rest. Check both, or Llama looks like a zero.
    details = field_of(usage, "prompt_tokens_details")
    cached = field_of(details, "cached_tokens") or field_of(usage, "cached_tokens") or 0
    written = (
        field_of(details, "cache_write_tokens")
        or field_of(usage, "cache_write_tokens")
        or 0
    )
    return CallResult(
        provider="openai" if base_url is None else "together",
        model=model,
        call=call,
        prompt_tokens=usage.prompt_tokens,
        cached_tokens=cached,
        cache_write_tokens=written,
        output_tokens=usage.completion_tokens,
        answer=answer.strip().replace("\n", " "),
    )


def run_anthropic(model, context, question, call) -> CallResult:
    import anthropic

    client = anthropic.Anthropic()
    # Only provider needing an explicit breakpoint. It goes on the last block of the
    # stable prefix; the varying question sits after it.
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        output_config={"effort": "low"},
        system=[
            {
                "type": "text",
                "text": context,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    )
    usage = resp.usage
    text = " ".join(b.text for b in resp.content if b.type == "text")
    return CallResult(
        provider="anthropic",
        model=model,
        call=call,
        # input_tokens is the uncached remainder only — the other three providers
        # report the whole prompt, so sum all three to compare like with like.
        prompt_tokens=(
            usage.input_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        ),
        cached_tokens=usage.cache_read_input_tokens,
        cache_write_tokens=usage.cache_creation_input_tokens,
        output_tokens=usage.output_tokens,
        answer=text.strip().replace("\n", " "),
    )


def run_google(model, context, question, call) -> CallResult:
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    # Implicit caching keys off a shared prefix, so bulk context goes first.
    resp = client.models.generate_content(model=model, contents=[context, question])
    usage = resp.usage_metadata
    return CallResult(
        provider="google",
        model=model,
        call=call,
        prompt_tokens=field_of(usage, "prompt_token_count") or 0,
        cached_tokens=field_of(usage, "cached_content_token_count") or 0,
        cache_write_tokens=0,
        output_tokens=field_of(usage, "candidates_token_count") or 0,
        answer=(resp.text or "").strip().replace("\n", " "),
    )


def run_together(model, context, question, call) -> CallResult:
    return run_openai(
        model,
        context,
        question,
        call,
        base_url="https://api.together.xyz/v1",
        api_key=os.environ.get("TOGETHER_API_KEY"),
    )


Runner = Callable[[str, str, str, int], CallResult]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    env: str
    runner: Runner
    models: tuple[str, ...]


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec("openai", "OPENAI_API_KEY", run_openai, ("gpt-5.6-terra",)),
    "anthropic": ProviderSpec("anthropic", "ANTHROPIC_API_KEY", run_anthropic, ("claude-opus-5",)),
    "google": ProviderSpec("google", "GEMINI_API_KEY", run_google, ("gemini-3.6-flash",)),
    "together": ProviderSpec(
        "together",
        "TOGETHER_API_KEY",
        run_together,
        (
            "deepseek-ai/DeepSeek-V4-Pro",
            "Qwen/Qwen3.6-Plus",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
    ),
}


def spec_for(model: str) -> ProviderSpec:
    for spec in PROVIDERS.values():
        if model in spec.models:
            return spec
    known = ", ".join(m for spec in PROVIDERS.values() for m in spec.models)
    raise KeyError(f"no provider registered for {model!r}; known models: {known}")


def sample(
    model: str,
    context: str,
    question: str,
    n_samples: int = 6,
    *,
    provider: str | None = None,
    on_call: Callable[[CallResult], None] | None = None,
) -> list[CallResult]:
    """Send the same prompt n_samples times and return every CallResult.

    Sequential by design: a cache entry only becomes readable once the first
    response starts, so firing N identical calls concurrently makes them all miss.
    If you parallelise, fire call 1 alone and fan out after it completes.
    """
    runner = PROVIDERS[provider].runner if provider else spec_for(model).runner
    calls = []
    for i in range(1, n_samples + 1):
        result = runner(model, context, question, i)
        calls.append(result)
        if on_call:
            on_call(result)
    return calls
