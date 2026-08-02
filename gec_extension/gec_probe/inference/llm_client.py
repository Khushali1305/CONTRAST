"""
gec_probe.inference.llm_client
------------------------------
Minimal provider-routing LLM client. All API keys are read from environment
variables; never hardcoded.

Supported providers: openai, anthropic, mistral, groq, local (HF/Ollama).
"""
from __future__ import annotations
import os
import time
import json
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Proactive throttling
# ------------------------------------------------------------------
# Reactive backoff (sleeping after a 429) only helps once you've already
# been rate-limited. On tight free-tier limits (seen in practice with
# Mistral's "la-3" tier returning code 1300 on back-to-back calls), it's
# far more effective to simply never send requests faster than the
# provider's allowed rate to begin with. Tune per provider as needed.
_MIN_INTERVAL_S = {
    "mistral": 4.0,   # free tier: roughly 1 request every few seconds
}
_last_call_at: dict[str, float] = {}


def _throttle(provider: str) -> None:
    min_interval = _MIN_INTERVAL_S.get(provider)
    if not min_interval:
        return
    now = time.monotonic()
    prev = _last_call_at.get(provider)
    if prev is not None:
        elapsed = now - prev
        wait = min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
    _last_call_at[provider] = time.monotonic()

# Provider routing tables (model_id -> provider, api_model_name).
# For local models the api_model_name equals the code-id because the launch
# script starts vLLM with `--served-model-name <code-id>`. For hosted APIs,
# api_model_name is the provider's canonical model string.
PROVIDER_BY_MODEL = {
    # OpenAI
    "gpt_4_1":       ("openai", "gpt-4.1"),
    # Anthropic
    "claude_haiku_4_5": ("anthropic", "claude-haiku-4-5"),
    # Mistral
    "mistral_large_3": ("mistral", "mistral-large-latest"),
    # Groq-hosted Llama
    "llama_3_1_8b":  ("groq", "llama-3.1-8b-instant"),
    "llama_3_1_70b": ("groq", "llama-3.1-70b-versatile"),
    # DeepSeek (native, NOT the Groq-hosted Llama-distill)
    "deepseek_v4_flash": ("deepseek", "deepseek-v4-flash"),
    # Local (vLLM --served-model-name == code-id)
    "qwen_2_5_32b":     ("local", "qwen_2_5_32b"),
    "gemma_3_27b":      ("local", "gemma_3_27b"),
    "flan_t5_large":    ("local", "flan_t5_large"),
    "flan_t5_xl":       ("local", "flan_t5_xl"),
    "llama_3_1_70b_local": ("local", "llama_3_1_70b_local"),
    "gector_baseline":  ("local", "gector_baseline"),
}


def _require_key(env_var: str) -> str:
    k = os.environ.get(env_var)
    if not k:
        raise RuntimeError(f"Missing API key in env var {env_var}")
    return k


def _call_openai(model_name: str, prompt: str, **kw) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=kw.get("temperature", 0.0),
        max_tokens=kw.get("max_tokens", 512),
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(model_name: str, prompt: str, **kw) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model_name,
        max_tokens=kw.get("max_tokens", 512),
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _call_mistral(model_name: str, prompt: str, **kw) -> str:
    # Mistral API is OpenAI-compatible; use openai SDK to avoid mistralai package.
    from openai import OpenAI
    client = OpenAI(
        api_key=_require_key("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1",
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=kw.get("temperature", 0.0),
        max_tokens=kw.get("max_tokens", 512),
    )
    return resp.choices[0].message.content or ""


def _call_groq(model_name: str, prompt: str, **kw) -> str:
    from groq import Groq
    client = Groq(api_key=_require_key("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=kw.get("temperature", 0.0),
        max_tokens=kw.get("max_tokens", 512),
    )
    return resp.choices[0].message.content


def _call_deepseek(model_name: str, prompt: str, **kw) -> str:
    """
    Native DeepSeek API (https://api.deepseek.com/v1). OpenAI-compatible.
    Uses the OpenAI SDK with a redirected base_url.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=_require_key("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=kw.get("temperature", 0.0),
        max_tokens=kw.get("max_tokens", 512),
    )
    return resp.choices[0].message.content or ""


def _call_local(model_name: str, prompt: str, **kw) -> str:
    """
    Local inference via vLLM's OpenAI-compatible server.

    The launch script starts vLLM with `--served-model-name <code-id>` so the
    API model field equals the code-side identifier regardless of the HF repo
    or quantization variant. PROVIDER_BY_MODEL therefore maps each local
    model_id to itself.

    `gector_baseline` is special: not served by vLLM, runs on CPU via
    `_call_gector`.
    """
    if model_name == "gector_baseline":
        return _call_gector(prompt)
    import requests
    base = os.environ.get("LOCAL_INFERENCE_URL", "http://localhost:8000/v1")
    resp = requests.post(
        f"{base}/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kw.get("temperature", 0.0),
            "max_tokens": kw.get("max_tokens", 512),
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gector(prompt: str) -> str:
    """
    GECToR CPU baseline. The corrector prompt has the form:
        "Correct ONLY the grammatical errors ... Sentence: <s> Corrected:"
    We extract the sentence and (TODO) route it to your GECToR setup.
    For now this is a passthrough (no correction).
    """
    try:
        sent = prompt.split("Sentence:", 1)[1].split("Corrected:", 1)[0].strip()
    except IndexError:
        sent = prompt.strip()
    # TODO: replace with subprocess call or direct import of your GECToR setup.
    return sent


_DISPATCH = {
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
    "mistral":   _call_mistral,
    "groq":      _call_groq,
    "deepseek":  _call_deepseek,
    "local":     _call_local,
}


def _is_rate_limited(e: Exception) -> bool:
    """Detect HTTP 429 / rate-limit errors across openai-SDK-based providers."""
    if e.__class__.__name__ in ("RateLimitError",):
        return True
    status = getattr(e, "status_code", None)
    if status is None:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
    if status == 429:
        return True
    return "429" in str(e) or "rate_limit" in str(e).lower() or "rate limit" in str(e).lower()


def _retry_after_seconds(e: Exception) -> Optional[float]:
    """Try to honor a server-provided Retry-After header, if present."""
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def call_llm(model_id: str, prompt: str, *,
             retries: int = 6, backoff: float = 2.0,
             rate_limit_retries: int = 6, rate_limit_base_sleep: float = 20.0,
             rate_limit_max_sleep: float = 120.0,
             api_key_env: Optional[str] = None, **kw) -> str:
    if model_id not in PROVIDER_BY_MODEL:
        raise ValueError(f"Unknown model_id {model_id!r}")
    provider, model_name = PROVIDER_BY_MODEL[model_id]
    fn = _DISPATCH[provider]
    last_err: Optional[Exception] = None
    rl_attempts = 0
    other_attempts = 0
    while True:
        _throttle(provider)
        try:
            return fn(model_name, prompt, **kw)
        except Exception as e:
            last_err = e
            if _is_rate_limited(e):
                rl_attempts += 1
                if rl_attempts > rate_limit_retries:
                    break
                sleep_s = _retry_after_seconds(e)
                if sleep_s is None:
                    # Exponential backoff with a much longer base than generic
                    # errors, since 429s are typically per-minute windows.
                    sleep_s = min(
                        rate_limit_base_sleep * (2 ** (rl_attempts - 1)),
                        rate_limit_max_sleep,
                    )
                log.warning(
                    "call_llm rate-limited (attempt %d/%d) model=%s: %s -- "
                    "sleeping %.1fs",
                    rl_attempts, rate_limit_retries, model_id, e, sleep_s,
                )
                time.sleep(sleep_s)
                continue
            other_attempts += 1
            if other_attempts > retries:
                break
            log.warning("call_llm failed (attempt %d/%d) model=%s: %s",
                        other_attempts, retries, model_id, e)
            time.sleep(backoff ** other_attempts)
    raise RuntimeError(f"call_llm permanently failed for {model_id}: {last_err}")