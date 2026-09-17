"""providers.py -- provider-agnostic LLM chat, zero dependencies (stdlib urllib).

Backends via LLM_PROVIDER env: mock (default) | openai | anthropic | ollama.
Env: LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY (falls back to OPENAI/ANTHROPIC key).
Returns assistant text; raises ProviderError on failure.
"""

import json
import os
import urllib.error
import urllib.request

DEFAULTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama": "http://localhost:11434/v1",
}


class ProviderError(Exception):
    """Any failure talking to a model backend."""


def _post(url, headers, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _mock_reply(messages):
    """Offline stub. Returns a valid per-turn classification JSON so the real
    parsing path runs. Deterministic; clearly not a real model."""
    return ('{"topic": "general", "risk": "SAFE", "shift": "NONE", '
            '"escalating": false, "technique": "NONE"}')


def call_chat(messages, model="", max_tokens=200, temperature=0.1, timeout=60):
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "mock":
        return _mock_reply(messages)

    base = os.getenv("LLM_BASE_URL", DEFAULTS.get(provider, DEFAULTS["openai"])).rstrip("/")
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "x"
    try:
        if provider == "anthropic":
            sys = " ".join(m["content"] for m in messages if m["role"] == "system")
            chat = [m for m in messages if m["role"] != "system"]
            body = {"model": model or "claude-sonnet-4-6", "max_tokens": max_tokens,
                    "temperature": temperature, "messages": chat}
            if sys:
                body["system"] = sys
            resp = _post(base + "/messages", {"x-api-key": key,
                         "anthropic-version": "2023-06-01", "content-type": "application/json"},
                         body, timeout)
            return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text").strip()
        body = {"model": model or "gpt-4o-mini", "max_tokens": max_tokens,
                "temperature": temperature, "messages": messages}
        resp = _post(base + "/chat/completions",
                     {"authorization": f"Bearer {key}", "content-type": "application/json"},
                     body, timeout)
        return (resp["choices"][0]["message"].get("content") or "").strip()
    except urllib.error.HTTPError as e:
        raise ProviderError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")
    except urllib.error.URLError as e:
        raise ProviderError(f"connection error: {e.reason}")
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise ProviderError(f"unexpected response shape: {type(e).__name__}: {e}")
