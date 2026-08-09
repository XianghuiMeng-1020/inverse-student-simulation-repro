"""OpenRouter gateway (plan p25-08 scaffold)."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


def openrouter_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
    response_format: dict[str, str] | None = None,
    timeout_s: float = 120.0,
) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        msg = "OPENROUTER_API_KEY is not set"
        raise RuntimeError(msg)
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/USER/inverse-student-sim",
        "X-Title": "ISS research",
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return str(data["choices"][0]["message"]["content"])


def openrouter_json_object(
    messages: list[dict[str, str]],
    *,
    model: str,
) -> dict[str, Any]:
    txt = openrouter_chat(
        messages,
        model=model,
        response_format={"type": "json_object"},
    )
    return json.loads(txt)
