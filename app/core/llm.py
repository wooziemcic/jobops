from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import openai


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


def chat_json_strict(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1800,
) -> Dict[str, Any]:
    """
    OpenAI v0.28.1 JSON-only chat call.
    Returns parsed JSON dict. Raises ValueError if parsing fails.
    """
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables or Streamlit secrets.")

    openai.api_key = api_key

    resp = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = resp["choices"][0]["message"]["content"].strip()

    # Allow model to wrap in ```json ... ``` but we strip it safely
    if content.startswith("```"):
        content = content.strip("`")
        # Remove leading 'json' if present
        if content.lower().startswith("json"):
            content = content[4:].strip()

    try:
        return json.loads(content)
    except Exception as e:
        raise ValueError(f"Model did not return valid JSON. Error: {e}\nRaw content:\n{content}")
