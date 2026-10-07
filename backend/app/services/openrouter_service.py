import os
import requests
import json
import logging

logger = logging.getLogger("openrouter_service")

# Priority list of current free models on OpenRouter
FREE_MODELS = [
    "z-ai/glm-5.2:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "minimax/minimax-m3:free",
    "openrouter/auto"
]

def query_openrouter(system_role: str, prompt: str, max_tokens: int = 1500, timeout: int = 6) -> str | None:
    """
    Attempts to call OpenRouter API using active free models with a short timeout.
    Returns generated content string if successful, or None if all fail.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENROUTER_API_KEY is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "CareerSphereAI"
    }

    # Try each candidate model with short timeout to prevent UI slowness
    for model in FREE_MODELS:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    if content and len(content.strip()) > 50:
                        return content
            else:
                logger.warning(f"OpenRouter model {model} returned status {response.status_code}: {response.text[:200]}")
        except Exception as e:
            logger.warning(f"OpenRouter model {model} attempt failed: {e}")
            continue

    return None
