import json
import os
from urllib import error, request


class LLMConfigurationError(RuntimeError):
    """Configuration ou réponse invalide du fournisseur LLM."""


def generate_answer(context: str, question: str) -> str:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    endpoint = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not api_key or not model:
        raise LLMConfigurationError(
            "Le mode LLM nécessite LLM_API_KEY et LLM_MODEL. Le mode documentaire reste disponible."
        )
    payload = json.dumps({
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ],
    }).encode()
    req = request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode())
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMConfigurationError("Le fournisseur LLM n'a pas pu répondre.") from exc
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMConfigurationError("Réponse LLM inattendue.") from exc
