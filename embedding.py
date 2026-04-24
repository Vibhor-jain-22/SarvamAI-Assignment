from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI


@dataclass(frozen=True)
class ModelConfig:
    chat_model: str
    embed_model: str
    vision_model: str


def load_model_config() -> ModelConfig:
    return ModelConfig(
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
    )


class OpenAIClient:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._models = load_model_config()

    def embed_text(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._models.embed_model, input=text)
        return list(resp.data[0].embedding)

    def chat_answer(self, *, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._models.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    def vision_to_text(self, *, image_bytes: bytes, mime_type: str) -> str:
        """
        Convert an image to a short text description for retrieval.
        The output is intentionally concise and factual to avoid inventing details.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        prompt = (
            "Describe the bike-related issue shown in the image, as a short factual text.\n"
            "If the image does not show a bike or a bike issue, say so."
        )

        # Prefer Responses API when available on the client; otherwise fall back to chat.completions
        responses_create = getattr(getattr(self._client, "responses", None), "create", None)
        if callable(responses_create):
            resp = responses_create(
                model=self._models.vision_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                temperature=0.0,
            )
            text = getattr(resp, "output_text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()

        resp2 = self._client.chat.completions.create(
            model=self._models.vision_model,
            messages=[
                {"role": "system", "content": "You extract concise factual descriptions from images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.0,
        )
        return (resp2.choices[0].message.content or "").strip()

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.models.embedding import Embedding
import o