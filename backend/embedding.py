from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional, Protocol


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


@dataclass(frozen=True)
class GeminiModelConfig:
    chat_model: str
    embed_model: str
    vision_model: str


def load_gemini_model_config() -> GeminiModelConfig:
    return GeminiModelConfig(
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        # Gemini Developer API embedding models can change; 'gemini-embedding-001' is the stable replacement.
        embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        vision_model=os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash"),
    )


@dataclass(frozen=True)
class MistralModelConfig:
    chat_model: str
    embed_model: str


def load_mistral_model_config() -> MistralModelConfig:
    return MistralModelConfig(
        chat_model=os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
        embed_model=os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
    )


class LLMProvider(Protocol):
    def embed_text(self, text: str) -> list[float]: ...
    def chat_answer(self, *, system_prompt: str, user_prompt: str) -> str: ...
    def vision_to_text(self, *, image_bytes: bytes, mime_type: str) -> str: ...


class OpenAIClient:
    def __init__(self) -> None:
        # Lazy import so the app can run without openai installed
        from openai import OpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)
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


class GeminiClient:
    def __init__(self) -> None:
        # Lazy import so the app can still run without google-genai installed
        from google import genai  # type: ignore

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._models = load_gemini_model_config()
        
    def embed_text(self, text: str) -> list[float]:
        text = text or ""
        resp = self._client.models.embed_content(model=self._models.embed_model, contents=text)
        #google-genai returns embeddings list; take first
        emb = resp.embeddings[0].values
        return list(emb)

    def chat_answer(self, *, system_prompt: str, user_prompt: str) -> str:
        contents = f"{system_prompt}\n\n{user_prompt}"
        resp = self._client.models.generate_content(model=self._models.chat_model, contents=contents)
        return (getattr(resp, "text", None) or "").strip()
    
    def vision_to_text(self, *, image_bytes: bytes, mime_type: str) -> str:
        from google.genai import types

        prompt = (
        "Describe the bike-related issue shown in the image, as a short factual text.\n"
        "If the image does not show a bike or a bike issue, say so."
        )
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        resp = self._client.models.generate_content(model=self._models.vision_model, contents=[prompt, part])
        return (getattr(resp, "text", None) or "").strip()

class MistralClient:
    def __init__(self) -> None:
        # Lazy import so the app can still run without mistralai installed
        # mistralai has had multiple SDK layouts across versions. Support both.
        try:
            from mistralai import Mistral  # type: ignore
        except Exception:  # pragma: no cover
            try:
                from mistralai.client import Mistral  # type: ignore
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "mistralai SDK is not installed or incompatible. "
                    "Try `pip install -r backend/requirements.txt`."
                ) from e

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set.")

        self._client = Mistral(api_key=api_key)
        self._models = load_mistral_model_config()

    def embed_text(self, text: str) -> list[float]:
        text = text or ""
        embeddings = getattr(self._client, "embeddings", None)
        if embeddings is None:
            raise RuntimeError("Mistral client does not support embeddings.")

        create = getattr(embeddings, "create", None)
        if callable(create):
            resp = create(model=self._models.embed_model, inputs=[text])
            return list(resp.data[0].embedding)

        # Older SDKs exposed embeddings as a method
        if callable(embeddings):
            resp = embeddings(model=self._models.embed_model, inputs=[text])
            data = getattr(resp, "data", None) or []
            if data:
                return list(data[0].embedding)
        raise RuntimeError("Unsupported mistralai embeddings API shape.")

    def chat_answer(self, *, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat = getattr(self._client, "chat", None)
        if chat is None:
            raise RuntimeError("Mistral client does not support chat.")

        complete = getattr(chat, "complete", None)
        if callable(complete):
            resp = complete(model=self._models.chat_model, messages=messages)
            return (resp.choices[0].message.content or "").strip()

        # Older SDKs exposed chat as a method
        if callable(chat):
            resp = chat(model=self._models.chat_model, messages=messages)
            choices = getattr(resp, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", None) if msg is not None else None
                return (content or "").strip()
        raise RuntimeError("Unsupported mistralai chat API shape.")

    def vision_to_text(self, *, image_bytes: bytes, mime_type: str) -> str:
        raise NotImplementedError("Vision is not supported for Mistral provider in this app.")


def get_llm_provider() -> LLMProvider:
    """
    Provider selection:
    - If MISTRAL_API_KEY is set, use Mistral.
    - Else if GEMINI_API_KEY is set, use Gemini (Google AI Studio).
    - Else use OpenAI (requires OPENAI_API_KEY).
    """
    if os.getenv("MISTRAL_API_KEY"):
        return MistralClient()
    if os.getenv("GEMINI_API_KEY"):
        return GeminiClient()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIClient()
    raise RuntimeError(
        "No LLM provider configured. Set one of MISTRAL_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY."
    )

