"""Gemini-powered contextual phrase prediction for assistive communication."""

from __future__ import annotations

import logging
from typing import Any

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a predictive text engine for an assistive communication device used by \
people with motor disabilities. Given the conversation history (phrases the user \
has already confirmed), predict the {n} most likely next phrases the user wants \
to say.

Rules:
- Return EXACTLY {n} short phrases (1-6 words each).
- Rank them by likelihood.
- Include a mix of: contextual follow-ups, common needs ("Yes", "No", "Help"), \
and social phrases ("Thank you", "Please").
- If no history is provided, return common starter phrases.
- Return ONLY the phrases, one per line, numbered 1-{n}. No explanations.
"""


class PhraseEngine:

    def __init__(self) -> None:
        self._model = None
        self._history: list[str] = []
        self._cached_phrases: list[str] = []
        self._cache_key: str = ""

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def _get_model(self) -> Any:
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(config.GEMINI_MODEL)
        return self._model

    async def generate_phrases(self, n: int = config.NUM_PHRASES) -> list[str]:
        cache_key = "|".join(self._history[-10:])
        if cache_key == self._cache_key and self._cached_phrases:
            return self._cached_phrases

        try:
            phrases = await self._call_gemini(n)
        except Exception:
            logger.exception("Gemini API call failed, using fallback phrases")
            phrases = self._fallback_phrases(n)

        self._cached_phrases = phrases[:n]
        self._cache_key = cache_key
        return self._cached_phrases

    async def _call_gemini(self, n: int) -> list[str]:
        model = self._get_model()

        history_text = (
            "Conversation so far: " + " -> ".join(self._history[-10:])
            if self._history
            else "No conversation history yet (start of session)."
        )

        prompt = (
            SYSTEM_PROMPT.format(n=n)
            + "\n\n"
            + history_text
            + "\n\nGenerate the next phrases:"
        )

        response = await model.generate_content_async(prompt)
        text = response.text.strip()
        return self._parse_response(text, n)

    def _parse_response(self, text: str, n: int) -> list[str]:
        lines = text.strip().split("\n")
        phrases = []
        for line in lines:
            cleaned = line.strip()
            if not cleaned:
                continue
            for sep in [".", ")", ":"]:
                if len(cleaned) > 2 and cleaned[0].isdigit() and sep in cleaned[:3]:
                    cleaned = cleaned.split(sep, 1)[1].strip()
                    break
            if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
                cleaned = cleaned[1:-1]
            if cleaned:
                phrases.append(cleaned)

        if len(phrases) < n:
            phrases.extend(self._fallback_phrases(n - len(phrases)))
        return phrases[:n]

    def _fallback_phrases(self, n: int) -> list[str]:
        defaults = [
            "Yes",
            "No",
            "Help me please",
            "Thank you",
            "I need water",
            "I'm in pain",
            "Call the nurse",
            "I'm okay",
        ]
        return defaults[:n]

    def confirm_phrase(self, phrase: str) -> None:
        self._history.append(phrase)
        self._cached_phrases = []
        self._cache_key = ""

    def delete_last(self) -> str | None:
        if self._history:
            removed = self._history.pop()
            self._cached_phrases = []
            self._cache_key = ""
            return removed
        return None

    def clear_history(self) -> None:
        self._history.clear()
        self._cached_phrases = []
        self._cache_key = ""


phrase_engine = PhraseEngine()
