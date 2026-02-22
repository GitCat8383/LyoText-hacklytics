"""Gemini-powered contextual phrase prediction for assistive communication."""

from __future__ import annotations

import logging
from typing import Any

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a predictive text engine for an assistive communication device used by \
people with motor disabilities. Given the words the user has already typed, \
predict the {n} most likely NEXT SINGLE WORDS the user wants to type.

Rules:
- Return EXACTLY {n} single words (ONE word each, no multi-word phrases).
- Rank them by likelihood given the sentence so far.
- Include a mix of: contextual next-words, common words ("I", "the", "please", \
"yes", "no", "help"), and punctuation-ending words when the sentence seems complete.
- If no history is provided, return common sentence-starting words.
- Return ONLY the words, one per line, numbered 1-{n}. No explanations.
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

        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your-gemini-api-key-here":
            logger.debug("No Gemini API key configured — using fallback phrases")
            return self._fallback_phrases(n)

        try:
            phrases = await self._call_gemini(n)
        except Exception:
            logger.warning("Gemini API call failed, using fallback phrases")
            phrases = self._fallback_phrases(n)

        self._cached_phrases = phrases[:n]
        self._cache_key = cache_key
        return self._cached_phrases

    async def _call_gemini(self, n: int) -> list[str]:
        model = self._get_model()

        history_text = (
            "Sentence so far: " + " ".join(self._history[-20:])
            if self._history
            else "No words typed yet (start of new sentence)."
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
        if not self._history:
            defaults = ["I", "Yes", "No", "Help", "Please", "Thank", "Hi", "The"]
        else:
            last = self._history[-1].lower()
            word_map: dict[str, list[str]] = {
                "i": ["need", "want", "am", "feel", "can", "have", "think", "like"],
                "need": ["help", "water", "food", "rest", "medicine", "you", "to", "a"],
                "want": ["to", "water", "food", "help", "more", "sleep", "that", "this"],
                "am": ["okay", "fine", "tired", "hungry", "cold", "hot", "in", "not"],
                "feel": ["good", "bad", "tired", "sick", "cold", "hot", "pain", "better"],
                "thank": ["you", "God", "goodness", "everyone", "her", "him", "them", "so"],
                "yes": ["please", "I", "that", "thank", "now", "definitely", "sure", "okay"],
                "no": ["thanks", "I", "not", "more", "problem", "way", "need", "please"],
                "help": ["me", "please", "now", "with", "her", "him", "them", "us"],
                "please": ["help", "call", "give", "come", "stop", "wait", "bring", "let"],
            }
            defaults = word_map.get(last, ["the", "I", "a", "is", "and", "to", "it", "my"])
        return defaults[:n]

    async def check_sentence(self) -> dict[str, Any]:
        """Check if the current sentence is meaningful and complete."""
        if not self._history:
            return {"complete": False, "meaningful": False, "suggestion": ""}

        text = " ".join(self._history)

        if len(self._history) < 2:
            return {"complete": False, "meaningful": False, "suggestion": ""}

        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your-gemini-api-key-here":
            return self._fallback_check(text)

        try:
            return await self._gemini_check(text)
        except Exception:
            logger.warning("Gemini sentence check failed, using fallback")
            return self._fallback_check(text)

    async def _gemini_check(self, text: str) -> dict[str, Any]:
        model = self._get_model()
        prompt = (
            "You are evaluating a sentence built word-by-word on an assistive "
            "communication device. The sentence so far is:\n\n"
            f'"{text}"\n\n'
            "Answer with EXACTLY one line in this format:\n"
            "COMPLETE=yes/no MEANINGFUL=yes/no SUGGESTION=<optional short fix>\n\n"
            "- COMPLETE=yes if it forms a grammatically complete sentence.\n"
            "- MEANINGFUL=yes if the intent is clear (even with minor grammar issues).\n"
            "- SUGGESTION: only if MEANINGFUL=yes but COMPLETE=no, suggest 1-2 words "
            "to finish it. Otherwise leave empty."
        )
        response = await model.generate_content_async(prompt)
        line = response.text.strip().split("\n")[0]

        complete = "COMPLETE=yes" in line.upper()
        meaningful = "MEANINGFUL=yes" in line.upper()
        suggestion = ""
        if "SUGGESTION=" in line.upper():
            suggestion = line.upper().split("SUGGESTION=", 1)[1].strip()
            # Restore original case from the raw line
            raw_suggestion = line.split("SUGGESTION=", 1)
            if len(raw_suggestion) > 1:
                suggestion = raw_suggestion[1].strip()

        return {"complete": complete, "meaningful": meaningful, "suggestion": suggestion}

    def _fallback_check(self, text: str) -> dict[str, Any]:
        words = text.split()
        has_subject = any(w.lower() in ("i", "you", "he", "she", "we", "they", "it") for w in words)
        has_verb = any(w.lower() in (
            "am", "is", "are", "was", "were", "need", "want", "have", "feel",
            "help", "call", "give", "come", "go", "like", "know", "think",
            "can", "will", "do", "did", "see", "get", "make", "take",
        ) for w in words)
        meaningful = has_subject and has_verb
        complete = meaningful and len(words) >= 3
        return {"complete": complete, "meaningful": meaningful, "suggestion": ""}

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
