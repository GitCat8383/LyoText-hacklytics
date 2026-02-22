"""Gemini-powered single-word recommendation engine for assistive communication."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)

OTHER_LABEL = "Other"

FALLBACK_WORDS = [
    "yes", "no", "help", "water", "pain", "nurse", "doctor", "medicine",
    "cold", "hot", "hungry", "tired", "bathroom", "family", "thank",
    "please", "stop", "more", "less", "bed", "pillow", "light", "dark",
    "sleep", "eat", "drink", "walk", "sit", "stand", "comfortable",
    "uncomfortable", "scared", "happy", "sad", "hello", "goodbye",
    "call", "emergency", "okay", "wait",
]

WORD_PROMPT = """\
You are a predictive single-word engine for an assistive communication device \
used by patients in a healthcare setting.

Given the current sentence so far and an exclude list, predict the {n} most \
likely NEXT single words the patient wants to say.

RULES (follow exactly):
- Return ONLY a JSON array of exactly {n} strings.
- Each string must be a SINGLE word (no spaces).
- Each word must be lowercase.
- No punctuation, no numbers, no special characters.
- All {n} words must be unique.
- None of the words may appear in the exclude list.
- Words must be relevant to patient care communication.
- If sentence is empty, suggest common conversation starters.

EXCLUDE LIST: {exclude}

SENTENCE SO FAR: {sentence}

Return ONLY the JSON array, nothing else."""

STRICT_RETRY_PROMPT = """\
Your previous response was invalid. You MUST return ONLY a valid JSON array \
of exactly {n} lowercase single words. No spaces, no punctuation, no duplicates, \
no words from the exclude list. Example: ["water","pain","nurse","thank","help"]

EXCLUDE LIST: {exclude}
SENTENCE SO FAR: {sentence}

Return ONLY the JSON array:"""


def _validate_words(raw: Any, n: int, exclude: set[str]) -> list[str] | None:
    """Validate parsed words. Returns list of n valid words or None."""
    if not isinstance(raw, list):
        return None

    valid: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        w = item.strip().lower()
        if not w or " " in w:
            continue
        if not re.match(r"^[a-z]+$", w):
            continue
        if w in exclude or w in seen:
            continue
        seen.add(w)
        valid.append(w)
        if len(valid) == n:
            break

    return valid if len(valid) == n else None


class PhraseEngine:
    """Single-word recommendation engine with session-wide deduplication."""

    def __init__(self) -> None:
        self._model = None
        self._shown_words: set[str] = set()
        self._selected_words: list[str] = []
        self._current_sentence: list[str] = []
        self._current_words: list[str] = []

    @property
    def history(self) -> list[str]:
        return list(self._current_sentence)

    @property
    def sentence(self) -> list[str]:
        return list(self._current_sentence)

    @property
    def sentence_text(self) -> str:
        return " ".join(self._current_sentence)

    def _get_model(self) -> Any:
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(config.GEMINI_MODEL)
        return self._model

    def _build_exclude(self) -> set[str]:
        """Build the full exclude set: shown + selected + current sentence."""
        exclude = set(self._shown_words)
        exclude.update(w.lower() for w in self._current_sentence)
        exclude.update(w.lower() for w in self._current_words)
        exclude.discard("")
        return exclude

    async def generate_words(self, n: int = 5) -> list[str]:
        """Generate n recommended next words using Gemini."""
        exclude = self._build_exclude()
        sentence_text = " ".join(self._current_sentence) if self._current_sentence else "(empty)"

        words = await self._call_gemini_words(n, exclude, sentence_text)
        if words is None:
            words = self._fallback_words(n, exclude)

        self._current_words = words
        self._shown_words.update(words)
        return words

    async def generate_other_words(self, n: int = 5) -> list[str]:
        """Generate n new non-duplicate words (for 'Other' selection)."""
        exclude = self._build_exclude()
        sentence_text = " ".join(self._current_sentence) if self._current_sentence else "(empty)"

        words = await self._call_gemini_words(n, exclude, sentence_text)
        if words is None:
            words = self._fallback_words(n, exclude)

        self._current_words = words
        self._shown_words.update(words)
        return words

    async def generate_phrases(self, n: int = config.NUM_PHRASES) -> list[str]:
        """Compatibility wrapper: returns 5 words + 'Other'."""
        words = await self.generate_words(n=5)
        return words + [OTHER_LABEL]

    async def _call_gemini_words(self, n: int, exclude: set[str], sentence_text: str) -> list[str] | None:
        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your-gemini-api-key-here":
            logger.debug("No Gemini API key — using fallback words")
            return None

        exclude_str = ", ".join(sorted(exclude)[:100]) if exclude else "(none)"

        try:
            model = self._get_model()
            prompt = WORD_PROMPT.format(n=n, exclude=exclude_str, sentence=sentence_text)
            response = await model.generate_content_async(prompt)
            text = response.text.strip()
            words = self._parse_json_words(text, n, exclude)
            if words is not None:
                return words

            logger.warning("Gemini returned invalid words, retrying with strict prompt")
            retry_prompt = STRICT_RETRY_PROMPT.format(n=n, exclude=exclude_str, sentence=sentence_text)
            response2 = await model.generate_content_async(retry_prompt)
            text2 = response2.text.strip()
            words2 = self._parse_json_words(text2, n, exclude)
            if words2 is not None:
                return words2

            logger.warning("Gemini retry also invalid — falling back")
            return None
        except Exception:
            logger.warning("Gemini API call failed", exc_info=True)
            return None

    def _parse_json_words(self, text: str, n: int, exclude: set[str]) -> list[str] | None:
        """Parse JSON array from Gemini response and validate."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*?\]", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        return _validate_words(parsed, n, exclude)

    def _fallback_words(self, n: int, exclude: set[str]) -> list[str]:
        """Pick n words from the fallback list, skipping excluded ones."""
        result: list[str] = []
        for w in FALLBACK_WORDS:
            if w not in exclude:
                result.append(w)
                if len(result) == n:
                    break
        while len(result) < n:
            result.append(f"word{len(result)+1}")
        return result

    def select_word(self, word: str) -> None:
        """Append a word to the current sentence."""
        self._current_sentence.append(word)
        self._selected_words.append(word.lower())
        self._shown_words.add(word.lower())

    def confirm_phrase(self, phrase: str) -> None:
        """Compatibility: alias for select_word."""
        self.select_word(phrase)

    def undo_last_word(self) -> str | None:
        """Remove the last word from the current sentence."""
        if self._current_sentence:
            removed = self._current_sentence.pop()
            return removed
        return None

    def delete_last(self) -> str | None:
        """Compatibility: alias for undo_last_word."""
        return self.undo_last_word()

    def clear_sentence(self) -> None:
        """Clear the current sentence without resetting session memory."""
        self._current_sentence.clear()

    def clear_history(self) -> None:
        """Compatibility: alias for clear_sentence."""
        self.clear_sentence()

    def done_send(self) -> str:
        """Return the full sentence, then clear it. Session memory is preserved."""
        text = self.sentence_text
        self._current_sentence.clear()
        return text

    def reset_session(self) -> None:
        """Full session reset: clear all memory."""
        self._shown_words.clear()
        self._selected_words.clear()
        self._current_sentence.clear()
        self._current_words.clear()

    def get_current_words(self) -> list[str]:
        """Return the current 5 words (without 'Other')."""
        return list(self._current_words)


phrase_engine = PhraseEngine()
