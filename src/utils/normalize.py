"""
normalize.py — Text normalization for SwitchNet WER computation.

Extend this with your existing normalization logic from the English baseline.
The key design: language-aware dispatch so EN and ES can have different rules.
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _remove_punctuation(text: str) -> str:
    # Keep letters, digits, spaces, apostrophes (contractions)
    return re.sub(r"[^\w\s']", " ", text)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_filler_words(text: str, fillers: list[str]) -> str:
    pattern = r"\b(" + "|".join(re.escape(w) for w in fillers) + r")\b"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Language-specific normalization
# ---------------------------------------------------------------------------

EN_FILLERS = ["um", "uh", "hmm", "mm", "mhm", "uh-huh"]
ES_FILLERS = ["eh", "este", "pues", "o sea", "bueno"]

# Common number-word mappings (extend as needed)
EN_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def _normalize_english(text: str) -> str:
    text = text.lower()
    # Expand common contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
        "'m": " am",
    }
    for c, expansion in contractions.items():
        text = text.replace(c, expansion)
    text = _remove_punctuation(text)
    text = _remove_filler_words(text, EN_FILLERS)
    text = _collapse_whitespace(text)
    return text


def _normalize_spanish(text: str) -> str:
    text = text.lower()
    # Normalize accented characters to ASCII for robust comparison
    # (optional — comment out if you want accent-sensitive eval)
    # text = unicodedata.normalize("NFD", text)
    # text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = _remove_punctuation(text)
    # Remove inverted punctuation (¿ ¡) not caught by the above
    text = re.sub(r"[¿¡]", "", text)
    text = _remove_filler_words(text, ES_FILLERS)
    text = _collapse_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_text(text: str, language: str = "en") -> str:
    """
    Normalize ASR reference or hypothesis text for WER computation.

    Args:
        text:     Raw transcript string.
        language: "en" | "es" | "bilingual"
                  For "bilingual", applies a union of both rule sets.

    Returns:
        Normalized string.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if language == "en":
        return _normalize_english(text)
    elif language == "es":
        return _normalize_spanish(text)
    elif language == "bilingual":
        # Apply both sets of rules
        text = _normalize_english(text)
        text = _normalize_spanish(text)
        return text
    else:
        # Unknown language: apply basic lowercase + punctuation removal
        return _collapse_whitespace(_remove_punctuation(text.lower()))