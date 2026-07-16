"""Minimal, meaning-preserving text normalization for CS classification."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without changing the customer's wording."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        raise ValueError("text must not be empty")
    return normalized


def canonical_for_duplicate_check(text: str) -> str:
    """Canonical form used only to catch superficial duplicates."""
    normalized = normalize_text(text).lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", normalized)
