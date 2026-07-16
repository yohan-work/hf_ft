import pytest

from cs_classifier.preprocess import canonical_for_duplicate_check, normalize_text


def test_normalize_text_collapses_whitespace_and_unicode() -> None:
    assert normalize_text("  결제\u3000실패\n") == "결제 실패"


def test_normalize_text_rejects_blank_input() -> None:
    with pytest.raises(ValueError):
        normalize_text(" \n ")


def test_canonical_duplicate_form_ignores_spacing_and_punctuation() -> None:
    assert canonical_for_duplicate_check("결제는 됐어요!") == canonical_for_duplicate_check("결제는 됐어요")
