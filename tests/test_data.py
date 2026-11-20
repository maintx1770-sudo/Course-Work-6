import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.data import ID2LABEL, LABEL2ID, LABEL_NAMES, clean_text


def test_label_mapping_is_consistent():
    assert LABEL_NAMES == ["World", "Sports", "Business", "Sci/Tech"]
    assert all(ID2LABEL[LABEL2ID[name]] == name for name in LABEL_NAMES)


def test_clean_text_decodes_html_entities():
    assert clean_text("It #39;s a test &amp; more") == "It 's a test & more"


def test_clean_text_normalizes_whitespace_and_backslash():
    assert clean_text("  a\\b   c\n d ") == "a b c d"


def test_clean_text_handles_non_string():
    assert clean_text(123) == "123"
