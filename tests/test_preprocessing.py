"""Tests for text preprocessing pipeline."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.preprocessing import preprocess_text, preprocess_series
import pandas as pd


def test_lowercase():
    """Test that output is lowercase."""
    result = preprocess_text("My CARD is NOT Working")
    assert result == result.lower()


def test_punctuation_removed():
    """Test that punctuation is removed."""
    result = preprocess_text("Where is my card? It's been 2 weeks!")
    assert '?' not in result
    assert '!' not in result
    assert "'" not in result


def test_negation_preserved():
    """Test that negation words are preserved."""
    result = preprocess_text("I did not receive my card")
    assert 'not' in result.split()


def test_lemmatization():
    """Test that words are lemmatized."""
    result = preprocess_text("I am waiting for my cards to arrive")
    assert 'card' in result.split()  # cards -> card


def test_empty_string():
    """Test empty string handling."""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""


def test_numbers_removed():
    """Test that numbers are removed."""
    result = preprocess_text("I was charged 50 dollars 3 times")
    assert not any(c.isdigit() for c in result)


def test_series_preprocessing():
    """Test batch preprocessing of pandas Series."""
    series = pd.Series(["My card is lost", "I need a refund"])
    result = preprocess_series(series)
    assert len(result) == 2
    assert isinstance(result, pd.Series)


def test_output_is_string():
    """Test that output is always a string."""
    result = preprocess_text("Hello world")
    assert isinstance(result, str)


def test_none_handling():
    """Test that None input returns empty string."""
    result = preprocess_text(None)
    assert result == ""
