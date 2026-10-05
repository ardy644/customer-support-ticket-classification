"""Data leakage detection tests."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
from src.config import TRAIN_PATH, TEST_PATH, TEXT_COL, LABEL_COL
from src.data_loader import load_data
from src.feature_engineering import build_tfidf_vectorizer, fit_transform_tfidf
from src.preprocessing import preprocess_series


def test_no_text_overlap():
    """Verify zero overlapping texts between train and test."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    overlap = set(train_df[TEXT_COL]) & set(test_df[TEXT_COL])
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping texts"


def test_load_data_leakage_check():
    """Verify load_data() performs leakage check."""
    # This should NOT raise
    X_train, y_train, X_test, y_test = load_data()
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_tfidf_fit_only_on_train():
    """Verify TF-IDF vectorizer is fit only on training data."""
    train_texts = ["alpha bravo charlie"]
    test_texts = ["delta echo foxtrot"]  # Totally different vocabulary
    
    vec = build_tfidf_vectorizer(max_features=100, min_df=1)
    X_train, X_test = fit_transform_tfidf(vec, train_texts, test_texts)
    
    # Test-only words should not be in vocabulary
    vocab = set(vec.vocabulary_.keys())
    test_words = {'delta', 'echo', 'foxtrot'}
    assert vocab.isdisjoint(test_words), f"Leakage: test words {vocab & test_words} in vocabulary"
    
    # Test features should be all zeros (no matching words)
    assert X_test.sum() == 0.0


def test_categories_match():
    """Verify train and test have the same 77 categories."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_cats = set(train_df[LABEL_COL].unique())
    test_cats = set(test_df[LABEL_COL].unique())
    assert train_cats == test_cats, f"Category mismatch: {train_cats.symmetric_difference(test_cats)}"
    assert len(train_cats) == 77
