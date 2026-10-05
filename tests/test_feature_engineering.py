"""Tests for TF-IDF feature engineering."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from scipy.sparse import issparse
from src.feature_engineering import build_tfidf_vectorizer, fit_transform_tfidf


@pytest.fixture
def sample_data():
    train_texts = [
        "my card is not working",
        "i want to transfer money",
        "where is my refund",
        "how do i change my pin",
        "my payment was declined",
    ]
    test_texts = [
        "card is broken",
        "send money abroad",
    ]
    return train_texts, test_texts


def test_tfidf_sparse_output(sample_data):
    """Test that TF-IDF produces sparse matrices."""
    train, test = sample_data
    vec = build_tfidf_vectorizer(max_features=100)
    X_train, X_test = fit_transform_tfidf(vec, train, test)
    assert issparse(X_train)
    assert issparse(X_test)


def test_tfidf_dtype(sample_data):
    """Test that TF-IDF uses float32 for memory efficiency."""
    train, test = sample_data
    vec = build_tfidf_vectorizer(max_features=100)
    X_train, X_test = fit_transform_tfidf(vec, train, test)
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32


def test_tfidf_shapes(sample_data):
    """Test TF-IDF output shapes."""
    train, test = sample_data
    vec = build_tfidf_vectorizer(max_features=100)
    X_train, X_test = fit_transform_tfidf(vec, train, test)
    assert X_train.shape[0] == len(train)
    assert X_test.shape[0] == len(test)
    assert X_train.shape[1] == X_test.shape[1]  # Same feature space


def test_tfidf_max_features(sample_data):
    """Test that max_features cap is respected."""
    train, test = sample_data
    max_f = 10
    vec = build_tfidf_vectorizer(max_features=max_f)
    X_train, _ = fit_transform_tfidf(vec, train, test)
    assert X_train.shape[1] <= max_f


def test_fit_only_on_train(sample_data):
    """Test that vectorizer vocab is learned only from training data."""
    train = ["alpha beta gamma"]
    test = ["delta epsilon zeta"]  # Completely different words
    vec = build_tfidf_vectorizer(max_features=100, min_df=1)
    X_train, X_test = fit_transform_tfidf(vec, train, test)
    # Test words not in vocabulary -> all zeros
    assert X_test.sum() == 0.0  # No matching features
