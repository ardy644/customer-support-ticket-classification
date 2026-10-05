"""Tests for model definitions."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from src.models import get_models


@pytest.fixture
def dummy_data():
    """Create small dummy dataset for testing."""
    np.random.seed(42)
    X = csr_matrix(np.random.rand(100, 50).astype(np.float32))
    y = np.array([f"cat_{i % 5}" for i in range(100)])
    return X, y


def test_get_models_returns_dict():
    """Test that get_models returns a dictionary."""
    models = get_models()
    assert isinstance(models, dict)
    assert len(models) == 3


def test_model_names():
    """Test expected model names."""
    models = get_models()
    assert 'MultinomialNB' in models
    assert 'LogisticRegression' in models
    assert 'LinearSVC' in models


def test_models_accept_sparse(dummy_data):
    """Test that all models work with sparse input."""
    X, y = dummy_data
    models = get_models()
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y), f"{name} prediction count mismatch"


def test_models_predict_valid_labels(dummy_data):
    """Test that predictions are valid labels."""
    X, y = dummy_data
    unique_labels = set(y)
    models = get_models()
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        assert all(p in unique_labels for p in preds), f"{name} produced invalid labels"
