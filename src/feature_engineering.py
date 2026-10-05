"""TF-IDF feature engineering for BANKING77 dataset."""
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse

from src.config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF,
    TFIDF_MAX_DF, TFIDF_SUBLINEAR_TF, TFIDF_DTYPE,
    MODELS_DIR, JOBLIB_COMPRESS
)


def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    max_df: float = TFIDF_MAX_DF,
    sublinear_tf: bool = TFIDF_SUBLINEAR_TF,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with configured defaults.
    
    Args:
        max_features: Maximum number of features (vocabulary size cap).
        ngram_range: Tuple of (min_n, max_n) for n-gram range.
        min_df: Minimum document frequency for a term.
        max_df: Maximum document frequency for a term.
        sublinear_tf: Whether to apply sublinear TF scaling.
    
    Returns:
        Configured TfidfVectorizer instance (unfitted).
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        dtype=TFIDF_DTYPE,
    )


def fit_transform_tfidf(vectorizer: TfidfVectorizer, X_train_text, X_test_text):
    """Fit vectorizer on train data and transform both splits.
    
    CRITICAL: fit_transform() is called ONLY on training data.
    Test data uses transform() only to prevent data leakage.
    
    Args:
        vectorizer: Unfitted TfidfVectorizer.
        X_train_text: Training text Series/list.
        X_test_text: Test text Series/list.
    
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf) as sparse CSR matrices.
    """
    # Fit on TRAIN only, transform train
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    
    # Transform test (NO fitting)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Verify sparse output
    assert issparse(X_train_tfidf), "Train TF-IDF must be sparse"
    assert issparse(X_test_tfidf), "Test TF-IDF must be sparse"
    
    return X_train_tfidf, X_test_tfidf


def save_vectorizer(vectorizer: TfidfVectorizer, filename: str = "tfidf_vectorizer.joblib"):
    """Save fitted vectorizer to disk."""
    path = MODELS_DIR / filename
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path, compress=JOBLIB_COMPRESS)
    return path


def load_vectorizer(filename: str = "tfidf_vectorizer.joblib") -> TfidfVectorizer:
    """Load a fitted vectorizer from disk."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {path}")
    return joblib.load(path)
