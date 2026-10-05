"""Data loading and validation for BANKING77 dataset."""
import pandas as pd
from src.config import TRAIN_PATH, TEST_PATH, NUM_CATEGORIES, TEXT_COL, LABEL_COL


def load_data() -> tuple:
    """Load and validate BANKING77 train/test CSVs.
    
    Returns:
        tuple: (X_train_text, y_train, X_test_text, y_test) as pandas Series.
    
    Raises:
        ValueError: If data integrity checks fail.
        FileNotFoundError: If CSV files are missing.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}")
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Validate columns
    expected_cols = {TEXT_COL, LABEL_COL}
    if set(train_df.columns) != expected_cols:
        raise ValueError(f"Train columns {set(train_df.columns)} != {expected_cols}")
    if set(test_df.columns) != expected_cols:
        raise ValueError(f"Test columns {set(test_df.columns)} != {expected_cols}")
    
    # Check for nulls
    if train_df[TEXT_COL].isna().any():
        raise ValueError("Training data contains null text values")
    if test_df[TEXT_COL].isna().any():
        raise ValueError("Test data contains null text values")
    
    # Validate category count
    train_cats = train_df[LABEL_COL].nunique()
    test_cats = test_df[LABEL_COL].nunique()
    if train_cats != NUM_CATEGORIES:
        raise ValueError(f"Expected {NUM_CATEGORIES} train categories, got {train_cats}")
    if test_cats != NUM_CATEGORIES:
        raise ValueError(f"Expected {NUM_CATEGORIES} test categories, got {test_cats}")
    
    # Verify categories match
    train_cat_set = set(train_df[LABEL_COL].unique())
    test_cat_set = set(test_df[LABEL_COL].unique())
    if train_cat_set != test_cat_set:
        diff = train_cat_set.symmetric_difference(test_cat_set)
        raise ValueError(f"Train/test category mismatch: {diff}")
    
    # CRITICAL: Check for data leakage (no overlapping texts)
    overlap = set(train_df[TEXT_COL]) & set(test_df[TEXT_COL])
    if len(overlap) > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping texts between train and test"
        )
    
    return (
        train_df[TEXT_COL],
        train_df[LABEL_COL],
        test_df[TEXT_COL],
        test_df[LABEL_COL],
    )


def load_raw_dataframes() -> tuple:
    """Load raw train and test DataFrames without validation.
    
    Returns:
        tuple: (train_df, test_df) as pandas DataFrames.
    """
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def get_category_list() -> list:
    """Return sorted list of all 77 category names."""
    train_df = pd.read_csv(TRAIN_PATH)
    return sorted(train_df[LABEL_COL].unique().tolist())
