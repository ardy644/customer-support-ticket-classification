"""Model training orchestrator with cross-validation."""
import time
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import CV_FOLDS, RANDOM_STATE


def train_and_evaluate_cv(models: dict, X_train, y_train, cv: int = CV_FOLDS) -> dict:
    """Train each model with stratified K-fold CV on training data only.
    
    Args:
        models: Dict of {model_name: unfitted_model}.
        X_train: Training feature matrix (sparse).
        y_train: Training labels.
        cv: Number of cross-validation folds.
    
    Returns:
        dict: {model_name: {mean_accuracy, std, fold_scores, train_time_sec}}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name} with {cv}-fold CV...")
        start_time = time.time()
        
        # Cross-validation on training data ONLY
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring='accuracy', n_jobs=-1
        )
        
        # Refit on full training set
        model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        
        results[name] = {
            'mean_accuracy': float(scores.mean()),
            'std': float(scores.std()),
            'fold_scores': scores.tolist(),
            'train_time_sec': round(elapsed, 2),
        }
        
        print(f"    {name}: CV Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f}) [{elapsed:.1f}s]")
    
    return results
