"""Hyperparameter tuning with GridSearchCV."""
import time
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.config import CV_FOLDS, RANDOM_STATE, OUTPUTS_DIR


PARAM_GRIDS = {
    'MultinomialNB': {
        'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    },
    'LogisticRegression': {
        'C': [0.1, 0.5, 1.0, 5.0, 10.0],
        'penalty': ['l2'],
        'solver': ['saga'],
        'max_iter': [1000],
    },
    'LinearSVC': {
        'C': [0.01, 0.1, 0.5, 1.0, 5.0],
        'loss': ['hinge', 'squared_hinge'],
        'max_iter': [2000],
    },
}


def _get_base_model(name: str):
    """Get a fresh base model instance by name."""
    if name == 'MultinomialNB':
        return MultinomialNB()
    elif name == 'LogisticRegression':
        return LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, multi_class='multinomial')
    elif name == 'LinearSVC':
        return LinearSVC(random_state=RANDOM_STATE, dual='auto')
    else:
        raise ValueError(f"Unknown model: {name}")


def tune_model(
    name: str,
    X_train,
    y_train,
    cv: int = CV_FOLDS,
) -> dict:
    """Run GridSearchCV for a single model.
    
    Args:
        name: Model name (key in PARAM_GRIDS).
        X_train: Training feature matrix.
        y_train: Training labels.
        cv: Number of CV folds.
    
    Returns:
        dict with best_params, best_score, best_estimator, and timing.
    """
    print(f"  Tuning {name}...")
    start_time = time.time()
    
    base_model = _get_base_model(name)
    param_grid = PARAM_GRIDS[name]
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    result = {
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'best_estimator': grid_search.best_estimator_,
        'tune_time_sec': round(elapsed, 2),
    }
    
    print(f"    {name}: Best CV Accuracy = {grid_search.best_score_:.4f}")
    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Tuning time: {elapsed:.1f}s")
    
    return result


def tune_all_models(X_train, y_train, cv: int = CV_FOLDS) -> dict:
    """Tune all models and return results.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        cv: Number of CV folds.
    
    Returns:
        dict: {model_name: {best_params, best_score, best_estimator, tune_time_sec}}
    """
    results = {}
    for name in PARAM_GRIDS:
        results[name] = tune_model(name, X_train, y_train, cv)
    
    # Save tuning results (excluding estimator objects) to JSON
    save_results = {}
    for name, res in results.items():
        save_results[name] = {
            'best_params': res['best_params'],
            'best_score': res['best_score'],
            'tune_time_sec': res['tune_time_sec'],
        }
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / 'tuning_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    return results
