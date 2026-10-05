"""Model evaluation and metrics for BANKING77 classification."""
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import OUTPUTS_DIR


def evaluate_model(model, X_test, y_test, model_name: str = "model") -> dict:
    """Evaluate a trained model on the test set.
    
    Args:
        model: Fitted classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        model_name: Name for reporting.
    
    Returns:
        dict with accuracy, F1 scores, classification report, confusion matrix.
    """
    y_pred = model.predict(X_test)
    
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'predictions': y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred),
    }
    
    return results


def compute_top_k_accuracy(model, X_test, y_test, k: int = 5) -> float:
    """Compute top-k accuracy: fraction of samples where true label is in top-k predictions.
    
    Args:
        model: Fitted classifier with predict_proba or decision_function.
        X_test: Test feature matrix.
        y_test: True test labels.
        k: Number of top predictions to consider.
    
    Returns:
        Top-k accuracy as a float.
    """
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X_test)
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test)
    else:
        return -1.0  # Cannot compute
    
    classes = model.classes_
    top_k_indices = np.argsort(scores, axis=1)[:, -k:]
    top_k_classes = classes[top_k_indices]
    
    y_test_arr = np.array(y_test)
    correct = sum(
        y_test_arr[i] in top_k_classes[i]
        for i in range(len(y_test_arr))
    )
    
    return float(correct / len(y_test_arr))


def compare_models(eval_results: dict) -> None:
    """Print a comparison table of all model evaluation results."""
    print("\n" + "="*80)
    print("MODEL COMPARISON ON TEST SET")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1-Macro':>10} {'F1-Weighted':>12} {'Prec-M':>8} {'Rec-M':>8}")
    print("-"*80)
    for name, res in eval_results.items():
        print(
            f"{name:<25} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f} "
            f"{res['f1_weighted']:>12.4f} {res['precision_macro']:>8.4f} {res['recall_macro']:>8.4f}"
        )
    print("="*80)


def save_evaluation_results(eval_results: dict, filename: str = "evaluation_results.json"):
    """Save evaluation results to JSON (excluding raw predictions)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    save_data = {}
    for name, res in eval_results.items():
        save_data[name] = {
            k: v for k, v in res.items()
            if k not in ('predictions', 'confusion_matrix', 'classification_report')
        }
    
    with open(OUTPUTS_DIR / filename, 'w') as f:
        json.dump(save_data, f, indent=2)
