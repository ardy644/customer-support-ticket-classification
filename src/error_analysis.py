"""Error analysis for BANKING77 classification models."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import OUTPUTS_DIR


def get_most_confused_pairs(y_true, y_pred, top_n: int = 20) -> list:
    """Find category pairs with highest mutual confusion.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        top_n: Number of top confused pairs to return.
    
    Returns:
        List of (true_label, pred_label, count) tuples, sorted by count descending.
    """
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    confused = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i][j] > 0:
                confused.append((labels[i], labels[j], int(cm[i][j])))
    
    confused.sort(key=lambda x: x[2], reverse=True)
    return confused[:top_n]


def get_worst_categories(y_true, y_pred, top_n: int = 10) -> pd.DataFrame:
    """Find categories with lowest F1 scores.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        top_n: Number of worst categories to return.
    
    Returns:
        DataFrame with category, precision, recall, f1-score, support.
    """
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    rows = []
    for cat, metrics in report.items():
        if cat in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        rows.append({
            'category': cat,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1-score'],
            'support': metrics['support'],
        })
    
    df = pd.DataFrame(rows)
    return df.nsmallest(top_n, 'f1_score')


def get_misclassified_examples(
    X_text, y_true, y_pred, n_per_category: int = 5
) -> dict:
    """Get sample misclassified texts for each category.
    
    Args:
        X_text: Original text Series.
        y_true: True labels.
        y_pred: Predicted labels.
        n_per_category: Number of examples per category.
    
    Returns:
        dict: {category: [(text, true_label, pred_label), ...]}
    """
    X_text = np.array(X_text)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    misclassified = {}
    wrong_mask = y_true != y_pred
    
    for cat in sorted(set(y_true)):
        cat_mask = (y_true == cat) & wrong_mask
        indices = np.where(cat_mask)[0]
        
        if len(indices) == 0:
            continue
        
        samples = []
        for idx in indices[:n_per_category]:
            samples.append({
                'text': str(X_text[idx]),
                'true_label': str(y_true[idx]),
                'predicted_label': str(y_pred[idx]),
            })
        misclassified[cat] = samples
    
    return misclassified


def analyze_confidence(model, X_test, y_test) -> dict:
    """Analyze prediction confidence for correct vs incorrect predictions.
    
    Returns:
        dict with mean/median confidence for correct and incorrect predictions.
    """
    y_pred = model.predict(X_test)
    correct_mask = np.array(y_test) == np.array(y_pred)
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        max_conf = proba.max(axis=1)
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test)
        max_conf = scores.max(axis=1)
    else:
        return {'note': 'Model does not support confidence scores'}
    
    return {
        'correct_mean_confidence': float(max_conf[correct_mask].mean()),
        'correct_median_confidence': float(np.median(max_conf[correct_mask])),
        'incorrect_mean_confidence': float(max_conf[~correct_mask].mean()),
        'incorrect_median_confidence': float(np.median(max_conf[~correct_mask])),
        'num_correct': int(correct_mask.sum()),
        'num_incorrect': int((~correct_mask).sum()),
    }


def plot_confusion_matrix_subset(y_true, y_pred, top_n: int = 20):
    """Plot confusion matrix heatmap for the top-N most confused categories."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find categories involved in most confusions
    confused_pairs = get_most_confused_pairs(y_true, y_pred, top_n=100)
    cat_counts = {}
    for true_cat, pred_cat, count in confused_pairs:
        cat_counts[true_cat] = cat_counts.get(true_cat, 0) + count
        cat_counts[pred_cat] = cat_counts.get(pred_cat, 0) + count
    
    top_confused_cats = sorted(cat_counts, key=cat_counts.get, reverse=True)[:top_n]
    
    # Build subset confusion matrix
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    mask = np.isin(y_true_arr, top_confused_cats) & np.isin(y_pred_arr, top_confused_cats)
    
    cm = confusion_matrix(
        y_true_arr[mask], y_pred_arr[mask], labels=top_confused_cats
    )
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=top_confused_cats, yticklabels=top_confused_cats,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Top {top_n} Most Confused Categories)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / 'confusion_matrix_subset.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: confusion_matrix_subset.png")


def plot_error_by_text_length(X_text, y_true, y_pred):
    """Plot accuracy vs text length."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    X_text = np.array(X_text)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    lengths = np.array([len(str(t)) for t in X_text])
    correct = y_true == y_pred
    
    # Bin by text length
    bins = np.percentile(lengths, np.arange(0, 101, 10))
    bins = np.unique(bins)
    bin_indices = np.digitize(lengths, bins)
    
    bin_accuracy = []
    bin_labels = []
    for b in range(1, len(bins)):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_accuracy.append(correct[mask].mean())
            bin_labels.append(f"{int(bins[b-1])}-{int(bins[b])}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(bin_accuracy)), bin_accuracy, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlabel('Text Length Range (characters)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Text Length')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / 'error_by_text_length.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: error_by_text_length.png")


def run_full_error_analysis(model, X_test_text, X_test_tfidf, y_test, model_name: str = "best_model") -> dict:
    """Run comprehensive error analysis."""
    print(f"\nRunning Error Analysis for {model_name}...")
    print("="*50)
    
    y_pred = model.predict(X_test_tfidf)
    
    # Most confused pairs
    confused = get_most_confused_pairs(y_test, y_pred)
    print("\nTop 10 Most Confused Pairs:")
    for true_cat, pred_cat, count in confused[:10]:
        print(f"  {true_cat} -> {pred_cat}: {count}")
    
    # Worst categories
    worst = get_worst_categories(y_test, y_pred)
    print(f"\nBottom 10 Categories by F1:")
    print(worst.to_string(index=False))
    
    # Misclassified examples
    misclassified = get_misclassified_examples(X_test_text, y_test, y_pred)
    
    # Confidence analysis
    confidence = analyze_confidence(model, X_test_tfidf, y_test)
    print(f"\nConfidence Analysis:")
    for k, v in confidence.items():
        print(f"  {k}: {v}")
    
    # Plots
    plot_confusion_matrix_subset(y_test, y_pred)
    plot_error_by_text_length(X_test_text, y_test, y_pred)
    
    # Save full report
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'model_name': model_name,
        'confused_pairs': [(t, p, c) for t, p, c in confused],
        'worst_categories': worst.to_dict(orient='records'),
        'confidence_analysis': confidence,
        'num_misclassified_categories': len(misclassified),
        'total_misclassified': sum(len(v) for v in misclassified.values()),
    }
    with open(OUTPUTS_DIR / 'error_analysis.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("  Saved: error_analysis.json")
    
    print("Error analysis complete!\n")
    return report
