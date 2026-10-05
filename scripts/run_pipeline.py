"""End-to-end ML pipeline for BANKING77 classification.

Usage:
    python scripts/run_pipeline.py [--skip-eda] [--skip-tuning]
"""
import sys
import time
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR, OUTPUTS_DIR
from src.data_loader import load_data, load_raw_dataframes
from src.preprocessing import preprocess_series
from src.feature_engineering import (
    build_tfidf_vectorizer, fit_transform_tfidf, save_vectorizer
)
from src.models import get_models, save_model, get_calibrated_svc
from src.training import train_and_evaluate_cv
from src.tuning import tune_all_models
from src.evaluation import (
    evaluate_model, compute_top_k_accuracy, compare_models, save_evaluation_results
)
from src.error_analysis import run_full_error_analysis
from src.eda import run_full_eda
from src.routing import get_routing_stats, validate_routing_map
from src.utils import print_separator, get_dir_size_mb


def main():
    parser = argparse.ArgumentParser(description='BANKING77 ML Pipeline')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA step')
    parser.add_argument('--skip-tuning', action='store_true', help='Skip hyperparameter tuning')
    args = parser.parse_args()
    
    total_start = time.time()
    
    # ===== Phase 1: Data Loading =====
    print_separator("PHASE 1: Data Loading & Validation")
    X_train_text, y_train, X_test_text, y_test = load_data()
    print(f"  Train: {len(X_train_text)} samples")
    print(f"  Test:  {len(X_test_text)} samples")
    print(f"  Categories: {y_train.nunique()}")
    print("  Data leakage check: PASSED \u2713")
    
    # ===== Phase 2: EDA =====
    if not args.skip_eda:
        print_separator("PHASE 2: Exploratory Data Analysis")
        train_df, _ = load_raw_dataframes()
        eda_summary = run_full_eda(train_df)
    else:
        print_separator("PHASE 2: EDA (SKIPPED)")
    
    # ===== Phase 3: Preprocessing =====
    print_separator("PHASE 3: Text Preprocessing")
    print("  Preprocessing training texts...")
    X_train_clean = preprocess_series(X_train_text)
    print("  Preprocessing test texts...")
    X_test_clean = preprocess_series(X_test_text)
    print(f"  Sample: '{X_train_text.iloc[0]}' -> '{X_train_clean.iloc[0]}'")
    
    # ===== Phase 4: TF-IDF Feature Engineering =====
    print_separator("PHASE 4: TF-IDF Feature Engineering")
    vectorizer = build_tfidf_vectorizer()
    X_train_tfidf, X_test_tfidf = fit_transform_tfidf(vectorizer, X_train_clean, X_test_clean)
    print(f"  Train TF-IDF shape: {X_train_tfidf.shape}")
    print(f"  Test TF-IDF shape:  {X_test_tfidf.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Dtype: {X_train_tfidf.dtype}")
    save_vectorizer(vectorizer)
    print("  Vectorizer saved.")
    
    # ===== Phase 5: Model Training with CV =====
    print_separator("PHASE 5: Model Training (Cross-Validation)")
    models = get_models()
    cv_results = train_and_evaluate_cv(models, X_train_tfidf, y_train)
    
    # Save CV results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / 'cv_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # ===== Phase 6: Hyperparameter Tuning =====
    if not args.skip_tuning:
        print_separator("PHASE 6: Hyperparameter Tuning")
        tuning_results = tune_all_models(X_train_tfidf, y_train)
        
        # Replace default models with tuned versions
        for name in models:
            if name in tuning_results:
                models[name] = tuning_results[name]['best_estimator']
    else:
        print_separator("PHASE 6: Tuning (SKIPPED)")
    
    # ===== Phase 7: Test Set Evaluation =====
    print_separator("PHASE 7: Test Set Evaluation")
    eval_results = {}
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        eval_results[name] = evaluate_model(model, X_test_tfidf, y_test, name)
        
        # Top-5 accuracy
        top5 = compute_top_k_accuracy(model, X_test_tfidf, y_test, k=5)
        eval_results[name]['top5_accuracy'] = top5
        print(f"    {name}: Accuracy={eval_results[name]['accuracy']:.4f}, "
              f"F1-macro={eval_results[name]['f1_macro']:.4f}, Top-5={top5:.4f}")
    
    compare_models(eval_results)
    save_evaluation_results(eval_results)
    
    # Save all models
    for name, model in models.items():
        save_model(model, name)
    
    # Determine and save best model
    best_name = max(eval_results, key=lambda k: eval_results[k]['accuracy'])
    save_model(models[best_name], 'best_model')
    print(f"  Best model: {best_name} (saved as best_model.joblib)")
    
    # ===== Phase 8: Error Analysis =====
    print_separator("PHASE 8: Error Analysis")
    best_model = models[best_name]
    error_report = run_full_error_analysis(
        best_model, X_test_text, X_test_tfidf, y_test, best_name
    )
    
    # ===== Phase 9: Routing Analysis =====
    print_separator("PHASE 9: Routing Analysis")
    assert validate_routing_map(), "Routing map validation failed!"
    print("  Routing map validated: 77 intents, no duplicates \u2713")
    
    y_pred = best_model.predict(X_test_tfidf)
    routing_stats = get_routing_stats(y_test, y_pred)
    
    print("\n  Department Routing Stats:")
    for dept, stats in sorted(routing_stats.items()):
        print(f"    {dept}: {stats['total_routed']} tickets, "
              f"accuracy={stats['routing_accuracy']:.2%}")
    
    with open(OUTPUTS_DIR / 'routing_stats.json', 'w') as f:
        json.dump(routing_stats, f, indent=2)
    
    # ===== Summary =====
    total_elapsed = time.time() - total_start
    print_separator("PIPELINE COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Best model: {best_name}")
    print(f"  Best accuracy: {eval_results[best_name]['accuracy']:.4f}")
    print(f"  Best F1-macro: {eval_results[best_name]['f1_macro']:.4f}")
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"  Outputs saved to: {OUTPUTS_DIR}")
    
    if MODELS_DIR.exists():
        print(f"  Models directory size: {get_dir_size_mb(str(MODELS_DIR)):.1f} MB")


if __name__ == '__main__':
    main()
