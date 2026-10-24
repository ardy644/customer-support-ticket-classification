# Code Review: SupportIQ Ticket Routing System

Overall, the project is extremely well-structured and follows excellent practices for a traditional Machine Learning pipeline. The separation of concerns is clear (data loading, preprocessing, modeling, evaluation, routing, app), and there are strict guards against data leakage which is commendable.

Below is a detailed breakdown of the review across Architecture, Code Quality, Bugs, and ML Techniques.

## 1. Architecture
* **Positives**: The modular design (`src/` vs `app/` vs `scripts/`) is clean and adheres to common standard repository layouts. The strict separation of `train.csv` and `test.csv` in `data_loader.py` prevents leakage effectively.
* **Suggestion**: While joblib is used for model serialization, it might be beneficial to bundle the `TfidfVectorizer` and the best model (e.g., `LogisticRegression`) into a single `sklearn.pipeline.Pipeline`. This would simplify inference code in the routing layer and reduce the risk of out-of-sync preprocessing steps during deployment.
* **Suggestion**: The `app/` directory uses streamlit which is great for prototypes. For a production deployment, serving the ML model via a lightweight API (like FastAPI or Flask) separate from the Streamlit frontend would allow other services to query the routing engine easily.

## 2. Code Quality
* **Positives**: Code is mostly clean, well-commented, and includes type hints. Use of constants via `src/config.py` is an excellent practice.
* **Deprecation Warnings**: There is a `FutureWarning` in `src/eda.py` due to the use of `sns.boxplot(..., palette=..., order=...)` without a `hue`. Seaborn will require `hue` to be passed in upcoming versions. This needs to be updated.
* **Redundancy**: `src/utils.py` contains unused utilities (like the `@timer` decorator, which doesn't seem to be widely used), though they might be intended for future use.

## 3. Bugs & Edge Cases
* **Convergence Warning**: The training script raises multiple `ConvergenceWarning: Liblinear failed to converge, increase the number of iterations` during `LinearSVC` training. This indicates that `SVM_MAX_ITER` (currently 2000) is too low for some folds/hyperparameter combinations. This needs to be increased.
* **Logic Bug in `routing.py` Confidence Score**: In `TicketRouter._get_confidence`, when computing softmax over `decision_function`, the code computes `np.exp(scores - scores.max())`. However, for a single sample prediction, `scores` is a 2D array of shape `(1, num_classes)`. To compute the scalar max properly over the classes, it should extract the 1D array using `[0]` just like `_get_top_predictions` does. Otherwise, the matrix operations could yield unexpected shapes in batch or edge cases.

## 4. ML Techniques
* **Positives**:
  - Using TF-IDF with `sublinear_tf=True` and n-grams is a solid baseline for text classification.
  - Using `StratifiedKFold` ensures robust evaluation.
  - Handling negations by explicitly preserving words like "not", "didn't" is crucial for intent classification and correctly implemented.
* **Suggestions for Improvement**:
  - **Class Imbalance**: The dataset has a 5.34x imbalance ratio. While `LogisticRegression` and `LinearSVC` handle this decently, adding `class_weight='balanced'` might improve the recall of minority classes (like `verify_my_identity`).
  - **Probability Calibration**: `LinearSVC` decision functions are converted to pseudo-probabilities using a simple softmax in `routing.py`. While functional, SVM scores don't inherently map to calibrated probabilities. Using the existing `get_calibrated_svc()` function (via `CalibratedClassifierCV`) would yield true probability estimates, which is vital for the confidence thresholds used in `TicketRouter.get_priority`.
  - **Hyperparameter Grid**: The tuning grid for `LogisticRegression` uses `saga`, which is good for L1/ElasticNet, but since the default penalty is L2, `lbfgs` or `liblinear` might converge faster and yield similar results without warnings.

---

### Actions Taken During This Review:
To address the immediate issues found in this review, the following code changes have been applied to the repository:
1. Fixed the Seaborn deprecation warning in `src/eda.py`.
2. Increased `SVM_MAX_ITER` to 5000 in `src/config.py` and `src/tuning.py` to fix `ConvergenceWarning`.
3. Fixed the shape bug in `_get_confidence` in `src/routing.py` to properly index the decision function scores.
