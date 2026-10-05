"""Model definitions for BANKING77 classification."""
import joblib
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.config import RANDOM_STATE, LR_MAX_ITER, SVM_MAX_ITER, MODELS_DIR, JOBLIB_COMPRESS


def get_models() -> dict:
    """Return dictionary of model name -> unfitted model instance.
    
    Models:
        - MultinomialNB: Naive Bayes baseline, fast and memory-efficient.
        - LogisticRegression: Strong linear model with L2 regularization.
        - LinearSVC: State-of-the-art linear SVM for text classification.
    
    Returns:
        dict: {model_name: model_instance}
    """
    return {
        'MultinomialNB': MultinomialNB(alpha=1.0),
        'LogisticRegression': LogisticRegression(
            max_iter=LR_MAX_ITER,
            solver='saga',
            multi_class='multinomial',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'LinearSVC': LinearSVC(
            max_iter=SVM_MAX_ITER,
            random_state=RANDOM_STATE,
            dual='auto',
        ),
    }


def get_calibrated_svc() -> CalibratedClassifierCV:
    """Return a LinearSVC wrapped with CalibratedClassifierCV for probability estimates."""
    base_svc = LinearSVC(
        max_iter=SVM_MAX_ITER,
        random_state=RANDOM_STATE,
        dual='auto',
    )
    return CalibratedClassifierCV(base_svc, cv=5)


def save_model(model, name: str):
    """Save a trained model to disk."""
    path = MODELS_DIR / f"{name}.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=JOBLIB_COMPRESS)
    return path


def load_model(name: str):
    """Load a trained model from disk."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
