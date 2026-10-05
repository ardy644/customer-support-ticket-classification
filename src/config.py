"""Central configuration for the BANKING77 Classification & Routing System."""
from pathlib import Path
import numpy as np

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"

# --- Dataset ---
NUM_CATEGORIES = 77
TEXT_COL = "text"
LABEL_COL = "category"

# --- Reproducibility ---
RANDOM_STATE = 42
CV_FOLDS = 5

# --- TF-IDF ---
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True
TFIDF_DTYPE = np.float32

# --- Models ---
LR_MAX_ITER = 1000
SVM_MAX_ITER = 2000
JOBLIB_COMPRESS = 3
