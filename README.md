# Intelligent Customer Support Ticket Classification and Routing System

An end-to-end, production-ready machine learning system that automatically classifies banking customer support queries across all **77 fine-grained intents** from the **BANKING77 dataset** and routes them to appropriate specialized support departments.

Built strictly with **traditional Machine Learning and NLP** (CPU-first, zero deep learning, zero external APIs) adhering to anti-leakage protocols and the official BANKING77 train/test split.

---

## Key Highlights

- **Traditional ML Only**: TF-IDF + Multinomial Naive Bayes, Logistic Regression, and Linear SVM. No CNNs, BERT, Transformers, LLMs, or external APIs.
- **CPU-First & Resource Efficient**: Entire training pipeline executes in **< 40 seconds** on standard CPU. Peak model storage is **< 8 MB**.
- **Zero Data Leakage**: Vectorizer is fitted strictly on the official training split (`10,003` samples). Evaluation is computed strictly on the official held-out test split (`3,080` samples).
- **High Accuracy**:
  - **Logistic Regression**: **87.18% Accuracy**, **87.18% Macro-F1**, **98.18% Top-5 Accuracy**
  - **Linear SVM**: **86.82% Accuracy**, **86.76% Macro-F1**, **97.11% Top-5 Accuracy**
  - **Multinomial Naive Bayes**: **83.96% Accuracy**, **83.87% Macro-F1**, **97.01% Top-5 Accuracy**
- **Intelligent Routing**: 77 fine-grained intents mapped to 10 operational departments with priority scoring (`URGENT`, `HIGH`, `MEDIUM`, `NORMAL`). Overall departmental routing accuracy exceeds **94%**.
- **Interactive Streamlit Dashboard**: 5-page application featuring EDA, model comparison, real-time single & batch prediction, error analysis, and routing analytics.
- **Automated Test Suite**: 34 unit, integration, and anti-leakage tests via `pytest`.

---

## System Architecture

```
                                  [ Customer Query ]
                                          │
                                          ▼
                         [ Text Cleaning & Normalization ]
                         (lowercase, strip non-alpha,
                          lemmatization, keep negations)
                                          │
                                          ▼
                            [ TF-IDF Vectorization ]
                             (float32, sublinear TF,
                              ngram (1,2), max 15k)
                                          │
                                          ▼
                         [ Trained Classifiers (Tuned) ]
                         ┌─────────────────────────────┐
                         │ Multinomial Naive Bayes     │
                         │ Logistic Regression (Best)  │
                         │ Linear SVM                  │
                         └──────────────┬──────────────┘
                                        │
                         Predicted Intent + Confidence
                                        │
                                        ▼
                           [ Ticket Routing Engine ]
                         ┌─────────────────────────────┐
                         │  Intent -> Department Map   │
                         │  Dynamic Priority Assigner  │
                         └──────────────┬──────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
         [ Streamlit Dashboard ]             [ Automated Support Queue ]
         - Live Ticket Classifier            - Card Services (96.45%)
         - Batch CSV Processor               - Card Security (94.71%)
         - EDA & Model Comparisons           - Payment Support (90.79%)
         - Error & Confusion Heatmaps        - Transfer Support (95.19%)
                                             - Account Mgmt (99.37%)
```

---

## Dataset & Split Integrity

The dataset is the official **BANKING77** benchmark consisting of online banking customer service queries:

| Split | Sample Count | Categories | Imbalance | Overlap with Test |
|---|---|---|---|---|
| **Train** (`Data/train.csv`) | 10,003 | 77 | 35 min to 187 max (5.34x) | 0 samples (verified) |
| **Test** (`Data/test.csv`) | 3,080 | 77 | 40 per category (balanced) | 0 samples (verified) |

### Anti-Leakage Safeguards
1. **Fit Isolation**: TF-IDF `fit_transform` is applied exclusively to training data; test features use `.transform()`.
2. **Vocabulary Containment**: Validated via unit test `test_tfidf_fit_only_on_train` that test-exclusive terms are never indexed.
3. **Cross-Validation Integrity**: Stratified 5-fold cross-validation is performed strictly within the training set.

---

## Actual Model Evaluation Results

Evaluated on the full 3,080 test samples (40 samples per category):

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Top-5 Accuracy | Train+CV Time |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** (Tuned) | **87.18%** | **87.18%** | **87.18%** | **87.76%** | **87.18%** | **98.18%** | ~17.7s |
| **Linear SVM** (Tuned) | **86.82%** | **86.76%** | **86.76%** | **87.36%** | **86.82%** | **97.11%** | ~8.5s |
| **Multinomial Naive Bayes** (Tuned) | **83.96%** | **83.87%** | **83.87%** | **84.88%** | **83.96%** | **97.01%** | ~3.8s |

### Best Hyperparameters Found via 5-Fold Stratified GridSearchCV
- **Logistic Regression**: `C=5.0`, `solver='saga'`, `max_iter=1000`
- **Linear SVM**: `C=0.5`, `loss='squared_hinge'`, `max_iter=2000`
- **Multinomial Naive Bayes**: `alpha=0.05`

---

## Departmental Routing Matrix

The 77 banking intents are routed across 10 operational departments:

| Department | Handled Intents | Test Volume | Routing Accuracy | Typical Intents Handled |
|---|---|---|---|---|
| **Account Management** | 4 | 158 | **99.37%** | `terminate_account`, `edit_personal_details`, `passcode_forgotten` |
| **Top-Up Support** | 10 | 384 | **97.66%** | `top_up_failed`, `automatic_top_up`, `pending_top_up` |
| **Card Services** | 19 | 760 | **96.45%** | `card_arrival`, `card_linking`, `activate_my_card`, `order_physical_card` |
| **Transfer Support** | 11 | 437 | **95.19%** | `transfer_timing`, `pending_transfer`, `cancel_transfer`, `declined_transfer` |
| **Identity & Verification** | 4 | 162 | **95.06%** | `verify_my_identity`, `unable_to_verify_identity`, `why_verify_identity` |
| **Foreign Exchange** | 6 | 241 | **95.02%** | `exchange_rate`, `exchange_charge`, `fiat_currency_support` |
| **Card Security** | 5 | 189 | **94.71%** | `lost_or_stolen_card`, `compromised_card`, `pin_blocked` |
| **Refunds** | 2 | 83 | **92.77%** | `request_refund`, `Refund_not_showing_up` |
| **Cash & ATM Services** | 7 | 286 | **91.61%** | `atm_support`, `declined_cash_withdrawal`, `wrong_amount_of_cash_received` |
| **Payment Support** | 9 | 380 | **90.79%** | `declined_card_payment`, `transaction_charged_twice`, `card_payment_fee_charged` |

### Priority Assignment Matrix
- **URGENT**: Security-critical incidents (e.g. `compromised_card`, `lost_or_stolen_phone`, `lost_or_stolen_card`) with high confidence.
- **HIGH**: Transaction failures (`declined_card_payment`, `failed_transfer`, `transaction_charged_twice`).
- **MEDIUM**: Operational blockers or predictions with confidence `< 0.30` requiring human review.
- **NORMAL**: Standard informational and self-service inquiries.

---

## Project Structure

```
DS lab project/
├── .gitignore                      # Strict exclusion rules for artifacts/caches
├── requirements.txt                # Lean dependencies (no DL frameworks)
├── README.md                       # Project documentation
├── Data/
│   ├── train.csv                   # Official BANKING77 train set (10,003 rows)
│   └── test.csv                    # Official BANKING77 test set (3,080 rows)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Paths, constants, reproducible seeds
│   ├── data_loader.py              # Loader with built-in leakage verification
│   ├── preprocessing.py            # Negation-preserving NLP pipeline
│   ├── feature_engineering.py      # TF-IDF sparse matrix builder (float32)
│   ├── models.py                   # Model registry & persistence
│   ├── training.py                 # Cross-validation orchestrator
│   ├── tuning.py                   # Stratified GridSearchCV
│   ├── evaluation.py               # Accuracy, F1, Top-5, Confusion Matrix
│   ├── error_analysis.py           # Confusion pairs, confidence diagnostics
│   ├── routing.py                  # Routing logic, reverse index, priority engine
│   └── utils.py                    # Benchmarking & formatting helpers
├── app/
│   ├── streamlit_app.py            # Multi-page dashboard home
│   └── pages/
│       ├── 1_EDA.py                # EDA distributions & token frequencies
│       ├── 2_Model_Comparison.py   # Metrics comparison tables & charts
│       ├── 3_Predict.py            # Live single & batch ticket router
│       ├── 4_Error_Analysis.py     # Misclassification & heatmap explorer
│       └── 5_Routing.py            # Departmental workload & accuracy analysis
├── models/                         # Serialized joblib models (7.9 MB total)
├── outputs/                        # Plots, JSON metrics, error analysis
│   └── eda/                        # Distribution & imbalance visualizations
├── tests/
│   ├── test_preprocessing.py       # Preprocessing integrity tests
│   ├── test_feature_engineering.py # Feature extraction tests
│   ├── test_models.py              # Model interface & prediction tests
│   ├── test_routing.py             # Department mapping & priority tests
│   ├── test_leakage.py             # Anti-leakage isolation tests
│   └── test_app.py                 # Streamlit headless integration tests
└── scripts/
    ├── run_pipeline.py             # End-to-end pipeline runner
    └── run_app.py                  # Streamlit launcher script
```

---

## Installation & Quickstart

### 1. Prerequisites
- Python 3.10+ (Tested up to Python 3.14)
- Git

### 2. Setup Environment
```bash
# Clone the repository
git clone <repository-url>
cd "DS lab project"

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```
All 34 tests execute and pass in under 10 seconds.

### 4. Execute End-to-End Pipeline
```bash
python scripts/run_pipeline.py
```
This runs data loading, EDA generation, text preprocessing, TF-IDF feature extraction, 5-fold cross-validation, hyperparameter tuning, test set evaluation, error analysis, and ticket routing.

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Access the application at `http://localhost:8501`.

---

## Streamlit Dashboard Features

1. **Overview**: Executive summary of system health, active baseline, and live performance metrics.
2. **1_EDA**: Category frequency bar charts, text length distributions, token frequencies, and class imbalance stats.
3. **2_Model_Comparison**: Side-by-side performance comparison of all 3 models on the official test set.
4. **3_Predict**:
   - **Interactive Classifier**: Input customer queries to see predicted intent, confidence score, department, and priority badge.
   - **Top-5 Intent Projections**: View probability distribution over the top 5 candidates.
   - **Batch Processor**: Upload a CSV file of customer queries and download enriched CSV with classifications and departments.
5. **4_Error_Analysis**: Diagnostic views showing most confused category pairs, worst-performing classes, and prediction confidence distributions.
6. **5_Routing**: Department-level ticket distribution, workload simulation, and departmental accuracy breakdown.

---

## Verification & Quality Gates

| Gate | Description | Status |
|---|---|---|
| **Data Integrity** | 10,003 train / 3,080 test, 77 categories, 0 text overlap | PASSED |
| **Leakage Isolation** | Vectorizer fit exclusively on train; test vocab isolated | PASSED |
| **Unit & Integration** | 34 automated tests across preprocessing, models, routing, leakage, and app | PASSED |
| **Resource Budget** | Models: 7.9 MB (budget < 50 MB); Training: 38s (budget < 20 min) | PASSED |
| **Git Baseline** | Clean history with milestone tags: `v0.1-baseline`, `v0.9-pipeline-complete`, `v1.0-release` | PASSED |

---

## License

MIT License. Designed for high-reliability financial and customer service operations.
