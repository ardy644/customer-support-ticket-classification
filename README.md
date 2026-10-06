# Intelligent Customer Support Ticket Classification and Routing System

An end-to-end, production-ready machine learning system that automatically classifies banking customer support queries across all **77 fine-grained intents** from the **BANKING77 dataset** and routes them to appropriate specialized support departments.

Built strictly with **traditional Machine Learning and NLP** (CPU-first, zero deep learning, zero external APIs) adhering to anti-leakage protocols and the official BANKING77 train/test split.

---

## Key Highlights

- **Traditional ML Only**: TF-IDF + Multinomial Naive Bayes, Logistic Regression, and Linear SVM. No CNNs, BERT, Transformers, LLMs, or external APIs.
- **CPU-First & Resource Efficient**: Inference uses sparse TF-IDF and CPU-based linear models. The checked-in training pipeline can regenerate the model artifacts; its total runtime depends on the machine and tuning settings. The current local model artifacts occupy about 7.9 MiB.
- **Zero Data Leakage**: Vectorizer is fitted strictly on the official training split (`10,003` samples). Evaluation is computed strictly on the official held-out test split (`3,080` samples).
- **High Accuracy**:
  - **Logistic Regression**: **87.18% Accuracy**, **87.18% Macro-F1**, **98.18% Top-5 Accuracy**
  - **Linear SVM**: **86.82% Accuracy**, **86.76% Macro-F1**, **97.11% Top-5 Accuracy**
  - **Multinomial Naive Bayes**: **83.96% Accuracy**, **83.87% Macro-F1**, **97.01% Top-5 Accuracy**
- **Intelligent Routing**: 77 fine-grained intents mapped to 10 operational departments with priority scoring (`URGENT`, `HIGH`, `MEDIUM`, `NORMAL`). Overall departmental routing accuracy exceeds **94%**.
- **Interactive Streamlit Dashboard**: 6-page application featuring an overview, EDA, model comparison, real-time single & batch prediction, error analysis, and routing analytics.
- **Automated Test Suite**: 35 unit, integration, and anti-leakage tests via `pytest`.

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
├── models/                         # Generated joblib models (not tracked by Git)
├── outputs/                        # Generated plots, JSON metrics, error analysis (not tracked by Git)
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
- Python 3.10+ (Python 3.11 or 3.12 is recommended for easiest package installation on Windows)
- Git

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/ardy644/customer-support-ticket-classification.git
cd customer-support-ticket-classification

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare NLTK data
The preprocessing module uses NLTK's English stopwords, tokenizer data, and WordNet corpora. Install them once before running tests, training, or prediction:
```bash
python -c "import nltk; [nltk.download(name) for name in ('stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4')]"
```

### 4. Run Automated Tests
```bash
python -m pytest tests/ -v
```
The current suite contains 35 tests. Runtime varies with the machine.

### 5. Prepare model and result artifacts
Model files and generated evaluation outputs are intentionally excluded from Git by `.gitignore`. A fresh clone therefore does not include the trained model or saved result files.

For a self-contained demo without retraining, copy the `models/` and `outputs/` directories from a prepared project folder into the clone. The current local copies are about 7.9 MiB and 1.0 MiB, respectively.

Alternatively, regenerate them from the included dataset by running the pipeline below. This performs training and evaluation; it is not needed when the prepared artifact folders are copied.

### 6. Execute End-to-End Pipeline (only if artifacts were not copied)
```bash
python scripts/run_pipeline.py
```
This runs data loading, EDA generation, text preprocessing, TF-IDF feature extraction, 5-fold cross-validation, hyperparameter tuning, test set evaluation, error analysis, and ticket routing.

### 7. Launch the Streamlit Dashboard
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
| **Unit & Integration** | 35 automated tests across preprocessing, models, routing, leakage, and app | PASSED |
| **Resource Budget** | Current local model artifacts: about 7.9 MiB; CPU-first sparse TF-IDF and linear models | PASSED |
| **Git Baseline** | Stable UI and theme release tags: `v1.1-ui`, `v1.1.1` | PASSED |

---

## Image-OCR Fallback Version (v2.0-image-ocr)

### 1. Purpose of the Fallback Version
In enterprise and financial customer service operations, users frequently submit **screenshots** of mobile app errors, wire transfer receipts, digital statements, and transaction disputes rather than manually typing raw text. 

The **Image-OCR Fallback Version** (`feature/image-ocr-fallback`) introduces multimodal image input capabilities while completely preserving the validated text classification system.

> [!NOTE]
> **Important Technical Clarification:** This is **not** direct computer vision classification. The machine learning models (Logistic Regression, Linear SVM) operate strictly on natural language representations. The architecture is:
> 
> $$\text{Image} \xrightarrow{\text{OCR}} \text{Extracted Text} \xrightarrow{\text{NLP}} \text{TF-IDF} \xrightarrow{\text{ML}} \text{Intent Classification} \xrightarrow{\text{Rules}} \text{Support Department}$$

### 2. Comparative Processing Architectures

- **Standard Text Version (Page 3 - Live Ticket Classifier):**
  $$\text{User Text Input} \longrightarrow \text{NLP Preprocessing} \longrightarrow \text{TF-IDF} \longrightarrow \text{Linear Classifier} \longrightarrow \text{Routing \& Priority}$$
- **Image Fallback Version (Page 6 - Image Ticket Classifier):**
  $$\text{Ticket Image / Screenshot} \longrightarrow \text{Image Preprocessing} \longrightarrow \text{Tesseract OCR} \longrightarrow \text{Extracted Text} \longrightarrow \text{NLP Preprocessing} \longrightarrow \text{TF-IDF} \longrightarrow \text{Linear Classifier} \longrightarrow \text{Routing \& Priority}$$

### 3. OCR Technology Stack
- **Python OCR Library:** `pytesseract` (Python wrapper for Tesseract)
- **Image Preprocessing:** `Pillow` (PIL) for RGB normalization, grayscale conversion, resolution upscaling (for images $< 800\text{px}$ width), contrast enhancement ($\times 1.8$), and sharpness tuning.
- **Underlying OCR Engine:** Google Tesseract OCR engine (CPU-first, local execution, zero cloud APIs, zero external network requests).

### 4. Tesseract OCR Installation on Windows
The Python package `pytesseract` requires the local Tesseract OCR engine binary on the operating system:

```powershell
# Option A: Windows Package Manager (recommended)
winget install UB-Mannheim.TesseractOCR

# Option B: Manual Installer
# Download and install the 64-bit installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

The system automatically detects Tesseract in standard locations:
- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- `%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe`
- Or via custom environment variable: `$env:TESSERACT_CMD = "C:\Path\To\tesseract.exe"`

*(Note: The page includes built-in interactive demo screenshot scenarios so the entire workflow can be demonstrated even before installing the local engine binary.)*

### 5. Supported Formats & Capabilities
- **File Formats:** `.png`, `.jpg`, `.jpeg`
- **Telemetry Separation:** Displays **OCR Confidence** (word-level character recognition certainty) strictly separate from **Model Confidence** (intent posterior probability).
- **Graceful Error Handling:** Explicit validation for missing images, corrupt files, unreadable/empty text, missing OCR engines, and classification failures.

---

## License

MIT License. Designed for high-reliability financial and customer service operations.
