"""BANKING77 Customer Support Ticket Classification & Routing System.

Streamlit Dashboard - Main Entry Point.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

st.set_page_config(
    page_title="BANKING77 Ticket Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 BANKING77 Customer Support Ticket Classification")
st.markdown("---")

st.markdown("""
### Welcome to the Intelligent Ticket Classification & Routing System

This system uses **traditional Machine Learning** to automatically classify banking customer support
tickets into **77 intent categories** and route them to the appropriate support department.

#### Models Used
| Model | Description |
|---|---|
| **Multinomial Naive Bayes** | Fast probabilistic baseline |
| **Logistic Regression** | Strong linear model with L2 regularization |
| **Linear SVM** | State-of-the-art linear classifier for text |

#### Features
- 📊 **EDA Dashboard** — Explore the BANKING77 dataset
- 🏆 **Model Comparison** — Compare model performance metrics
- 🔮 **Live Prediction** — Classify customer queries in real-time
- 🔍 **Error Analysis** — Understand model mistakes
- 🎯 **Ticket Routing** — Route tickets to support departments

👈 **Select a page from the sidebar** to get started!
""")

st.markdown("---")

# Display quick stats if models are available
try:
    import json
    results_path = project_root / "outputs" / "evaluation_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        st.subheader("📈 Quick Model Performance Summary")
        cols = st.columns(len(results))
        for col, (name, metrics) in zip(cols, results.items()):
            with col:
                st.metric(label=name, value=f"{metrics['accuracy']:.2%}", delta=f"F1: {metrics['f1_macro']:.2%}")
except Exception:
    st.info("Run the pipeline first to see model performance metrics here.")
