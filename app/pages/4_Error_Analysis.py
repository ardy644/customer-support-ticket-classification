"""Error Analysis Page."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.config import OUTPUTS_DIR

st.set_page_config(page_title="Error Analysis - BANKING77", page_icon="🔍", layout="wide")
st.title("🔍 Error Analysis")

# Load error analysis results
error_path = OUTPUTS_DIR / "error_analysis.json"
if not error_path.exists():
    st.warning("⚠️ No error analysis results found. Please run the pipeline first.")
    st.stop()

with open(error_path) as f:
    error_data = json.load(f)

# Summary metrics
st.subheader("Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Model", error_data.get('model_name', 'Unknown'))
col2.metric("Categories with Errors", error_data.get('num_misclassified_categories', 0))
col3.metric("Total Misclassified", error_data.get('total_misclassified', 0))

st.markdown("---")

# Most confused pairs
st.subheader("Most Confused Category Pairs")
confused = error_data.get('confused_pairs', [])
if confused:
    confused_df = pd.DataFrame(confused, columns=['True Label', 'Predicted Label', 'Count'])
    
    tab1, tab2 = st.tabs(["Chart", "Table"])
    
    with tab1:
        top_n = min(15, len(confused_df))
        subset = confused_df.head(top_n)
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [f"{r['True Label']}\n→ {r['Predicted Label']}" for _, r in subset.iterrows()]
        ax.barh(range(len(labels)), subset['Count'].values, color='coral')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Misclassification Count')
        ax.set_title('Top Most Confused Category Pairs')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with tab2:
        st.dataframe(confused_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Worst categories
st.subheader("Worst Performing Categories (by F1 Score)")
worst = error_data.get('worst_categories', [])
if worst:
    worst_df = pd.DataFrame(worst)
    st.dataframe(worst_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Confidence analysis
st.subheader("Confidence Analysis")
conf = error_data.get('confidence_analysis', {})
if conf and 'note' not in conf:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Correct Predictions**")
        st.metric("Mean Confidence", f"{conf.get('correct_mean_confidence', 0):.4f}")
        st.metric("Median Confidence", f"{conf.get('correct_median_confidence', 0):.4f}")
        st.metric("Count", conf.get('num_correct', 0))
    with col2:
        st.markdown("**Incorrect Predictions**")
        st.metric("Mean Confidence", f"{conf.get('incorrect_mean_confidence', 0):.4f}")
        st.metric("Median Confidence", f"{conf.get('incorrect_median_confidence', 0):.4f}")
        st.metric("Count", conf.get('num_incorrect', 0))

st.markdown("---")

# Confusion matrix image
st.subheader("Confusion Matrix (Top Confused Categories)")
cm_path = OUTPUTS_DIR / "confusion_matrix_subset.png"
if cm_path.exists():
    st.image(str(cm_path), use_container_width=True)
else:
    st.info("Confusion matrix plot not found.")

# Error by text length
st.subheader("Accuracy by Text Length")
err_len_path = OUTPUTS_DIR / "error_by_text_length.png"
if err_len_path.exists():
    st.image(str(err_len_path), use_container_width=True)
