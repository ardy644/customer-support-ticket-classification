"""Model Comparison Page."""
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
import numpy as np

from src.config import OUTPUTS_DIR

st.set_page_config(page_title="Model Comparison - BANKING77", page_icon="🏆", layout="wide")
st.title("🏆 Model Comparison")

# Load evaluation results
eval_path = OUTPUTS_DIR / "evaluation_results.json"
tuning_path = OUTPUTS_DIR / "tuning_results.json"

if not eval_path.exists():
    st.warning("⚠️ No evaluation results found. Please run the pipeline first.")
    st.stop()

with open(eval_path) as f:
    eval_results = json.load(f)

# Summary table
st.subheader("Performance Summary")
rows = []
for name, metrics in eval_results.items():
    rows.append({
        'Model': name,
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'F1 (Macro)': f"{metrics['f1_macro']:.4f}",
        'F1 (Weighted)': f"{metrics['f1_weighted']:.4f}",
        'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
        'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")

# Bar chart comparison
st.subheader("Visual Comparison")
metric_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
model_names = list(eval_results.keys())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metric_names))
width = 0.25

for i, name in enumerate(model_names):
    values = [eval_results[name][m] for m in metric_names]
    ax.bar(x + i*width, values, width, label=name, alpha=0.85)

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(['Accuracy', 'F1 Macro', 'F1 Weighted', 'Prec Macro', 'Rec Macro'], rotation=15)
ax.legend()
ax.set_ylim(0.7, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Tuning results
if tuning_path.exists():
    st.markdown("---")
    st.subheader("Hyperparameter Tuning Results")
    with open(tuning_path) as f:
        tuning_results = json.load(f)
    
    for name, res in tuning_results.items():
        with st.expander(f"{name}", expanded=True):
            st.json(res)

# Best model highlight
st.markdown("---")
st.subheader("🏅 Best Model")
best_name = max(eval_results, key=lambda k: eval_results[k]['accuracy'])
best = eval_results[best_name]
st.success(f"**{best_name}** with accuracy **{best['accuracy']:.4f}** and F1-macro **{best['f1_macro']:.4f}**")
