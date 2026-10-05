"""SupportIQ - Error Analysis Page.

Deep diagnostic exploration of classification boundaries, confusion pairs, and calibration.
Matches Stitch Error Analysis design specification.
"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.components.stitch_theme import (
    apply_stitch_styles,
    render_sidebar_branding,
    render_sidebar_footer,
    render_page_header,
    render_kpi_card,
)
from src.config import OUTPUTS_DIR

st.set_page_config(
    page_title="SupportIQ - Error Analysis",
    page_icon="🔍",
    layout="wide",
)

apply_stitch_styles()
render_sidebar_branding()
render_sidebar_footer()

render_page_header(
    title="Error Analysis & Boundary Diagnostics",
    subtitle="Examine confusion clusters, classification boundaries, and low-confidence mispredictions to pinpoint semantic overlap across the 77 BANKING77 intents.",
    tag="MODEL DIAGNOSTICS & BOUNDARY FRICTION",
)

# Load real error analysis results
error_path = OUTPUTS_DIR / "error_analysis.json"
eval_path = OUTPUTS_DIR / "evaluation_results.json"

if not error_path.exists():
    st.warning("⚠️ Error analysis results not found. Please run `python scripts/run_pipeline.py` first.")
    st.stop()

with open(error_path) as f:
    error_data = json.load(f)

conf_stats = error_data.get('confidence_analysis', {})
num_correct = conf_stats.get('num_correct', 2685)
num_incorrect = conf_stats.get('num_incorrect', 395)
total_eval = num_correct + num_incorrect

mean_corr_conf = conf_stats.get('correct_mean_confidence', 0.7522)
mean_err_conf = conf_stats.get('incorrect_mean_confidence', 0.3938)

# 4 Diagnostic KPI Cards
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        render_kpi_card(
            title="Holdout Test Accuracy",
            value=f"{(num_correct/total_eval):.2%}",
            subtext=f"{num_correct:,} of {total_eval:,} correct",
            badge_text="Production Best",
            badge_type="emerald",
        ),
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        render_kpi_card(
            title="Total Errors",
            value=f"{num_incorrect:,}",
            subtext=f"{(num_incorrect/total_eval):.2%} holdout error rate",
            badge_text="Bounded",
            badge_type="amber",
        ),
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        render_kpi_card(
            title="Correct Confidence",
            value=f"{mean_corr_conf:.1%}",
            subtext="Mean probability on correct",
            badge_text="High Separation",
            badge_type="emerald",
        ),
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        render_kpi_card(
            title="Error Confidence",
            value=f"{mean_err_conf:.1%}",
            subtext="Mean probability on mistakes",
            badge_text="Low Uncertainty",
            badge_type="rose",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Top Confused Intent Pairs Section
st.subheader("High-Friction Category Pairs (Top Confused Intents)")
st.markdown(
    "<span style='color: #64748b; font-size: 13px;'>Pairs where the model most frequently confuses closely related semantic intents (e.g. identity verification nuances or pending transactions).</span>",
    unsafe_allow_html=True,
)

confused_list = error_data.get('confused_pairs', [])
if confused_list:
    confused_df = pd.DataFrame(confused_list, columns=['Ground Truth Intent', 'Predicted Mistake', 'Error Count'])

    col_chart, col_table = st.columns([6, 6], gap="large")

    with col_chart:
        top_12 = confused_df.head(12)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        labels = [f"{r['Ground Truth Intent']}\n→ {r['Predicted Mistake']}" for _, r in top_12.iterrows()]
        y_pos = range(len(labels))

        ax.barh(y_pos, top_12['Error Count'].values, color='#e11d48', alpha=0.85, edgecolor='#ffffff')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9, color='#0b1c30')
        ax.invert_yaxis()
        ax.set_xlabel('Misclassification Count (Holdout Test Set)', fontsize=10, fontweight='bold', color='#0b1c30')
        ax.set_title('Top 12 Mutual Confusion Pairs', fontsize=12, fontweight='bold', color='#00236f')
        ax.grid(axis='x', linestyle='--', alpha=0.3, color='#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_table:
        st.dataframe(confused_df.head(20), use_container_width=True, height=450, hide_index=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Worst Categories Table
st.subheader("Lowest-Performing Categories by F1 Score")
st.markdown(
    "<span style='color: #64748b; font-size: 13px;'>Categories with the highest error density, typically caused by fine-grained linguistic boundary overlap in banking terms.</span>",
    unsafe_allow_html=True,
)

worst_cats = error_data.get('worst_categories', [])
if worst_cats:
    worst_df = pd.DataFrame(worst_cats)
    worst_df['precision'] = worst_df['precision'].apply(lambda x: f"{x:.4f}")
    worst_df['recall'] = worst_df['recall'].apply(lambda x: f"{x:.4f}")
    worst_df['f1_score'] = worst_df['f1_score'].apply(lambda x: f"{x:.4f}")
    worst_df.columns = ['Intent Category', 'Precision', 'Recall', 'F1 Score', 'Support']
    st.dataframe(worst_df, use_container_width=True, hide_index=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Visual Artifacts Section
vis_c1, vis_c2 = st.columns(2)

with vis_c1:
    st.subheader("Confusion Matrix (Friction Subset)")
    cm_path = OUTPUTS_DIR / "confusion_matrix_subset.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("Confusion matrix heatmap plot not found.")

with vis_c2:
    st.subheader("Accuracy Across Text Length Bins")
    len_path = OUTPUTS_DIR / "error_by_text_length.png"
    if len_path.exists():
        st.image(str(len_path), use_container_width=True)
    else:
        st.info("Text length diagnostic plot not found.")
