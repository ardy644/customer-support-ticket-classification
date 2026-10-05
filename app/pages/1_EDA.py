"""EDA Dashboard Page."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.config import TRAIN_PATH, LABEL_COL, TEXT_COL, EDA_DIR

st.set_page_config(page_title="EDA - BANKING77", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.markdown("All analysis is performed on the **training set only** to prevent data leakage.")

@st.cache_data
def load_train_data():
    return pd.read_csv(TRAIN_PATH)

train_df = load_train_data()

# Summary metrics
st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", f"{len(train_df):,}")
col2.metric("Categories", train_df[LABEL_COL].nunique())
col3.metric("Avg Text Length", f"{train_df[TEXT_COL].str.len().mean():.0f} chars")
col4.metric("Avg Word Count", f"{train_df[TEXT_COL].str.split().str.len().mean():.1f} words")

st.markdown("---")

# Category distribution
st.subheader("Category Distribution")
vc = train_df[LABEL_COL].value_counts()

tab1, tab2 = st.tabs(["Bar Chart", "Data Table"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 16))
    vc.plot(kind='barh', ax=ax, color=sns.color_palette('viridis', len(vc)))
    ax.set_xlabel('Number of Samples')
    ax.set_title('Category Distribution (Training Set)')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.dataframe(
        pd.DataFrame({'Category': vc.index, 'Count': vc.values, 'Percentage': (vc.values/vc.sum()*100).round(2)}),
        use_container_width=True, height=400
    )

st.markdown("---")

# Text length analysis
st.subheader("Text Length Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    train_df[TEXT_COL].str.len().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Text Length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Text Length Distribution')
    mean_len = train_df[TEXT_COL].str.len().mean()
    ax.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.0f}')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    train_df[TEXT_COL].str.split().str.len().hist(bins=30, ax=ax, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Word Count Distribution')
    mean_wc = train_df[TEXT_COL].str.split().str.len().mean()
    ax.axvline(mean_wc, color='red', linestyle='--', label=f'Mean: {mean_wc:.0f}')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# Class imbalance
st.subheader("Class Imbalance")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(vc.values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(vc.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {vc.mean():.0f}')
ax.axvline(vc.min(), color='green', linestyle='--', linewidth=2, label=f'Min: {vc.min()} ({vc.idxmin()})')
ax.axvline(vc.max(), color='purple', linestyle='--', linewidth=2, label=f'Max: {vc.max()} ({vc.idxmax()})')
ax.set_xlabel('Samples per Category')
ax.set_ylabel('Number of Categories')
ax.set_title('Class Imbalance Distribution')
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.metric("Imbalance Ratio (max/min)", f"{vc.max()/vc.min():.2f}x")

st.markdown("---")

# Sample browser
st.subheader("Sample Text Browser")
selected_cat = st.selectbox("Select a category:", sorted(train_df[LABEL_COL].unique()))
samples = train_df[train_df[LABEL_COL] == selected_cat][TEXT_COL].head(10)
for i, text in enumerate(samples, 1):
    st.markdown(f"**{i}.** {text}")
