"""SupportIQ - Dataset Analytics Page.

Visual exploration of BANKING77 corpus telemetry, feature space, and class balance.
Matches Stitch Dataset Analytics design specification.
"""
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

from app.components.stitch_theme import (
    apply_stitch_styles,
    render_sidebar_branding,
    render_sidebar_footer,
    render_page_header,
    render_kpi_card,
    style_plot_axes,
)
from src.config import TRAIN_PATH, LABEL_COL, TEXT_COL

st.set_page_config(
    page_title="SupportIQ - Dataset Analytics",
    page_icon="📊",
    layout="wide",
)

apply_stitch_styles()
render_sidebar_branding()
render_sidebar_footer()

render_page_header(
    title="Dataset Analytics",
    subtitle="Explore the BANKING77 customer-support dataset structure, class distributions, lexical variance, and routing mappings across production fine-tuning baselines.",
    tag="CORPUS TELEMETRY & FEATURE SPACE",
)


@st.cache_data
def load_train_df():
    return pd.read_csv(TRAIN_PATH)


train_df = load_train_df()
vc = train_df[LABEL_COL].value_counts()
char_lengths = train_df[TEXT_COL].str.len()
word_counts = train_df[TEXT_COL].str.split().str.len()

# 4 Primary KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        render_kpi_card(
            title="Training Samples",
            value=f"{len(train_df):,}",
            subtext="76.5% of total corpus",
            badge_text="Stratified 5-Fold",
            badge_type="emerald",
        ),
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        render_kpi_card(
            title="Test Samples",
            value="3,080",
            subtext="23.5% official holdout",
            badge_text="Zero Leakage",
            badge_type="blue",
        ),
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        render_kpi_card(
            title="Intent Classes",
            value=f"{train_df[LABEL_COL].nunique()}",
            subtext="~130 samples per intent",
            badge_text="Balanced Test",
            badge_type="gray",
        ),
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        render_kpi_card(
            title="Avg Text Length",
            value=f"{char_lengths.mean():.0f} chars",
            subtext=f"Median: {char_lengths.median():.0f} chars · {word_counts.mean():.1f} words",
            badge_text="Clean NLP",
            badge_type="emerald",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Category Distribution Section
st.subheader("Category Distribution (Training Set)")
st.markdown(
    "<span style='color: var(--stitch-muted); font-size: 13px;'>Training split contains intentional natural imbalance (35 to 187 samples), while test set is perfectly balanced (40 samples per category).</span>",
    unsafe_allow_html=True,
)

tab_chart, tab_table = st.tabs(["📊 Distribution Bar Chart", "📋 Data Table & Search"])

with tab_chart:
    fig, ax = plt.subplots(figsize=(12, 16))
    plot_colors = style_plot_axes(fig, ax)
    
    # Clean Stitch Navy to Cobalt gradient
    palette = sns.blend_palette([plot_colors["primary"], plot_colors["secondary"]], n_colors=len(vc))
    vc.plot(kind='barh', ax=ax, color=palette)
    
    ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold', color=plot_colors['text'])
    ax.set_title('BANKING77 Training Set Intent Frequency (77 Classes)', fontsize=14, fontweight='bold', color=plot_colors['heading'], pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3, color=plot_colors['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_table:
    search_q = st.text_input("Filter categories by name:", placeholder="e.g., card, transfer, fee, pin...")
    cat_summary = pd.DataFrame({
        'Category': vc.index,
        'Train Count': vc.values,
        'Percentage': (vc.values / len(train_df) * 100).round(2),
        'Test Count': 40,
    })
    if search_q:
        cat_summary = cat_summary[cat_summary['Category'].str.contains(search_q, case=False)]
    st.dataframe(cat_summary, use_container_width=True, height=450, hide_index=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Text Length Analysis Section
st.subheader("Text Length & Word Frequency Diagnostics")
c_len1, c_len2 = st.columns(2)

with c_len1:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_colors = style_plot_axes(fig, ax)
    char_lengths.hist(bins=40, ax=ax, color=plot_colors['primary'], edgecolor=plot_colors['surface'], alpha=0.9)
    ax.axvline(char_lengths.mean(), color=plot_colors['danger'], linestyle='--', linewidth=2, label=f'Mean: {char_lengths.mean():.1f}')
    ax.axvline(char_lengths.median(), color=plot_colors['warning'], linestyle=':', linewidth=2, label=f'Median: {char_lengths.median():.0f}')
    ax.set_xlabel('Character Length', fontsize=10, fontweight='bold', color=plot_colors['text'])
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold', color=plot_colors['text'])
    ax.set_title('Character Length Distribution', fontsize=12, fontweight='bold', color=plot_colors['heading'])
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=plot_colors['grid'])
    ax.legend(frameon=True, facecolor=plot_colors['surface'], edgecolor=plot_colors['border'], labelcolor=plot_colors['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with c_len2:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_colors = style_plot_axes(fig, ax)
    word_counts.hist(bins=30, ax=ax, color=plot_colors['secondary'], edgecolor=plot_colors['surface'], alpha=0.9)
    ax.axvline(word_counts.mean(), color=plot_colors['danger'], linestyle='--', linewidth=2, label=f'Mean: {word_counts.mean():.1f}')
    ax.axvline(word_counts.median(), color=plot_colors['warning'], linestyle=':', linewidth=2, label=f'Median: {word_counts.median():.0f}')
    ax.set_xlabel('Word Count', fontsize=10, fontweight='bold', color=plot_colors['text'])
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold', color=plot_colors['text'])
    ax.set_title('Word Count Distribution', fontsize=12, fontweight='bold', color=plot_colors['heading'])
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=plot_colors['grid'])
    ax.legend(frameon=True, facecolor=plot_colors['surface'], edgecolor=plot_colors['border'], labelcolor=plot_colors['text'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Sample Browser Section
st.subheader("Intent Utterance Browser")
st.markdown("<span style='color: var(--stitch-muted); font-size: 13px;'>Inspect authentic training examples across any of the 77 banking categories.</span>", unsafe_allow_html=True)

selected_cat = st.selectbox("Select Intent Category:", sorted(train_df[LABEL_COL].unique()))
cat_samples = train_df[train_df[LABEL_COL] == selected_cat][TEXT_COL].tolist()

st.markdown(f"**Found {len(cat_samples)} training utterances for `{selected_cat}`:**")
cols_samp = st.columns(min(3, len(cat_samples[:6])))
for idx, text in enumerate(cat_samples[:6]):
    col = cols_samp[idx % len(cols_samp)]
    with col:
        st.markdown(
            f"""
            <div style="background: var(--stitch-surface); border: 1px solid var(--stitch-border); border-radius: 8px; padding: 14px; margin-bottom: 12px; height: 110px; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="font-size: 13px; color: var(--stitch-text); line-height: 1.4;">"{text}"</div>
                <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--stitch-info); font-weight: 600;">Sample #{idx+1}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
