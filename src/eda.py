"""Exploratory Data Analysis for BANKING77 dataset."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.config import EDA_DIR, TEXT_COL, LABEL_COL


def ensure_eda_dir():
    """Create EDA output directory if it doesn't exist."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def plot_category_distribution(train_df: pd.DataFrame):
    """Plot category distribution bar chart."""
    ensure_eda_dir()
    vc = train_df[LABEL_COL].value_counts()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    vc.plot(kind='barh', ax=ax, color=sns.color_palette('viridis', len(vc)))
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Category')
    ax.set_title('BANKING77 Category Distribution (Training Set)')
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(EDA_DIR / 'category_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: category_distribution.png")


def plot_text_length_distribution(train_df: pd.DataFrame):
    """Plot text length distribution histogram."""
    ensure_eda_dir()
    lengths = train_df[TEXT_COL].str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Text Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Text Length Distribution')
    axes[0].axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean: {lengths.mean():.0f}')
    axes[0].axvline(lengths.median(), color='orange', linestyle='--', label=f'Median: {lengths.median():.0f}')
    axes[0].legend()
    
    # Word count histogram
    word_counts = train_df[TEXT_COL].str.split().str.len()
    axes[1].hist(word_counts, bins=40, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Word Count Distribution')
    axes[1].axvline(word_counts.mean(), color='red', linestyle='--', label=f'Mean: {word_counts.mean():.0f}')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(EDA_DIR / 'text_length_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: text_length_distribution.png")


def plot_text_length_by_category(train_df: pd.DataFrame, top_n: int = 20):
    """Plot text length boxplot for top-N most frequent categories."""
    ensure_eda_dir()
    top_cats = train_df[LABEL_COL].value_counts().head(top_n).index.tolist()
    subset = train_df[train_df[LABEL_COL].isin(top_cats)].copy()
    subset['text_length'] = subset[TEXT_COL].str.len()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=subset, y=LABEL_COL, x='text_length', ax=ax,
                order=top_cats, palette='viridis')
    ax.set_xlabel('Text Length (characters)')
    ax.set_ylabel('Category')
    ax.set_title(f'Text Length by Category (Top {top_n})')
    plt.tight_layout()
    fig.savefig(EDA_DIR / 'text_length_by_category.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: text_length_by_category.png")


def plot_top_words(train_df: pd.DataFrame, top_n: int = 30):
    """Plot top-N most frequent words across all training texts."""
    ensure_eda_dir()
    all_words = ' '.join(train_df[TEXT_COL].values).lower().split()
    word_freq = Counter(all_words).most_common(top_n)
    words, counts = zip(*word_freq)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(words)), counts, color='teal')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Most Frequent Words (Training Set)')
    plt.tight_layout()
    fig.savefig(EDA_DIR / 'top_words.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: top_words.png")


def plot_top_bigrams(train_df: pd.DataFrame, top_n: int = 25):
    """Plot top-N most frequent bigrams."""
    ensure_eda_dir()
    from sklearn.feature_extraction.text import CountVectorizer
    
    cv = CountVectorizer(ngram_range=(2, 2), max_features=top_n, stop_words='english')
    bigram_matrix = cv.fit_transform(train_df[TEXT_COL])
    bigram_counts = bigram_matrix.sum(axis=0).A1
    bigrams = cv.get_feature_names_out()
    
    # Sort by frequency
    sorted_idx = bigram_counts.argsort()[::-1]
    bigrams = bigrams[sorted_idx][:top_n]
    counts = bigram_counts[sorted_idx][:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(bigrams)), counts, color='darkorange')
    ax.set_yticks(range(len(bigrams)))
    ax.set_yticklabels(bigrams)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Bigrams (Training Set)')
    plt.tight_layout()
    fig.savefig(EDA_DIR / 'top_bigrams.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: top_bigrams.png")


def plot_class_imbalance(train_df: pd.DataFrame):
    """Visualize class imbalance with min/max/mean markers."""
    ensure_eda_dir()
    vc = train_df[LABEL_COL].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vc.values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(vc.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {vc.mean():.0f}')
    ax.axvline(vc.min(), color='green', linestyle='--', linewidth=2, label=f'Min: {vc.min()} ({vc.idxmin()})')
    ax.axvline(vc.max(), color='purple', linestyle='--', linewidth=2, label=f'Max: {vc.max()} ({vc.idxmax()})')
    ax.set_xlabel('Number of Samples per Category')
    ax.set_ylabel('Number of Categories')
    ax.set_title('Class Imbalance Distribution')
    ax.legend()
    plt.tight_layout()
    fig.savefig(EDA_DIR / 'class_imbalance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: class_imbalance.png")


def plot_sample_texts(train_df: pd.DataFrame, n_categories: int = 5, n_samples: int = 3):
    """Print sample texts from random categories for manual inspection."""
    ensure_eda_dir()
    categories = train_df[LABEL_COL].value_counts().head(n_categories).index.tolist()
    
    lines = []
    for cat in categories:
        lines.append(f"\n=== {cat} ===")
        samples = train_df[train_df[LABEL_COL] == cat][TEXT_COL].head(n_samples)
        for i, text in enumerate(samples, 1):
            lines.append(f"  {i}. {text}")
    
    report = '\n'.join(lines)
    with open(EDA_DIR / 'sample_texts.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("  Saved: sample_texts.txt")
    print(report)


def generate_eda_summary(train_df: pd.DataFrame) -> dict:
    """Generate EDA summary statistics."""
    vc = train_df[LABEL_COL].value_counts()
    lengths = train_df[TEXT_COL].str.len()
    word_counts = train_df[TEXT_COL].str.split().str.len()
    
    summary = {
        'total_samples': len(train_df),
        'num_categories': train_df[LABEL_COL].nunique(),
        'min_samples_per_cat': int(vc.min()),
        'max_samples_per_cat': int(vc.max()),
        'mean_samples_per_cat': float(vc.mean()),
        'imbalance_ratio': float(vc.max() / vc.min()),
        'smallest_category': vc.idxmin(),
        'largest_category': vc.idxmax(),
        'mean_text_length': float(lengths.mean()),
        'median_text_length': float(lengths.median()),
        'max_text_length': int(lengths.max()),
        'min_text_length': int(lengths.min()),
        'mean_word_count': float(word_counts.mean()),
    }
    return summary


def run_full_eda(train_df: pd.DataFrame) -> dict:
    """Run all EDA analyses and save outputs."""
    print("\nRunning Exploratory Data Analysis...")
    print("="*50)
    
    summary = generate_eda_summary(train_df)
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Categories: {summary['num_categories']}")
    print(f"  Imbalance ratio: {summary['imbalance_ratio']:.2f}")
    print(f"  Mean text length: {summary['mean_text_length']:.1f} chars")
    
    plot_category_distribution(train_df)
    plot_text_length_distribution(train_df)
    plot_text_length_by_category(train_df)
    plot_top_words(train_df)
    plot_top_bigrams(train_df)
    plot_class_imbalance(train_df)
    plot_sample_texts(train_df)
    
    import json
    ensure_eda_dir()
    with open(EDA_DIR / 'eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved: eda_summary.json")
    
    print("EDA complete!\n")
    return summary
