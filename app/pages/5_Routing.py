"""Ticket Routing Page."""
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
import json

from src.config import OUTPUTS_DIR, MODELS_DIR, TEST_PATH, LABEL_COL, TEXT_COL
from src.routing import ROUTING_MAP, get_department, get_routing_stats, TicketRouter

st.set_page_config(page_title="Routing - BANKING77", page_icon="🎯", layout="wide")
st.title("🎯 Ticket Routing Dashboard")

# Department overview
st.subheader("Department Overview")
dept_data = []
for dept, intents in ROUTING_MAP.items():
    dept_data.append({
        'Department': dept,
        'Intents': len(intents),
        'Intent List': ', '.join(sorted(intents)),
    })
dept_df = pd.DataFrame(dept_data)

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(
        dept_df[['Department', 'Intents']],
        use_container_width=True, hide_index=True
    )

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(dept_df['Department'], dept_df['Intents'], color='teal')
    ax.set_xlabel('Number of Intents')
    ax.set_title('Intents per Department')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# Department intent browser
st.subheader("Intent Browser")
selected_dept = st.selectbox("Select a department:", list(ROUTING_MAP.keys()))
st.write(f"**Intents handled by {selected_dept}:**")
for intent in sorted(ROUTING_MAP[selected_dept]):
    st.markdown(f"- `{intent}`")

st.markdown("---")

# Routing stats on test set
st.subheader("Routing Performance on Test Set")
routing_stats_path = OUTPUTS_DIR / "routing_stats.json"

if routing_stats_path.exists():
    with open(routing_stats_path) as f:
        routing_stats = json.load(f)
    
    stats_rows = []
    for dept, stats in routing_stats.items():
        stats_rows.append({
            'Department': dept,
            'Total Routed': stats['total_routed'],
            'Correctly Routed': stats['correctly_routed'],
            'Routing Accuracy': f"{stats['routing_accuracy']:.2%}",
        })
    
    stats_df = pd.DataFrame(stats_rows)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Pie chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    depts = [r['Department'] for r in stats_rows]
    totals = [r['Total Routed'] for r in stats_rows]
    accuracies = [routing_stats[d]['routing_accuracy'] for d in depts]
    
    axes[0].pie(totals, labels=depts, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Ticket Distribution by Department')
    
    axes[1].barh(depts, accuracies, color='seagreen')
    axes[1].set_xlabel('Routing Accuracy')
    axes[1].set_title('Routing Accuracy by Department')
    axes[1].set_xlim(0, 1.05)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Run the pipeline to generate routing statistics.")
