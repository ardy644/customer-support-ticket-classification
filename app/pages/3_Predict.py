"""SupportIQ - Live Ticket Classifier Page.

Interactive real-time inference and batch CSV processing.
Matches Stitch Live Ticket Classifier design specification.
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import joblib

from app.components.stitch_theme import (
    apply_stitch_styles,
    render_sidebar_branding,
    render_sidebar_footer,
    render_page_header,
    render_confidence_bar,
)
from src.config import MODELS_DIR
from src.routing import TicketRouter, get_department


def _safe_csv_text(value):
    """Neutralize spreadsheet formulas in user-provided text before CSV export."""
    text = "" if pd.isna(value) else str(value)
    if text.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + text
    return text

st.set_page_config(
    page_title="SupportIQ - Live Ticket Classifier",
    page_icon="🔮",
    layout="wide",
)

apply_stitch_styles()
render_sidebar_branding()
render_sidebar_footer()

render_page_header(
    title="Live Ticket Classifier",
    subtitle="Enter a customer support message or upload a CSV batch to classify banking intents, extract sublinear n-gram token saliency, and trigger real-time dispatch routing.",
    tag="INFERENCE ENGINE · LIVE PIPELINE V1.0",
)


@st.cache_resource
def load_production_pipeline():
    """Load the trained best model and fitted TF-IDF vectorizer."""
    model_path = MODELS_DIR / "best_model.joblib"
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    if not model_path.exists() or not vec_path.exists():
        return None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


model, vectorizer = load_production_pipeline()

if model is None:
    st.warning("⚠️ Production model artifacts not found. Please run the training pipeline first via `python scripts/run_pipeline.py`.")
    st.stop()

router = TicketRouter(model, vectorizer)

# Mode Navigation Tabs
tab_single, tab_batch = st.tabs(["⚡ Single Ticket Inference", "📁 Batch CSV Classification"])

# Quick validation scenario presets matching Stitch
PRESETS = {
    "Card charged twice": "I noticed my account was charged twice for the exact same transaction of £45.20 at Sainsbury's this morning. The payment appears twice on my statement.",
    "Transfer is pending": "I initiated a wire transfer to my friend's account yesterday morning, but the status is still marked as pending and the money hasn't arrived.",
    "Can't verify identity": "The mobile app keeps rejecting photos of my driver's license during the ID verification step. It says the image is blurry even though it is completely clear.",
    "ATM cash withdrawal failed": "I tried to withdraw £100 from an ATM on Oxford Street. The machine made a dispensing sound but no cash came out, yet my balance was deducted.",
    "Don't recognise payment": "There is a direct debit payment of £79.99 to an unknown merchant called TechCorp that I never authorized or signed up for.",
    "Virtual card expired": "My disposable virtual card seems to have expired before I could complete my online purchase. Can I get a replacement disposable card?",
}

# --- TAB 1: Single Ticket Inference ---
with tab_single:
    left_col, right_col = st.columns([6, 6], gap="large")

    with left_col:
        st.markdown(
            """
            <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: var(--stitch-heading); margin-bottom: 4px;">
                Customer Support Ticket Input
            </div>
            <div style="font-size: 13px; color: var(--stitch-muted); margin-bottom: 12px;">
                Simulate incoming omnichannel dialogue. The linear classifier processes sublinear TF-IDF character & word n-grams in real-time.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick validation scenario chips
        st.markdown("<span style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--stitch-muted); letter-spacing: 0.06em;'>Validation Scenarios</span>", unsafe_allow_html=True)
        preset_cols = st.columns(3)
        selected_scenario_text = None
        preset_keys = list(PRESETS.keys())
        for idx, p_key in enumerate(preset_keys):
            col_target = preset_cols[idx % 3]
            with col_target:
                if st.button(p_key, key=f"btn_p_{idx}", use_container_width=True):
                    selected_scenario_text = PRESETS[p_key]

        # Session state management for textarea content
        if "ticket_input_text" not in st.session_state:
            st.session_state.ticket_input_text = PRESETS["Card charged twice"]

        if selected_scenario_text:
            st.session_state.active_ticket_box = selected_scenario_text

        if "active_ticket_box" not in st.session_state:
            st.session_state.active_ticket_box = st.session_state.ticket_input_text

        with st.form("single_ticket_form"):
            ticket_text = st.text_area(
                "Customer Query Utterance:",
                height=130,
                placeholder="Type customer message or select a validation scenario above...",
                key="active_ticket_box",
            )
            classify_clicked = st.form_submit_button(
                "⚡ Classify Ticket", type="primary", use_container_width=True
            )

        char_count = len(ticket_text)
        st.markdown(
            f"<div style='text-align: right; font-size: 11px; color: var(--stitch-muted); font-family: monospace;'>{char_count} characters</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.get("last_inference_ms") is not None:
            st.caption(f"Last measured inference: {st.session_state.last_inference_ms:.1f} ms · CPU")

    with right_col:
        if classify_clicked:
            query_to_process = ticket_text.strip()
            if query_to_process:
                with st.spinner("Classifying ticket..."):
                    t0 = time.perf_counter()
                    st.session_state.last_ticket_result = router.route(query_to_process)
                    st.session_state.last_inference_ms = (time.perf_counter() - t0) * 1000
            else:
                st.session_state.last_ticket_result = None

        result = st.session_state.get("last_ticket_result")
        if result is None:
            st.info("Enter a customer message and select **Classify Ticket** to see its intent and routing recommendation.")
        else:

            conf = result['confidence']
            intent = result['predicted_intent']
            dept = result['department']
            priority = result['priority']

            # Determine Stitch confidence banner
            if conf >= 0.70:
                banner_class = "banner-high"
                banner_text = "High Confidence — Automatic Routing Recommended"
                banner_badge = "AUTO-APPROVED"
                banner_badge_bg = "#004a32"
            elif conf >= 0.40:
                banner_class = "banner-medium"
                banner_text = "Medium Confidence — Human Review Recommended"
                banner_badge = "NEEDS REVIEW"
                banner_badge_bg = "#8a4b08"
            else:
                banner_class = "banner-low"
                banner_text = "Low Confidence — Escalate to Senior Agent"
                banner_badge = "ESCALATED"
                banner_badge_bg = "#ba1a1a"

            # Priority badge class
            p_badge_type = "rose" if priority == "URGENT" else ("amber" if priority == "HIGH" else "emerald")

            st.markdown(
                f"""
                <div class="stitch-result-card">
                    <!-- Status Banner -->
                    <div class="stitch-banner {banner_class}">
                        <span style="font-weight: 600; font-size: 13px;">{banner_text}</span>
                        <span style="font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; background: {banner_badge_bg}; color: white;">
                            {banner_badge}
                        </span>
                    </div>

                    <!-- Primary Classification Result -->
                    <div style="background: var(--stitch-inset); border: 1px solid var(--stitch-border); border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                            <div>
                                <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--stitch-muted);">
                                    PREDICTED INTENT
                                </div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: var(--stitch-heading);">
                                    {intent}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--stitch-muted);">
                                    CONFIDENCE
                                </div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 700; color: var(--stitch-success);">
                                    {conf:.1%}
                                </div>
                            </div>
                        </div>

                        <div style="display: flex; gap: 10px; align-items: center; padding-top: 10px; border-top: 1px solid var(--stitch-border);">
                            <span style="font-size: 12px; color: var(--stitch-secondary);">Target Department: <strong>{dept}</strong></span>
                            <span style="color: var(--stitch-border);">·</span>
                            <span style="font-size: 12px; color: var(--stitch-secondary);">Priority:</span>
                            <span class="stitch-badge badge-{p_badge_type}">{priority}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            st.subheader("Top 5 Candidate Projections")

            top_preds = result.get('top_predictions', [])
            for p in top_preds:
                p_dept = get_department(p['intent'])
                render_confidence_bar(
                    intent=p['intent'],
                    confidence=p['confidence'],
                    department=p_dept,
                )

            with st.expander("🔍 Normalized NLP Feature Token Sequence"):
                st.code(result.get('preprocessed_text', ''), language='text')

# --- TAB 2: Batch CSV Classification ---
with tab_batch:
    st.markdown(
        """
        <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: var(--stitch-heading); margin-bottom: 4px;">
            High-Throughput Batch Ticket Classifier
        </div>
        <div style="font-size: 13px; color: var(--stitch-muted); margin-bottom: 16px;">
            Upload any CSV file containing customer messages under a <code>text</code> column to run bulk inference and automated departmental dispatch.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV Batch File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            batch_df = None
            st.error("Uploaded CSV is empty. Add a 'text' column with at least one ticket query.")

        if batch_df is not None and 'text' not in batch_df.columns:
            st.error("Uploaded CSV must contain a 'text' column header.")
        elif batch_df is not None:
            # Keep missing and non-string CSV cells predictable for inference and export.
            batch_df['text'] = batch_df['text'].fillna('').astype(str)
            st.info(f"Loaded CSV batch containing **{len(batch_df):,}** ticket queries.")
            if st.button("🚀 Process Batch Inference", type="primary"):
                with st.spinner(f"Classifying {len(batch_df)} queries..."):
                    t_start = time.time()
                    batch_results = router.route_batch(batch_df['text'].tolist())
                    b_elapsed = round(time.time() - t_start, 2)

                results_df = pd.DataFrame([
                    {
                        'Customer Text': _safe_csv_text(r['input_text'][:70] + ('...' if len(r['input_text']) > 70 else '')),
                        'Predicted Intent': r['predicted_intent'],
                        'Department': r['department'],
                        'Confidence': f"{r['confidence']:.2%}",
                        'Priority': r['priority'],
                    }
                    for r in batch_results
                ])

                # Batch KPI stats
                k1, k2, k3 = st.columns(3)
                k1.metric("Processed Queries", f"{len(batch_df):,}", f"{b_elapsed}s total")
                k2.metric("Avg Throughput", f"{len(batch_df)/max(b_elapsed, 0.001):.0f} tickets/sec")
                k3.metric("Departments Engaged", len(set(r['department'] for r in batch_results)))

                st.dataframe(results_df, use_container_width=True, height=380, hide_index=True)

                csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Enriched Classification CSV",
                    data=csv_bytes,
                    file_name="supportiq_classified_tickets.csv",
                    mime="text/csv",
                )
