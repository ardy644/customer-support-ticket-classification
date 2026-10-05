"""Live Prediction Page."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from src.config import MODELS_DIR
from src.routing import TicketRouter, get_department, ROUTING_MAP

st.set_page_config(page_title="Predict - BANKING77", page_icon="🔮", layout="wide")
st.title("🔮 Live Ticket Classification")

@st.cache_resource
def load_best_model():
    """Load the best model and vectorizer."""
    import joblib
    model_path = MODELS_DIR / "best_model.joblib"
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    
    if not model_path.exists() or not vec_path.exists():
        return None, None
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

model, vectorizer = load_best_model()

if model is None:
    st.warning("⚠️ No trained model found. Please run the pipeline first.")
    st.stop()

router = TicketRouter(model, vectorizer)

# Priority color mapping
PRIORITY_COLORS = {
    'URGENT': '🔴',
    'HIGH': '🟠',
    'MEDIUM': '🟡',
    'NORMAL': '🟢',
}

st.subheader("Single Ticket Classification")
st.markdown("Enter a customer support query below to classify and route it.")

# Text input
user_input = st.text_area(
    "Customer Query:",
    placeholder="e.g., I haven't received my new card yet, it's been over a week...",
    height=100,
)

# Example queries
with st.expander("💡 Try these example queries"):
    examples = [
        "I lost my card and someone might be using it",
        "Why was I charged twice for the same transaction?",
        "How do I transfer money to another account?",
        "I can't verify my identity, the app keeps rejecting my documents",
        "What exchange rate will I get if I convert USD to EUR?",
        "My contactless payment isn't working anymore",
        "I want to cancel my account",
        "I was charged a fee for my card payment",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}"):
            user_input = ex

if st.button("🔮 Classify & Route", type="primary") and user_input:
    with st.spinner("Classifying..."):
        result = router.route(user_input)
    
    st.markdown("---")
    
    # Main result
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Intent", result['predicted_intent'])
    col2.metric("Department", result['department'])
    col3.metric("Confidence", f"{result['confidence']:.1%}")
    
    # Priority badge
    priority = result['priority']
    st.markdown(f"**Priority:** {PRIORITY_COLORS.get(priority, '⚪')} {priority}")
    
    # Top-5 predictions
    if result['top_predictions']:
        st.subheader("Top 5 Predictions")
        for i, pred in enumerate(result['top_predictions'], 1):
            dept = get_department(pred['intent'])
            st.markdown(
                f"{i}. **{pred['intent']}** ({pred['confidence']:.2%}) → {dept}"
            )
    
    # Preprocessed text
    with st.expander("🔧 Preprocessed Text"):
        st.code(result['preprocessed_text'])

st.markdown("---")

# Batch prediction
st.subheader("Batch Prediction")
st.markdown("Upload a CSV file with a `text` column to classify multiple tickets.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    
    if 'text' not in batch_df.columns:
        st.error("CSV must have a 'text' column.")
    else:
        if st.button("🚀 Classify Batch"):
            with st.spinner(f"Classifying {len(batch_df)} tickets..."):
                results = router.route_batch(batch_df['text'].tolist())
            
            results_df = pd.DataFrame([
                {
                    'text': r['input_text'][:80] + '...' if len(r['input_text']) > 80 else r['input_text'],
                    'intent': r['predicted_intent'],
                    'department': r['department'],
                    'confidence': f"{r['confidence']:.2%}",
                    'priority': r['priority'],
                }
                for r in results
            ])
            
            st.dataframe(results_df, use_container_width=True, height=400)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "predictions.csv",
                "text/csv",
            )
