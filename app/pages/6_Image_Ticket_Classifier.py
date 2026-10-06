"""SupportIQ - Image Ticket Classifier Page.

Extracts customer support ticket text from screenshots and images via local OCR,
then classifies intents and routes departments using the production ML pipeline.
Matches Stitch Design System specifications.
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from PIL import Image, ImageDraw
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
from src.ocr_pipeline import (
    load_and_validate_image,
    extract_text_from_image,
    preprocess_image_for_ocr,
    is_tesseract_installed,
    get_tesseract_status,
)

st.set_page_config(
    page_title="SupportIQ - Image Ticket Classifier",
    page_icon="📷",
    layout="wide",
)

apply_stitch_styles()
render_sidebar_branding()
render_sidebar_footer()

render_page_header(
    title="Image Ticket Classifier",
    subtitle="Upload a customer-support screenshot or image. The system uses OCR to extract the ticket text and then classifies and routes it using the existing machine-learning pipeline.",
    tag="FALLBACK PIPELINE · IMAGE OCR → NLP ML",
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


def generate_synthetic_support_image(text: str, title: str = "Banking App Support Ticket") -> Image.Image:
    """Generate a clean synthetic customer support ticket screenshot for testing."""
    width, height = 700, 260
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (width, 50)], fill=(0, 35, 111))
    draw.text((20, 16), title, fill=(255, 255, 255))

    # Border
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline=(226, 232, 240), width=2)

    # Message bubble
    draw.rectangle([(20, 70), (width - 20, height - 25)], fill=(239, 244, 255), outline=(220, 233, 255), width=1)
    draw.text((35, 82), "CUSTOMER MESSAGE (Incoming)", fill=(100, 116, 139))

    # Text wrapping
    words = text.split()
    lines = []
    curr = []
    for w in words:
        curr.append(w)
        if len(" ".join(curr)) > 60:
            lines.append(" ".join(curr[:-1]))
            curr = [w]
    if curr:
        lines.append(" ".join(curr))

    y = 110
    for line in lines[:4]:
        draw.text((35, y), line, fill=(11, 28, 48))
        y += 28

    return img


# Demo scenario library
DEMO_SCENARIOS = {
    "Double Billing Screenshot": (
        "I was charged twice for the same transaction of £45.20 at Sainsbury's this morning. Please refund the duplicate payment.",
        "Mobile Banking - Dispute Transaction"
    ),
    "Pending Wire Transfer Receipt": (
        "I initiated a bank transfer to my landlord yesterday morning but the status is still pending and money has not arrived.",
        "Online Banking - Transfer Details"
    ),
    "Stolen Card Emergency Alert": (
        "My card was stolen from my wallet on the train and someone is making unauthorized contactless payments.",
        "Security Center - Urgent Report"
    ),
    "Failed ATM Withdrawal": (
        "I tried to withdraw £60 from an ATM on Oxford Street. No cash came out but my account balance was still deducted.",
        "ATM Support - Cash Withdrawal"
    ),
}

# Tesseract status diagnostics
tess_status = get_tesseract_status()
tess_available = tess_status["installed"]

if not tess_available:
    st.markdown(
        """
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #92400e; font-size: 13px; margin-bottom: 4px;">
                <span>⚠️ Local Tesseract OCR Engine Notice</span>
            </div>
            <div style="font-size: 12px; color: #78350f; line-height: 1.5;">
                The <code>pytesseract</code> Python package is ready, but the system <strong>Tesseract OCR</strong> binary was not detected on Windows PATH or default directories.
                <br>• To enable live optical scanning from your custom uploaded images, install Tesseract OCR: <code>winget install UB-Mannheim.TesseractOCR</code>
                <br>• Built-in sample scenarios can demonstrate classification and routing from their known sample text. They do not run OCR and do not report OCR confidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Session state initialization
if "ocr_active_image" not in st.session_state:
    st.session_state.ocr_active_image = None
if "ocr_extracted_text" not in st.session_state:
    st.session_state.ocr_extracted_text = ""
if "ocr_confidence_score" not in st.session_state:
    st.session_state.ocr_confidence_score = None
if "ocr_classification_result" not in st.session_state:
    st.session_state.ocr_classification_result = None
if "ocr_error_message" not in st.session_state:
    st.session_state.ocr_error_message = None

# Top Section: Two-Column Input & Preview
input_col, preview_col = st.columns([6, 6], gap="large")

with input_col:
    st.markdown(
        """
        <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: #00236f; margin-bottom: 4px;">
            1. Image Input & Preprocessing
        </div>
        <div style="font-size: 13px; color: #64748b; margin-bottom: 14px;">
            Upload an image (PNG, JPG, JPEG) or select an authentic banking test scenario.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset Demo Scenario Selector
    st.markdown("<span style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.06em;'>Quick Test Scenarios (Synthesized Screenshots)</span>", unsafe_allow_html=True)
    scenario_cols = st.columns(2)
    s_keys = list(DEMO_SCENARIOS.keys())
    for idx, s_key in enumerate(s_keys):
        c_target = scenario_cols[idx % 2]
        with c_target:
            if st.button(s_key, key=f"btn_demo_{idx}", use_container_width=True):
                s_text, s_title = DEMO_SCENARIOS[s_key]
                synth_img = generate_synthetic_support_image(s_text, s_title)
                st.session_state.ocr_active_image = synth_img
                # Clear previous downstream states
                st.session_state.ocr_extracted_text = ""
                st.session_state.ocr_extracted_textbox = ""
                st.session_state.ocr_confidence_score = None
                st.session_state.ocr_classification_result = None
                st.session_state.ocr_error_message = None
                st.session_state.preset_sample_text = s_text

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Ticket Image:",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG. Max file size: 10MB.",
    )

    uploaded_file_id = getattr(uploaded_file, "file_id", None) if uploaded_file is not None else None
    if uploaded_file is not None and uploaded_file_id != st.session_state.get("ocr_uploaded_file_id"):
        st.session_state.ocr_uploaded_file_id = uploaded_file_id
        try:
            pil_img, image_meta = load_and_validate_image(uploaded_file)
            if pil_img is None:
                st.session_state.ocr_active_image = None
                st.session_state.preset_sample_text = None
                st.session_state.ocr_extracted_text = ""
                st.session_state.ocr_extracted_textbox = ""
                st.session_state.ocr_confidence_score = None
                st.session_state.ocr_classification_result = None
                st.error(image_meta.get("error_message", "Unsupported or invalid image file."))
            else:
                st.session_state.ocr_active_image = pil_img.copy()
                st.session_state.ocr_extracted_text = ""
                st.session_state.ocr_extracted_textbox = ""
                st.session_state.ocr_confidence_score = None
                st.session_state.ocr_classification_result = None
                # Clear preset text so OCR runs on uploaded file
                st.session_state.preset_sample_text = None
        except Exception as e:
            st.session_state.ocr_active_image = None
            st.session_state.preset_sample_text = None
            st.session_state.ocr_extracted_text = ""
            st.session_state.ocr_extracted_textbox = ""
            st.session_state.ocr_confidence_score = None
            st.session_state.ocr_classification_result = None
            st.error(f"Unable to read image: {str(e)}")
    elif uploaded_file is None and st.session_state.get("ocr_uploaded_file_id") is not None:
        st.session_state.ocr_uploaded_file_id = None
        if not getattr(st.session_state, "preset_sample_text", None):
            st.session_state.ocr_active_image = None
            st.session_state.ocr_extracted_text = ""
            st.session_state.ocr_extracted_textbox = ""
            st.session_state.ocr_confidence_score = None
            st.session_state.ocr_classification_result = None

    # Preprocessing options
    enable_enhancement = st.checkbox(
        "Apply Grayscale & Contrast Enhancement before OCR",
        value=True,
        help="Converts to grayscale, scales resolution if below 800px width, and enhances contrast.",
    )

    # OCR Action Button
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    extract_btn = st.button("🔍 Extract Text (OCR)", type="primary", use_container_width=True)

    if extract_btn:
        st.session_state.ocr_error_message = None
        st.session_state.ocr_classification_result = None

        if st.session_state.ocr_active_image is None:
            st.warning("Please upload an image or select a sample scenario first.")
        else:
            with st.spinner("Processing image and running optical character recognition..."):
                active_img = st.session_state.ocr_active_image

                if tess_available:
                    ocr_res = extract_text_from_image(active_img, preprocess=enable_enhancement)
                    if ocr_res["success"]:
                        st.session_state.ocr_extracted_text = ocr_res["cleaned_text"]
                        st.session_state.ocr_extracted_textbox = ocr_res["cleaned_text"]
                        st.session_state.ocr_confidence_score = ocr_res.get("ocr_confidence")
                        st.session_state.ocr_error_message = None
                    else:
                        st.session_state.ocr_extracted_text = ""
                        st.session_state.ocr_extracted_textbox = ""
                        st.session_state.ocr_confidence_score = None
                        st.session_state.ocr_error_message = ocr_res.get("error_message", "OCR processing failed.")
                else:
                    # Fallback mode when Tesseract binary is not installed
                    preset_text = getattr(st.session_state, "preset_sample_text", None)
                    if preset_text:
                        st.session_state.ocr_extracted_text = preset_text
                        st.session_state.ocr_extracted_textbox = preset_text
                        st.session_state.ocr_confidence_score = None
                        st.session_state.ocr_error_message = None
                    else:
                        st.session_state.ocr_extracted_text = ""
                        st.session_state.ocr_extracted_textbox = ""
                        st.session_state.ocr_confidence_score = None
                        st.session_state.ocr_error_message = (
                            "Tesseract OCR executable was not found on this system. "
                            "Please install Tesseract OCR (`winget install UB-Mannheim.TesseractOCR`) "
                            "or select one of the Quick Test Scenarios above to test the fallback flow."
                        )

with preview_col:
    st.markdown(
        """
        <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: #00236f; margin-bottom: 4px;">
            Image Preview & Inspection
        </div>
        <div style="font-size: 13px; color: #64748b; margin-bottom: 14px;">
            Preserves authentic aspect ratio without distortion.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.ocr_active_image is not None:
        preview_img = st.session_state.ocr_active_image
        st.image(preview_img, use_container_width=True, caption=f"Input Image ({preview_img.size[0]}×{preview_img.size[1]} px, mode: {preview_img.mode})")

        if enable_enhancement:
            with st.expander("👁️ View Preprocessed OCR Target Image"):
                enhanced_preview = preprocess_image_for_ocr(preview_img)
                st.image(enhanced_preview, use_container_width=True, caption="Grayscale & Contrast-Enhanced for OCR Engine")
    else:
        st.markdown(
            """
            <div style="border: 2px dashed #cbd5e1; border-radius: 12px; height: 260px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #94a3b8; background: #ffffff;">
                <div style="font-size: 32px; margin-bottom: 8px;">🖼️</div>
                <div style="font-size: 14px; font-weight: 600; color: #475569;">No image loaded</div>
                <div style="font-size: 12px;">Upload a file or choose a sample scenario on the left</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Middle Section: OCR Extracted Text & Classification Trigger
st.markdown("---")
st.subheader("2. OCR Text Extraction Result")

if st.session_state.ocr_error_message:
    st.error(f"❌ {st.session_state.ocr_error_message}")

text_col_l, text_col_r = st.columns([8, 4], gap="large")

with text_col_l:
    if not tess_available and getattr(st.session_state, "preset_sample_text", None):
        text_source_label = "Sample ticket text (known preset content; OCR not run)"
    else:
        text_source_label = "OCR Extracted Text (Output from OCR Engine)"
    st.markdown(
        f"<span style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.06em;'>{text_source_label}</span>",
        unsafe_allow_html=True,
    )
    
    editable_ocr_text = st.text_area(
        "Extracted Ticket Text:",
        value=st.session_state.ocr_extracted_text,
        height=110,
        placeholder="OCR-extracted text will appear here once you click 'Extract Text'...",
        help="This text was generated by the OCR engine, not manual input. You may edit or refine it before classification if needed.",
        key="ocr_extracted_textbox",
    )
    st.session_state.ocr_extracted_text = editable_ocr_text

with text_col_r:
    st.markdown("<span style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.06em;'>OCR Telemetry</span>", unsafe_allow_html=True)

    ocr_c = st.session_state.ocr_confidence_score
    ocr_c_str = f"{ocr_c:.1%}" if ocr_c is not None else "N/A"
    chars_count = len(st.session_state.ocr_extracted_text)
    words_count = len(st.session_state.ocr_extracted_text.split())

    st.markdown(
        f"""
        <div class="stitch-kpi-card" style="padding: 14px 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 11px; color: #64748b; font-weight: 600;">OCR Confidence</span>
                <span class="stitch-badge badge-blue" style="font-family: monospace;">{ocr_c_str}</span>
            </div>
            <div style="font-size: 12px; color: #0b1c30; margin-bottom: 4px;">
                Extracted Words: <strong>{words_count}</strong> ({chars_count} characters)
            </div>
            <div style="font-size: 11px; color: #64748b;">
                Engine: <strong>Local Tesseract v5</strong> (CPU)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    classify_img_btn = st.button("🚀 Classify Image Ticket", type="primary", use_container_width=True)

# Run classification on explicit action
if classify_img_btn:
    raw_ocr = st.session_state.ocr_extracted_text.strip()
    if not raw_ocr:
        st.error("Cannot classify: No readable text was detected in this image. Please extract text first or upload a clearer image.")
    else:
        with st.spinner("Classifying OCR-extracted ticket text with ML pipeline..."):
            t0 = time.time()
            res = router.route(raw_ocr)
            st.session_state.ocr_classification_result = res

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Bottom Section: Classification & Routing Results
if st.session_state.ocr_classification_result is not None:
    res = st.session_state.ocr_classification_result

    st.markdown("---")
    st.subheader("3. Classification & Support Routing Results")

    intent = res["predicted_intent"]
    dept = res["department"]
    model_conf = res["confidence"]
    priority = res["priority"]
    top_preds = res.get("top_predictions", [])

    # Status banner
    if model_conf >= 0.70:
        banner_class = "banner-high"
        banner_text = "High Confidence — Automatic Routing Recommended"
        banner_badge = "AUTO-APPROVED"
        banner_badge_bg = "#004a32"
    elif model_conf >= 0.40:
        banner_class = "banner-medium"
        banner_text = "Medium Confidence — Human Review Recommended"
        banner_badge = "NEEDS REVIEW"
        banner_badge_bg = "#d97706"
    else:
        banner_class = "banner-low"
        banner_text = "Low Confidence — Escalate to Senior Agent"
        banner_badge = "ESCALATED"
        banner_badge_bg = "#ba1a1a"

    p_badge_type = "rose" if priority == "URGENT" else ("amber" if priority == "HIGH" else "emerald")

    res_l, res_r = st.columns([6, 6], gap="large")

    with res_l:
        st.markdown(
            f"""
            <div class="stitch-result-card">
                <div class="stitch-banner {banner_class}">
                    <span style="font-weight: 600; font-size: 13px;">{banner_text}</span>
                    <span style="font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; background: {banner_badge_bg}; color: white;">
                        {banner_badge}
                    </span>
                </div>
                
                <div style="background: #f8f9ff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b;">
                                PREDICTED BANKING INTENT
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; color: #00236f;">
                                {intent}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b;">
                                MODEL CONFIDENCE
                            </div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 700; color: #059669;">
                                {model_conf:.1%}
                            </div>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 12px; align-items: center; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 13px;">
                        <span>Target Department: <strong>{dept}</strong></span>
                        <span style="color: #cbd5e1;">·</span>
                        <span>Priority:</span>
                        <span class="stitch-badge badge-{p_badge_type}">{priority}</span>
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; padding: 4px 6px;">
                    <span>OCR Confidence: <strong style="color: #0b1c30;">{ocr_c_str}</strong></span>
                    <span>Model Confidence: <strong style="color: #0b1c30;">{model_conf:.1%}</strong></span>
                    <span style="font-style: italic;">(Scores are kept strictly separate)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_r:
        st.markdown(
            """
            <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 15px; font-weight: 700; color: #00236f; margin-bottom: 10px;">
                Top 5 Candidate Intent Projections
            </div>
            """,
            unsafe_allow_html=True,
        )

        for p in top_preds:
            p_dept = get_department(p["intent"])
            render_confidence_bar(
                intent=p["intent"],
                confidence=p["confidence"],
                department=p_dept,
            )

        with st.expander("🔍 Normalized NLP Tokens (from OCR text)"):
            st.code(res.get("preprocessed_text", ""), language="text")
