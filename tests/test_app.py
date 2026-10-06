"""Automated tests for Streamlit application pages using AppTest."""
import sys
import io
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest


def test_main_app_page():
    """Verify main app page renders without error."""
    at = AppTest.from_file(str(project_root / "app" / "streamlit_app.py"))
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_dataset_analytics_page():
    """Verify Dataset Analytics page renders without error."""
    at = AppTest.from_file(str(project_root / "app" / "pages" / "1_EDA.py"))
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_model_comparison_page():
    """Verify Model Comparison page renders with metrics."""
    at = AppTest.from_file(str(project_root / "app" / "pages" / "2_Model_Comparison.py"))
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_predict_page():
    """Verify Live Prediction page renders."""
    at = AppTest.from_file(str(project_root / "app" / "pages" / "3_Predict.py"))
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_error_analysis_page():
    """Verify Error Analysis page renders."""
    at = AppTest.from_file(str(project_root / "app" / "pages" / "4_Error_Analysis.py"))
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_routing_page():
    """Verify Ticket Routing page renders."""
    at = AppTest.from_file(str(project_root / "app" / "pages" / "5_Routing.py"))
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_image_ticket_classifier_page():
    """Verify Image Ticket Classifier page renders without error."""
    at = AppTest.from_file(str(project_root / "app" / "pages" / "6_Image_Ticket_Classifier.py"))
    at.run(timeout=30)
    assert len(at.exception) == 0


def test_image_sample_mode_populates_text_without_fake_ocr_confidence(monkeypatch):
    """Sample mode demonstrates routing without pretending OCR ran."""
    from src import ocr_pipeline

    monkeypatch.setattr(
        ocr_pipeline,
        "get_tesseract_status",
        lambda: {"installed": False, "path": None, "version": None, "error": "not installed"},
    )
    at = AppTest.from_file(str(project_root / "app" / "pages" / "6_Image_Ticket_Classifier.py"))
    at.run(timeout=30)

    at.button[0].click().run(timeout=30)
    extract_button = next(button for button in at.button if "Extract Text" in button.label)
    extract_button.click().run(timeout=30)

    assert len(at.exception) == 0
    assert "charged twice" in at.text_area(key="ocr_extracted_textbox").value
    assert at.session_state["ocr_confidence_score"] is None
    assert any("OCR not run" in str(element.value) for element in at.markdown)

    classify_button = next(button for button in at.button if "Classify Image Ticket" in button.label)
    classify_button.click().run(timeout=30)
    assert len(at.exception) == 0
    assert at.session_state["ocr_classification_result"]["predicted_intent"] == "transaction_charged_twice"


def test_uploaded_image_text_survives_extract_and_classify_reruns(monkeypatch):
    """An uploaded image remains selected through OCR and classifier reruns."""
    from src import ocr_pipeline

    monkeypatch.setattr(
        ocr_pipeline,
        "get_tesseract_status",
        lambda: {"installed": True, "path": "tesseract", "version": "test", "error": None},
    )
    monkeypatch.setattr(
        ocr_pipeline,
        "extract_text_from_image",
        lambda *args, **kwargs: {
            "success": True,
            "cleaned_text": "I was charged twice for this purchase",
            "ocr_confidence": 0.83,
            "error_message": None,
        },
    )
    image_bytes = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(image_bytes, format="PNG")

    at = AppTest.from_file(str(project_root / "app" / "pages" / "6_Image_Ticket_Classifier.py"))
    at.run(timeout=30)
    at.file_uploader[0].set_value(("ticket.png", image_bytes.getvalue(), "image/png")).run(timeout=30)
    next(button for button in at.button if "Extract Text" in button.label).click().run(timeout=30)
    assert at.text_area(key="ocr_extracted_textbox").value == "I was charged twice for this purchase"

    next(button for button in at.button if "Classify Image Ticket" in button.label).click().run(timeout=30)
    assert len(at.exception) == 0
    assert at.text_area(key="ocr_extracted_textbox").value == "I was charged twice for this purchase"
    assert at.session_state["ocr_classification_result"]["predicted_intent"] == "transaction_charged_twice"
    assert at.session_state["ocr_confidence_score"] == 0.83
