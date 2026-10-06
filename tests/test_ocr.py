"""Comprehensive unit tests for image OCR extraction and fallback classification.

Tests mock pytesseract to ensure test stability regardless of host Tesseract binary state.
"""
import io
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image, ImageDraw

from src.ocr_pipeline import (
    preprocess_image_for_ocr,
    load_and_validate_image,
    extract_text_from_image,
    classify_image_ticket,
    is_tesseract_installed,
)
from src.routing import TicketRouter, ROUTING_MAP, get_department, get_priority


@pytest.fixture
def sample_png_bytes():
    """Create an in-memory valid PNG image."""
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "My card was lost and stolen", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def sample_jpeg_bytes():
    """Create an in-memory valid JPEG image."""
    img = Image.new("RGB", (500, 150), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Transfer money to account", fill=(10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def mock_router():
    """Mock TicketRouter returning a valid BANKING77 routing structure."""
    router = MagicMock(spec=TicketRouter)
    router.route.return_value = {
        "input_text": "My card was lost and stolen",
        "preprocessed_text": "card lost stolen",
        "predicted_intent": "lost_or_stolen_card",
        "department": "Card Security",
        "confidence": 0.9412,
        "priority": "URGENT",
        "top_predictions": [
            {"intent": "lost_or_stolen_card", "confidence": 0.9412},
            {"intent": "compromised_card", "confidence": 0.0350},
        ],
    }
    return router


# 1. Valid image handling
def test_valid_image_handling(sample_png_bytes):
    img, meta = load_and_validate_image(sample_png_bytes)
    assert img is not None
    assert meta["valid"] is True
    assert meta["format"] in ("PNG", "BYTES")


# 2. PNG handling
def test_png_handling(sample_png_bytes):
    img, meta = load_and_validate_image(sample_png_bytes)
    assert img is not None
    preprocessed = preprocess_image_for_ocr(img)
    assert isinstance(preprocessed, Image.Image)
    assert preprocessed.mode == "L"  # Grayscale


# 3. JPEG handling
def test_jpeg_handling(sample_jpeg_bytes):
    img, meta = load_and_validate_image(sample_jpeg_bytes)
    assert img is not None
    preprocessed = preprocess_image_for_ocr(img)
    assert isinstance(preprocessed, Image.Image)
    assert preprocessed.mode == "L"


# 4. Invalid image handling (corrupt bytes)
def test_invalid_image_handling():
    corrupt_data = b"NOT_A_REAL_IMAGE_HEADER_12345"
    img, meta = load_and_validate_image(corrupt_data)
    assert img is None
    assert meta["valid"] is False
    assert meta["error_code"] == "CORRUPT_OR_UNSUPPORTED_IMAGE"


# 5. Unsupported file handling (None / empty / wrong type)
def test_unsupported_file_handling():
    img_none, meta_none = load_and_validate_image(None)
    assert img_none is None
    assert meta_none["error_code"] == "NO_IMAGE"

    img_empty, meta_empty = load_and_validate_image(b"")
    assert img_empty is None
    assert meta_empty["error_code"] in ("CORRUPT_OR_UNSUPPORTED_IMAGE", "EMPTY_IMAGE_FILE")


# 6. OCR extraction with mocked Tesseract
@patch("src.ocr_pipeline.is_tesseract_installed", return_value=True)
@patch("pytesseract.image_to_string", return_value="Why was I charged a card payment fee?")
@patch("pytesseract.image_to_data", return_value={"conf": ["92", "88", "95"]})
def test_ocr_extraction_success(mock_data, mock_string, mock_installed, sample_png_bytes):
    result = extract_text_from_image(sample_png_bytes)
    assert result["success"] is True
    assert "charged a card payment fee" in result["cleaned_text"]
    assert result["ocr_confidence"] is not None
    assert result["ocr_confidence"] > 0.8
    assert result["word_count"] > 0


# 7. Empty OCR output
@patch("src.ocr_pipeline.is_tesseract_installed", return_value=True)
@patch("pytesseract.image_to_string", return_value="   \n\t  ")
def test_empty_ocr_output(mock_string, mock_installed, sample_png_bytes):
    result = extract_text_from_image(sample_png_bytes)
    assert result["success"] is False
    assert result["error_code"] == "EMPTY_OCR_TEXT"
    assert "No readable text was detected" in result["error_message"]


# 8. OCR exception handling
@patch("src.ocr_pipeline.is_tesseract_installed", return_value=True)
@patch("pytesseract.image_to_string", side_effect=RuntimeError("Tesseract engine crashed"))
def test_ocr_exception_handling(mock_string, mock_installed, sample_png_bytes):
    result = extract_text_from_image(sample_png_bytes)
    assert result["success"] is False
    assert result["error_code"] == "OCR_EXCEPTION"
    assert "Tesseract engine crashed" in result["error_message"]


# 9. OCR engine unavailable handling
@patch("src.ocr_pipeline.is_tesseract_installed", return_value=False)
def test_ocr_engine_unavailable(mock_installed, sample_png_bytes):
    result = extract_text_from_image(sample_png_bytes)
    assert result["success"] is False
    assert result["error_code"] == "OCR_ENGINE_UNAVAILABLE"
    assert "Tesseract OCR executable was not found" in result["error_message"]


# 10. OCR output reaching classifier & returning complete prediction
@patch("src.ocr_pipeline.is_tesseract_installed", return_value=True)
@patch("pytesseract.image_to_string", return_value="My card was lost and stolen")
@patch("pytesseract.image_to_data", return_value={"conf": ["96", "94", "92"]})
def test_classify_image_ticket_flow(mock_data, mock_string, mock_installed, sample_png_bytes, mock_router):
    result = classify_image_ticket(sample_png_bytes, router=mock_router)
    assert result["success"] is True
    assert mock_router.route.called
    assert mock_router.route.call_args[0][0] == "My card was lost and stolen"

    # Verify essential fields
    assert result["predicted_intent"] == "lost_or_stolen_card"
    assert result["model_confidence"] == 0.9412
    assert result["department"] == "Card Security"
    assert result["priority"] == "URGENT"
    assert result["ocr_confidence"] is not None
    assert len(result["top_predictions"]) >= 1


# 11. Existing text classifier still works
def test_existing_text_classifier_contract():
    # Test that existing router API contract is intact
    from src.routing import TicketRouter
    assert hasattr(TicketRouter, "route")
    assert hasattr(TicketRouter, "route_batch")


# 12. Existing routing logic and maps still work
def test_existing_routing_still_works():
    assert get_department("lost_or_stolen_card") == "Card Security"
    assert get_department("card_arrival") == "Card Services"
    assert get_department("exchange_rate") == "Foreign Exchange"
    assert get_priority("lost_or_stolen_card", confidence=0.9) in ("URGENT", "HIGH")
    assert get_priority("card_arrival", confidence=0.9) == "NORMAL"
    assert len(ROUTING_MAP) == 10
