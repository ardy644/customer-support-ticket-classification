"""Image preprocessing and local OCR extraction pipeline for BANKING77.

Architecture:
Image -> Preprocessing -> Tesseract OCR -> Text -> NLP Pipeline -> Routing
"""
import io
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

from PIL import Image, ImageEnhance
import pytesseract


# Known Tesseract installation locations on Windows
KNOWN_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
]


def find_tesseract_binary() -> Optional[str]:
    """Locate the tesseract executable on Windows or POSIX environments."""
    # 1. Custom environment variable override
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Check PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # 3. Check known Windows installation paths
    for p in KNOWN_TESSERACT_PATHS:
        if os.path.isfile(p):
            return p

    return None


def configure_tesseract():
    """Configure pytesseract binary path if found."""
    found = find_tesseract_binary()
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
    return found


# Initialize configuration on module import
_TESSERACT_BIN = configure_tesseract()


def is_tesseract_installed() -> bool:
    """Check if Tesseract OCR engine is installed and responsive."""
    configure_tesseract()
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def get_tesseract_status() -> Dict[str, Any]:
    """Return diagnostic telemetry for Tesseract OCR availability."""
    bin_path = configure_tesseract()
    try:
        ver = str(pytesseract.get_tesseract_version())
        return {
            "installed": True,
            "path": bin_path or pytesseract.pytesseract.tesseract_cmd,
            "version": ver,
            "error": None,
        }
    except Exception as e:
        return {
            "installed": False,
            "path": bin_path,
            "version": None,
            "error": str(e),
        }


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Preprocess PIL Image to optimize optical character recognition.
    
    Steps:
        1. RGB conversion for transparency/alpha layers
        2. Grayscale conversion
        3. Upscaling small images (width < 800) for OCR resolution clarity
        4. Contrast and sharpness enhancement
    """
    # 1. Handle palette and RGBA formats cleanly with a white background
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Grayscale
    gray = image.convert("L")

    # 3. Upscale if text is low-resolution
    w, h = gray.size
    if w < 800:
        scale_factor = max(1.5, min(2.5, 800.0 / max(w, 1)))
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        gray = gray.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 4. Enhance contrast for sharper letter boundaries
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.8)

    # 5. Sharpen slightly
    sharp = ImageEnhance.Sharpness(enhanced).enhance(1.4)

    return sharp


def load_and_validate_image(image_input: Union[bytes, io.BytesIO, str, Path, Image.Image]) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """Validate and open an image safely.
    
    Returns:
        (PIL.Image, metadata_dict) on success, or (None, error_dict) on failure.
    """
    if image_input is None:
        return None, {
            "valid": False,
            "error_code": "NO_IMAGE",
            "error_message": "No image input was provided.",
        }

    try:
        if isinstance(image_input, Image.Image):
            img = image_input
            img_format = img.format or "PIL"
        elif isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
            img_format = img.format or "FILE"
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
            img_format = img.format or "BYTES"
        elif hasattr(image_input, "read"):
            content = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            if not content:
                return None, {
                    "valid": False,
                    "error_code": "EMPTY_IMAGE_FILE",
                    "error_message": "Uploaded file is empty (0 bytes).",
                }
            img = Image.open(io.BytesIO(content))
            img_format = img.format or "STREAM"
        else:
            return None, {
                "valid": False,
                "error_code": "UNSUPPORTED_TYPE",
                "error_message": f"Unsupported image input type: {type(image_input).__name__}",
            }

        # PIL images without their original file metadata are used for generated
        # in-memory samples; upload formats are constrained by the UI.
        if img_format not in {"PNG", "JPEG", "PIL"}:
            return None, {
                "valid": False,
                "error_code": "UNSUPPORTED_FORMAT",
                "error_message": "Unsupported image format. Please use a PNG or JPEG image.",
            }

        # Verify image integrity
        img.verify()

        # Reopen after verify() (Pillow requirement: verify closes or invalidates the stream)
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, "seek"):
            image_input.seek(0)
            img = Image.open(io.BytesIO(image_input.read()))
            image_input.seek(0)
        else:
            img = img.copy()

        return img, {
            "valid": True,
            "format": img_format,
            "size": img.size,
            "mode": img.mode,
        }

    except Exception as e:
        return None, {
            "valid": False,
            "error_code": "CORRUPT_OR_UNSUPPORTED_IMAGE",
            "error_message": f"Unable to read image file: {str(e)}",
        }


def extract_text_from_image(
    image_input: Union[bytes, io.BytesIO, str, Path, Image.Image],
    preprocess: bool = True,
    tesseract_config: str = "--oem 3 --psm 6",
) -> Dict[str, Any]:
    """Extract textual ticket content from an image via local Tesseract OCR.
    
    Returns:
        dict containing:
            success: bool
            raw_text: str
            cleaned_text: str
            ocr_confidence: float (0.0 to 1.0) or None
            word_count: int
            char_count: int
            image_format: str
            image_size: tuple
            error_code: str or None
            error_message: str or None
    """
    img, meta = load_and_validate_image(image_input)
    if not meta.get("valid"):
        return {
            "success": False,
            "raw_text": "",
            "cleaned_text": "",
            "ocr_confidence": None,
            "word_count": 0,
            "char_count": 0,
            "image_format": "UNKNOWN",
            "image_size": (0, 0),
            "error_code": meta.get("error_code"),
            "error_message": meta.get("error_message"),
        }

    # Verify Tesseract is available
    if not is_tesseract_installed():
        return {
            "success": False,
            "raw_text": "",
            "cleaned_text": "",
            "ocr_confidence": None,
            "word_count": 0,
            "char_count": 0,
            "image_format": meta.get("format", "UNKNOWN"),
            "image_size": meta.get("size", (0, 0)),
            "error_code": "OCR_ENGINE_UNAVAILABLE",
            "error_message": (
                "Tesseract OCR executable was not found on this system. "
                "Please install Tesseract OCR (e.g. via 'winget install UB-Mannheim.TesseractOCR') "
                "or configure the TESSERACT_CMD environment variable."
            ),
        }

    try:
        # Preprocessing
        target_img = preprocess_image_for_ocr(img) if preprocess else img

        # Extract text
        raw_text = pytesseract.image_to_string(target_img, config=tesseract_config)
        cleaned_text = raw_text.strip() if raw_text else ""

        # Extract OCR confidence metrics
        ocr_confidence: Optional[float] = None
        try:
            data = pytesseract.image_to_data(target_img, output_type=pytesseract.Output.DICT, config=tesseract_config)
            confs = [float(c) for c in data.get("conf", []) if float(c) > 0]
            if confs:
                ocr_confidence = round(float(sum(confs) / len(confs)) / 100.0, 4)
        except Exception:
            ocr_confidence = None

        if not cleaned_text:
            return {
                "success": False,
                "raw_text": raw_text or "",
                "cleaned_text": "",
                "ocr_confidence": 0.0,
                "word_count": 0,
                "char_count": 0,
                "image_format": meta.get("format", "UNKNOWN"),
                "image_size": meta.get("size", (0, 0)),
                "error_code": "EMPTY_OCR_TEXT",
                "error_message": "No readable text was detected in this image.",
            }

        return {
            "success": True,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "ocr_confidence": ocr_confidence,
            "word_count": len(cleaned_text.split()),
            "char_count": len(cleaned_text),
            "image_format": meta.get("format", "UNKNOWN"),
            "image_size": meta.get("size", (0, 0)),
            "error_code": None,
            "error_message": None,
        }

    except Exception as e:
        return {
            "success": False,
            "raw_text": "",
            "cleaned_text": "",
            "ocr_confidence": None,
            "word_count": 0,
            "char_count": 0,
            "image_format": meta.get("format", "UNKNOWN"),
            "image_size": meta.get("size", (0, 0)),
            "error_code": "OCR_EXCEPTION",
            "error_message": f"OCR processing failure: {str(e)}",
        }


def classify_image_ticket(
    image_input: Union[bytes, io.BytesIO, str, Path, Image.Image],
    router: Any,
    preprocess_image: bool = True,
) -> Dict[str, Any]:
    """End-to-end pipeline: Image -> OCR -> Extracted Text -> ML Classifier -> Support Routing.
    
    Reuses the existing TicketRouter for intent classification, department dispatch,
    and priority scoring without altering model behavior.
    """
    ocr_result = extract_text_from_image(image_input, preprocess=preprocess_image)

    if not ocr_result["success"]:
        return {
            "success": False,
            "ocr": ocr_result,
            "classification": None,
            "error_code": ocr_result["error_code"],
            "error_message": ocr_result["error_message"],
        }

    # Pass extracted text directly into existing router
    text_to_classify = ocr_result["cleaned_text"]
    try:
        route_result = router.route(text_to_classify)
        return {
            "success": True,
            "ocr": ocr_result,
            "classification": route_result,
            "extracted_text": text_to_classify,
            "ocr_confidence": ocr_result.get("ocr_confidence"),
            "model_confidence": route_result.get("confidence"),
            "predicted_intent": route_result.get("predicted_intent"),
            "department": route_result.get("department"),
            "priority": route_result.get("priority"),
            "top_predictions": route_result.get("top_predictions", []),
            "preprocessed_text": route_result.get("preprocessed_text", ""),
            "error_code": None,
            "error_message": None,
        }
    except Exception as e:
        return {
            "success": False,
            "ocr": ocr_result,
            "classification": None,
            "error_code": "CLASSIFICATION_FAILURE",
            "error_message": f"ML ticket classification error: {str(e)}",
        }
