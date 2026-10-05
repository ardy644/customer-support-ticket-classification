"""Automated tests for Streamlit application pages using AppTest."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
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
