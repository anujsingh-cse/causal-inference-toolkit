"""
Unit tests for Streamlit App module.
"""

from causal_toolkit.demo.app import STREAMLIT_AVAILABLE, create_app


def test_app_import_and_structure():
    """Verify create_app runs without throwing errors when executed headless."""
    assert isinstance(STREAMLIT_AVAILABLE, bool)
    # When streamlit is not present, create_app prints warning and returns safely
    if not STREAMLIT_AVAILABLE:
        create_app()
