"""
Pytest configuration and fixtures for TissUUmaps tests.
"""

import os
import tempfile

import pytest

# Check if required native libraries are available
try:
    import pyvips

    PYVIPS_AVAILABLE = True
except (ImportError, OSError):
    PYVIPS_AVAILABLE = False

try:
    import openslide

    OPENSLIDE_AVAILABLE = True
except (ImportError, OSError):
    OPENSLIDE_AVAILABLE = False

# Whether tissuumaps can be imported (requires pyvips and openslide)
TISSUUMAPS_AVAILABLE = PYVIPS_AVAILABLE and OPENSLIDE_AVAILABLE


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "requires_tissuumaps: tests that require tissuumaps to be importable"
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def app_client(temp_dir):
    """Create a Flask test client with a temporary slide directory."""
    if not TISSUUMAPS_AVAILABLE:
        pytest.skip("TissUUmaps not available (missing pyvips or openslide)")

    # Set environment before importing tissuumaps
    os.environ["SLIDE_DIR"] = temp_dir

    from tissuumaps import app

    app.config["TESTING"] = True
    app.config["SLIDE_DIR"] = temp_dir
    app.basedir = temp_dir

    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_tmap_project():
    """Return a sample TissUUmaps project structure."""
    return {
        "layers": [{"name": "test_layer", "tileSource": "test.dzi"}],
        "markerFiles": [],
        "regionFiles": [],
    }
