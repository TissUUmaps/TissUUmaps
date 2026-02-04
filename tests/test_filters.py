"""
Tests for file and directory filter functions in views.py.
"""

import os
import tempfile

import pytest

from tests.conftest import TISSUUMAPS_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TISSUUMAPS_AVAILABLE, reason="TissUUmaps not available (missing native libs)"
)


class TestFileFilter:
    """Tests for the _fnfilter function."""

    def test_fnfilter_tmap_file(self, temp_dir):
        """Test that .tmap files are accepted."""
        from tissuumaps.views import _fnfilter

        tmap_file = os.path.join(temp_dir, "test.tmap")
        with open(tmap_file, "w") as f:
            f.write("{}")
        assert _fnfilter(tmap_file) is True

    def test_fnfilter_dzi_file(self, temp_dir):
        """Test that .dzi files are accepted."""
        from tissuumaps.views import _fnfilter

        dzi_file = os.path.join(temp_dir, "test.dzi")
        with open(dzi_file, "w") as f:
            f.write('<?xml version="1.0"?>')
        assert _fnfilter(dzi_file) is True

    def test_fnfilter_h5ad_file(self, temp_dir):
        """Test that .h5ad files are accepted."""
        from tissuumaps.views import _fnfilter

        h5ad_file = os.path.join(temp_dir, "test.h5ad")
        with open(h5ad_file, "w") as f:
            f.write("dummy")
        assert _fnfilter(h5ad_file) is True

    def test_fnfilter_qptiff_file(self, temp_dir):
        """Test that .qptiff files are accepted."""
        from tissuumaps.views import _fnfilter

        qptiff_file = os.path.join(temp_dir, "test.qptiff")
        with open(qptiff_file, "w") as f:
            f.write("dummy")
        assert _fnfilter(qptiff_file) is True

    def test_fnfilter_unsupported_file(self, temp_dir):
        """Test that unsupported files are rejected."""
        from tissuumaps.views import _fnfilter

        txt_file = os.path.join(temp_dir, "test.txt")
        with open(txt_file, "w") as f:
            f.write("dummy")
        assert _fnfilter(txt_file) is False


class TestDirectoryFilter:
    """Tests for the _dfilter function."""

    def test_dfilter_private_directory(self):
        """Test that directories containing 'private' are filtered out."""
        from tissuumaps.views import _dfilter

        assert _dfilter("/path/to/private/folder") is False
        assert _dfilter("/path/private_data/folder") is False

    def test_dfilter_tissuumaps_directory(self):
        """Test that .tissuumaps directories are filtered out."""
        from tissuumaps.views import _dfilter

        assert _dfilter("/path/to/.tissuumaps/cache") is False

    def test_dfilter_normal_directory(self):
        """Test that normal directories are allowed."""
        from tissuumaps.views import _dfilter

        assert _dfilter("/path/to/data/folder") is True
        assert _dfilter("/home/user/images") is True
