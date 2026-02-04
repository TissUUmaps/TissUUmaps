"""
Tests for multi-channel TIFF support functions in views.py.
"""

import os

import pytest

# Skip tests if pyvips is not available or can't be loaded
try:
    import pyvips

    PYVIPS_AVAILABLE = True
except (ImportError, OSError):
    PYVIPS_AVAILABLE = False
    pyvips = None

pytestmark = pytest.mark.skipif(
    not PYVIPS_AVAILABLE, reason="pyvips library not available"
)


class TestIsMultichannelTiff:
    """Tests for the _is_multichannel_tiff function."""

    def test_unsupported_format_returns_false(self, temp_dir):
        """Test that unsupported file formats return False."""
        from tissuumaps.views import _is_multichannel_tiff

        # Test with non-TIFF extension
        result = _is_multichannel_tiff("/path/to/file.png")
        assert result == (False, 0, [])

        result = _is_multichannel_tiff("/path/to/file.jpg")
        assert result == (False, 0, [])

        result = _is_multichannel_tiff("/path/to/file.txt")
        assert result == (False, 0, [])

    def test_supported_tiff_extensions(self, temp_dir):
        """Test that supported TIFF extensions are recognized."""
        from tissuumaps.views import _is_multichannel_tiff

        # These should attempt to open the file (and fail gracefully)
        # since the files don't exist
        for ext in [".tif", ".tiff", ".ome.tif", ".ome.tiff", ".qptiff"]:
            result = _is_multichannel_tiff(f"/nonexistent/file{ext}")
            # Should return False due to file not found, but not crash
            assert result[0] is False

    def test_single_channel_tiff(self, temp_dir):
        """Test that single-channel TIFF is not detected as multi-channel."""
        from tissuumaps.views import _is_multichannel_tiff

        # Create a simple single-channel TIFF using pyvips
        img = pyvips.Image.black(100, 100, bands=1)
        tiff_path = os.path.join(temp_dir, "single_channel.tif")
        img.tiffsave(tiff_path)

        result = _is_multichannel_tiff(tiff_path)
        is_multichannel, num_channels, channel_names = result
        assert is_multichannel is False

    def test_rgb_tiff_not_multichannel(self, temp_dir):
        """Test that RGB (3-band) TIFF is not detected as multi-channel."""
        from tissuumaps.views import _is_multichannel_tiff

        # Create an RGB TIFF (3 bands)
        img = pyvips.Image.black(100, 100, bands=3)
        tiff_path = os.path.join(temp_dir, "rgb.tif")
        img.tiffsave(tiff_path)

        result = _is_multichannel_tiff(tiff_path)
        is_multichannel, num_channels, channel_names = result
        # RGB (3 bands) should not be detected as multi-channel (threshold is >4)
        assert is_multichannel is False

    def test_rgba_tiff_not_multichannel(self, temp_dir):
        """Test that RGBA (4-band) TIFF is not detected as multi-channel."""
        from tissuumaps.views import _is_multichannel_tiff

        # Create an RGBA TIFF (4 bands)
        img = pyvips.Image.black(100, 100, bands=4)
        tiff_path = os.path.join(temp_dir, "rgba.tif")
        img.tiffsave(tiff_path)

        result = _is_multichannel_tiff(tiff_path)
        is_multichannel, num_channels, channel_names = result
        # RGBA (4 bands) should not be detected as multi-channel (threshold is >4)
        assert is_multichannel is False

    def test_multiband_tiff_detected(self, temp_dir):
        """Test that multi-band (>4 bands) TIFF is detected as multi-channel."""
        from tissuumaps.views import _is_multichannel_tiff

        # Create a 5-band TIFF
        img = pyvips.Image.black(100, 100, bands=5)
        tiff_path = os.path.join(temp_dir, "multiband.tif")
        img.tiffsave(tiff_path)

        result = _is_multichannel_tiff(tiff_path)
        is_multichannel, num_channels, channel_names = result
        assert is_multichannel is True
        assert num_channels == 5
        assert len(channel_names) == 5


class TestConvertToPyramidal8bit:
    """Tests for the _convert_to_pyramidal_8bit function."""

    def test_convert_8bit_image(self, temp_dir):
        """Test conversion of an 8-bit image."""
        from tissuumaps import app
        from tissuumaps.views import _convert_to_pyramidal_8bit

        # Create a simple 8-bit image
        img = pyvips.Image.black(256, 256, bands=1).cast("uchar")
        # Add some data
        img = img + 128

        output_path = os.path.join(temp_dir, "output.tif")
        _convert_to_pyramidal_8bit(img, output_path)

        assert os.path.isfile(output_path)

        # Verify the output is a valid TIFF
        result_img = pyvips.Image.new_from_file(output_path)
        assert result_img.width == 256
        assert result_img.height == 256

    def test_convert_16bit_image(self, temp_dir):
        """Test conversion of a 16-bit image (requires rescaling)."""
        from tissuumaps import app
        from tissuumaps.views import _convert_to_pyramidal_8bit

        # Create a 16-bit image with values > 255
        img = pyvips.Image.black(256, 256, bands=1).cast("ushort")
        img = img + 1000  # Values > 255

        output_path = os.path.join(temp_dir, "output_16bit.tif")
        _convert_to_pyramidal_8bit(img, output_path)

        assert os.path.isfile(output_path)

        # Verify the output is a valid TIFF
        result_img = pyvips.Image.new_from_file(output_path)
        assert result_img.width == 256
        assert result_img.height == 256


class TestExtractMultichannelLayers:
    """Tests for the _extract_multichannel_layers function."""

    def test_extract_single_layer(self, temp_dir):
        """Test extraction from a single-channel image."""
        from tissuumaps import app
        from tissuumaps.views import _extract_multichannel_layers

        # Ensure basedir is set
        app.basedir = temp_dir

        # Create a simple single-channel TIFF
        img = pyvips.Image.black(100, 100, bands=1)
        input_path = os.path.join(temp_dir, "input.tif")
        img.tiffsave(input_path)

        output_folder = os.path.join(temp_dir, "output")
        layers = _extract_multichannel_layers(input_path, output_folder)

        assert len(layers) == 1
        assert "name" in layers[0]
        assert "tileSource" in layers[0]

    def test_extract_multiband_layers(self, temp_dir):
        """Test extraction from a multi-band (>4 bands) image."""
        from tissuumaps import app
        from tissuumaps.views import _extract_multichannel_layers

        # Ensure basedir is set
        app.basedir = temp_dir

        # Create a 5-band TIFF
        img = pyvips.Image.black(100, 100, bands=5)
        input_path = os.path.join(temp_dir, "multiband.tif")
        img.tiffsave(input_path)

        output_folder = os.path.join(temp_dir, "output")
        layers = _extract_multichannel_layers(input_path, output_folder)

        assert len(layers) == 5
        for i, layer in enumerate(layers):
            assert "name" in layer
            assert "tileSource" in layer
            assert f"Channel_{i}" in layer["name"]

    def test_ome_tif_basename_handling(self, temp_dir):
        """Test that .ome extension is properly stripped from basename."""
        from tissuumaps import app
        from tissuumaps.views import _extract_multichannel_layers

        app.basedir = temp_dir

        # Create a TIFF with .ome.tif extension
        img = pyvips.Image.black(100, 100, bands=1)
        input_path = os.path.join(temp_dir, "test.ome.tif")
        img.tiffsave(input_path)

        output_folder = os.path.join(temp_dir, "output")
        layers = _extract_multichannel_layers(input_path, output_folder)

        # The layer name should not contain ".ome"
        assert len(layers) == 1
        assert ".ome" not in layers[0]["name"].lower()
