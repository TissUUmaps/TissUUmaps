"""
Tests for Flask routes in views.py.
"""

import json
import os

import pytest


class TestPingRoute:
    """Tests for the /ping endpoint."""

    def test_ping_returns_pong(self, app_client):
        """Test that /ping endpoint returns 'pong'."""
        response = app_client.get("/ping")
        assert response.status_code == 200
        assert response.data == b"pong"


class TestIndexRoute:
    """Tests for the / (index) endpoint."""

    def test_index_returns_html(self, app_client):
        """Test that index returns HTML content."""
        response = app_client.get("/")
        assert response.status_code == 200
        assert b"TissUUmaps" in response.data or response.content_type == "text/html"


class TestFileTreeRoute:
    """Tests for the file tree endpoints."""

    def test_filetree_requires_read_only_false(self, app_client, temp_dir):
        """Test that filetree endpoint works when not read-only."""
        from tissuumaps import app

        app.config["READ_ONLY"] = False
        response = app_client.get("/filetree")
        # Should return the filetree template or empty when read-only
        assert response.status_code == 200


class TestTmapFileRoute:
    """Tests for the .tmap file endpoint."""

    def test_tmap_file_not_found(self, app_client):
        """Test that nonexistent .tmap file returns 404."""
        response = app_client.get("/nonexistent.tmap")
        assert response.status_code == 404

    def test_tmap_file_loads(self, app_client, temp_dir, sample_tmap_project):
        """Test that a valid .tmap file loads correctly."""
        from tissuumaps import app

        # Create a test tmap file
        tmap_path = os.path.join(temp_dir, "test.tmap")
        with open(tmap_path, "w") as f:
            json.dump(sample_tmap_project, f)

        response = app_client.get("/test.tmap")
        assert response.status_code == 200
        assert b"TissUUmaps" in response.data


class TestCsvFileRoute:
    """Tests for the CSV file endpoint."""

    def test_csv_file_not_found(self, app_client):
        """Test that nonexistent CSV file returns 404."""
        response = app_client.get("/nonexistent.csv")
        assert response.status_code == 404

    def test_csv_file_loads(self, app_client, temp_dir):
        """Test that a valid CSV file is served correctly."""
        # Create a test CSV file
        csv_path = os.path.join(temp_dir, "test.csv")
        with open(csv_path, "w") as f:
            f.write("col1,col2,col3\n1,2,3\n4,5,6\n")

        response = app_client.get("/test.csv")
        assert response.status_code == 200
        assert b"col1,col2,col3" in response.data


class TestJsonFileRoute:
    """Tests for the JSON/GeoJSON file endpoints."""

    def test_json_file_not_found(self, app_client):
        """Test that nonexistent JSON file returns 404."""
        response = app_client.get("/nonexistent.json")
        assert response.status_code == 404

    def test_json_file_loads(self, app_client, temp_dir):
        """Test that a valid JSON file is served correctly."""
        # Create a test JSON file
        json_path = os.path.join(temp_dir, "test.json")
        with open(json_path, "w") as f:
            json.dump({"key": "value"}, f)

        response = app_client.get("/test.json")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["key"] == "value"

    def test_geojson_file_loads(self, app_client, temp_dir):
        """Test that a valid GeoJSON file is served correctly."""
        # Create a test GeoJSON file
        geojson_path = os.path.join(temp_dir, "regions.geojson")
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                }
            ],
        }
        with open(geojson_path, "w") as f:
            json.dump(geojson_data, f)

        response = app_client.get("/regions.geojson")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["type"] == "FeatureCollection"


class TestErrorHandlers:
    """Tests for error handlers."""

    def test_404_error_page(self, app_client):
        """Test that 404 errors return a proper error page."""
        response = app_client.get("/this/path/does/not/exist.xyz")
        assert response.status_code == 404
