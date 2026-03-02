# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PDF download stages."""

import json
from unittest.mock import patch

import pytest

from nemo_curator.stages.pdf.download import PDFDownloader, PDFURLGenerator


@pytest.fixture
def datasets_config(tmp_path):
    """Create a temporary datasets.json config file."""
    config = {
        "TEST_DATASET": {
            "description": "Test PDFs",
            "source": "url",
            "urls": [
                "https://example.com/doc1.pdf",
                "https://example.com/doc2.pdf",
                "https://example.com/doc3.pdf",
            ],
        },
        "LOCAL_ONLY": {
            "description": "Local PDFs",
            "source": "local",
            "urls": [],
        },
    }
    config_path = tmp_path / "datasets.json"
    config_path.write_text(json.dumps(config))
    return str(config_path)


class TestPDFURLGenerator:
    def test_generate_urls(self, datasets_config):
        gen = PDFURLGenerator(
            config_path=datasets_config,
            dataset_name="TEST_DATASET",
        )
        urls = gen.generate_urls()
        assert len(urls) == 3
        assert urls[0] == "https://example.com/doc1.pdf"

    def test_generate_urls_with_max_files(self, datasets_config):
        gen = PDFURLGenerator(
            config_path=datasets_config,
            dataset_name="TEST_DATASET",
            max_files=2,
        )
        urls = gen.generate_urls()
        assert len(urls) == 2

    def test_unknown_dataset(self, datasets_config):
        gen = PDFURLGenerator(
            config_path=datasets_config,
            dataset_name="NONEXISTENT",
        )
        with pytest.raises(KeyError, match="Unknown dataset"):
            gen.generate_urls()

    def test_non_url_source(self, datasets_config):
        gen = PDFURLGenerator(
            config_path=datasets_config,
            dataset_name="LOCAL_ONLY",
        )
        with pytest.raises(ValueError, match="not a URL source"):
            gen.generate_urls()

    def test_missing_config_file(self, tmp_path):
        gen = PDFURLGenerator(
            config_path=str(tmp_path / "missing.json"),
            dataset_name="TEST",
        )
        with pytest.raises(FileNotFoundError):
            gen.generate_urls()


class TestPDFDownloader:
    def test_get_output_filename(self, tmp_path):
        downloader = PDFDownloader(download_dir=str(tmp_path))
        assert downloader._get_output_filename(
            "https://example.com/path/to/document.pdf"
        ) == "document.pdf"

    def test_get_output_filename_adds_extension(self, tmp_path):
        downloader = PDFDownloader(download_dir=str(tmp_path))
        assert downloader._get_output_filename(
            "https://example.com/path/to/document"
        ) == "document.pdf"

    def test_get_output_filename_strips_query(self, tmp_path):
        downloader = PDFDownloader(download_dir=str(tmp_path))
        assert downloader._get_output_filename(
            "https://example.com/doc.pdf?token=abc"
        ) == "doc.pdf"

    def test_download_to_path_success(self, tmp_path):
        downloader = PDFDownloader(download_dir=str(tmp_path))
        output = tmp_path / "test.pdf"

        with patch("urllib.request.urlretrieve") as mock_retrieve:
            mock_retrieve.return_value = (str(output), None)
            success, error = downloader._download_to_path(
                "https://example.com/test.pdf", str(output)
            )

        assert success is True
        assert error is None

    def test_download_to_path_failure(self, tmp_path):
        downloader = PDFDownloader(download_dir=str(tmp_path))
        output = tmp_path / "test.pdf"

        with patch("urllib.request.urlretrieve", side_effect=OSError("Network error")):
            success, error = downloader._download_to_path(
                "https://example.com/test.pdf", str(output)
            )

        assert success is False
        assert "Network error" in error

    def test_download_idempotent(self, tmp_path):
        """Base class download() skips existing non-empty files."""
        downloader = PDFDownloader(download_dir=str(tmp_path), verbose=True)

        # Create an existing file
        existing = tmp_path / "test.pdf"
        existing.write_bytes(b"%PDF-1.4 test content")

        result = downloader.download("https://example.com/test.pdf")
        assert result == str(existing)
