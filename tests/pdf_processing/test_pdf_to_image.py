# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for PDFToImageStage."""

import base64
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

from nemo_curator.stages.pdf.image_conversion import PDFToImageStage
from nemo_curator.tasks import DocumentBatch


class TestPDFToImageStage:
    """Test cases for PDFToImageStage."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = PDFToImageStage()
        assert stage.pdf_path_field == "pdf_path"
        assert stage.dpi == 300
        assert stage.image_format == "PNG"
        assert stage.output_field == "page_images"
        assert stage.name == "pdf_to_image"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        stage = PDFToImageStage(
            pdf_path_field="custom_pdf_path",
            dpi=150,
            image_format="JPEG",
            output_field="custom_images",
        )
        assert stage.pdf_path_field == "custom_pdf_path"
        assert stage.dpi == 150
        assert stage.image_format == "JPEG"
        assert stage.output_field == "custom_images"

    def test_inputs_outputs(self):
        """Test inputs and outputs configuration."""
        stage = PDFToImageStage()
        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["pdf_path"])
        assert outputs == (["data"], ["pdf_path", "page_images"])

    @patch("nemo_curator.stages.pdf.image_conversion.pdf_to_images")
    def test_process_success(self, mock_pdf_to_images):
        """Test process with successful PDF conversion."""
        # Create mock images
        mock_image1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        mock_image2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
        mock_pdf_to_images.return_value = [mock_image1, mock_image2]

        # Create test batch
        df = pd.DataFrame({"pdf_path": ["/tmp/test.pdf"]})
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = PDFToImageStage()
        result = stage.process(batch)

        # Verify
        assert "page_images" in result.to_pandas().columns
        page_images_json = result.to_pandas()["page_images"].iloc[0]
        page_images = json.loads(page_images_json)

        assert len(page_images) == 2
        assert page_images[0]["page_number"] == 0
        assert page_images[1]["page_number"] == 1
        assert page_images[0]["width"] == 100
        assert page_images[0]["height"] == 100
        assert "image_base64" in page_images[0]

        # Verify base64 decoding works
        decoded = base64.b64decode(page_images[0]["image_base64"])
        assert len(decoded) > 0

    @patch("nemo_curator.stages.pdf.image_conversion.pdf_to_images")
    def test_process_with_invalid_pdf(self, mock_pdf_to_images):
        """Test process with invalid PDF path."""
        # Mock exception
        mock_pdf_to_images.side_effect = Exception("PDF not found")

        # Create test batch
        df = pd.DataFrame({"pdf_path": ["/tmp/invalid.pdf"]})
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = PDFToImageStage()
        result = stage.process(batch)

        # Verify empty result on error
        assert "page_images" in result.to_pandas().columns
        page_images_json = result.to_pandas()["page_images"].iloc[0]
        page_images = json.loads(page_images_json)
        assert page_images == []

    @patch("nemo_curator.stages.pdf.image_conversion.pdf_to_images")
    def test_process_json_structure(self, mock_pdf_to_images):
        """Test JSON structure of page_images output."""
        # Create mock image
        mock_image = Image.new("RGB", (200, 300), color=(0, 0, 255))
        mock_pdf_to_images.return_value = [mock_image]

        # Create test batch
        df = pd.DataFrame({"pdf_path": ["/tmp/test.pdf"]})
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = PDFToImageStage()
        result = stage.process(batch)

        # Verify JSON structure
        page_images_json = result.to_pandas()["page_images"].iloc[0]
        page_images = json.loads(page_images_json)

        assert len(page_images) == 1
        page_data = page_images[0]

        # Check required fields
        assert "page_number" in page_data
        assert "width" in page_data
        assert "height" in page_data
        assert "image_base64" in page_data

        # Verify values
        assert page_data["page_number"] == 0
        assert page_data["width"] == 200
        assert page_data["height"] == 300
        assert isinstance(page_data["image_base64"], str)

    @patch("nemo_curator.stages.pdf.image_conversion.pdf_to_images")
    def test_base64_decoding_produces_valid_images(self, mock_pdf_to_images):
        """Test base64 decoding produces valid images."""
        # Create mock image
        mock_image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        mock_pdf_to_images.return_value = [mock_image]

        # Create test batch
        df = pd.DataFrame({"pdf_path": ["/tmp/test.pdf"]})
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = PDFToImageStage()
        result = stage.process(batch)

        # Decode and verify
        page_images_json = result.to_pandas()["page_images"].iloc[0]
        page_images = json.loads(page_images_json)
        base64_str = page_images[0]["image_base64"]

        # Decode base64 and load as image
        from io import BytesIO

        image_bytes = base64.b64decode(base64_str)
        decoded_image = Image.open(BytesIO(image_bytes))

        # Verify image properties
        assert decoded_image.size == (100, 100)
        assert decoded_image.mode == "RGB"

    @patch("nemo_curator.stages.pdf.image_conversion.pdf_to_images")
    def test_process_multiple_pdfs(self, mock_pdf_to_images):
        """Test processing multiple PDFs."""
        # Create different mock images for each PDF
        mock_image1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        mock_image2 = Image.new("RGB", (200, 200), color=(0, 255, 0))

        def pdf_to_images_side_effect(path, dpi):
            if "pdf1" in path:
                return [mock_image1]
            return [mock_image2]

        mock_pdf_to_images.side_effect = pdf_to_images_side_effect

        # Create test batch with multiple PDFs
        df = pd.DataFrame({"pdf_path": ["/tmp/pdf1.pdf", "/tmp/pdf2.pdf"]})
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = PDFToImageStage()
        result = stage.process(batch)

        # Verify both processed
        result_df = result.to_pandas()
        assert len(result_df) == 2

        # Check first PDF result
        page_images1 = json.loads(result_df["page_images"].iloc[0])
        assert len(page_images1) == 1
        assert page_images1[0]["width"] == 100

        # Check second PDF result
        page_images2 = json.loads(result_df["page_images"].iloc[1])
        assert len(page_images2) == 1
        assert page_images2[0]["width"] == 200
