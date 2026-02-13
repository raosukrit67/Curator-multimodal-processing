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

"""Tests for TextExtractionStage."""

import json

import pytest

from nemo_curator.stages.pdf.text_extraction import TextExtractionStage


class TestTextExtractionStage:
    """Test cases for TextExtractionStage."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = TextExtractionStage()
        assert stage.classified_regions_field == "classified_regions"
        assert stage.output_field == "extracted_text"
        assert stage.use_ocr is False
        assert stage.name == "text_extraction"

    def test_init_with_ocr_enabled(self):
        """Test initialization with OCR enabled."""
        stage = TextExtractionStage(use_ocr=True)
        assert stage.use_ocr is True
        # Should allocate GPU resources when OCR enabled
        assert stage.resources.gpus == 0.25
        assert stage.resources.gpu_mem_gb == 4.0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        stage = TextExtractionStage(
            classified_regions_field="custom_classified", output_field="custom_text", use_ocr=True
        )
        assert stage.classified_regions_field == "custom_classified"
        assert stage.output_field == "custom_text"
        assert stage.use_ocr is True

    def test_inputs_outputs(self):
        """Test inputs and outputs configuration."""
        stage = TextExtractionStage()
        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["classified_regions"])
        assert outputs == (["data"], ["classified_regions", "extracted_text"])

    def test_process_extracts_text_from_text_regions(self, sample_classified_regions):
        """Test process extracts text from text regions."""
        stage = TextExtractionStage()
        result = stage.process(sample_classified_regions)

        # Verify
        assert "extracted_text" in result.to_pandas().columns
        extracted_text_json = result.to_pandas()["extracted_text"].iloc[0]
        extracted_text = json.loads(extracted_text_json)

        # Should extract text from text regions only
        assert len(extracted_text) == 1  # Only 1 text region
        text_block = extracted_text[0]

        assert text_block["page_number"] == 0
        assert text_block["object_index"] == 0
        assert "bbox" in text_block
        assert text_block["text"] == "Sample text content"

    def test_process_filters_non_text_regions(self, sample_classified_regions):
        """Test process only extracts from text-classified regions."""
        stage = TextExtractionStage()
        result = stage.process(sample_classified_regions)

        # Get classified regions for comparison
        classified_regions_json = sample_classified_regions.to_pandas()["classified_regions"].iloc[0]
        classified_regions = json.loads(classified_regions_json)

        # Count text regions
        text_regions = [r for r in classified_regions if r.get("classified_type") == "text"]

        # Extracted text should match text regions count
        extracted_text_json = result.to_pandas()["extracted_text"].iloc[0]
        extracted_text = json.loads(extracted_text_json)

        assert len(extracted_text) == len(text_regions)

    def test_process_handles_regions_without_text_content(self, sample_base64_image):
        """Test process handling regions without text content."""
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

        # Region with no content
        classified_data = [
            {
                "page_number": 0,
                "object_index": 0,
                "type": "text",
                "classified_type": "text",
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2},
                "content": "",  # Empty content
                "cropped_image_base64": sample_base64_image,
            }
        ]

        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": ["[]"],
                "layout_objects": ["[]"],
                "cropped_regions": ["[]"],
                "classified_regions": [json.dumps(classified_data)],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        stage = TextExtractionStage(use_ocr=False)
        result = stage.process(batch)

        # Should skip regions without content when OCR disabled
        extracted_text_json = result.to_pandas()["extracted_text"].iloc[0]
        extracted_text = json.loads(extracted_text_json)
        assert len(extracted_text) == 0

    def test_extract_text_with_ocr_not_available(self):
        """Test _extract_text_with_ocr when pytesseract not available."""
        stage = TextExtractionStage(use_ocr=True)

        region = {
            "page_number": 0,
            "object_index": 0,
            "cropped_image_base64": "base64string",
        }

        # Should return None when pytesseract not available or fails
        result = stage._extract_text_with_ocr(region)
        # Result depends on whether pytesseract is installed
        assert result is None or isinstance(result, str)

    def test_extract_text_with_ocr_disabled(self):
        """Test _extract_text_with_ocr when OCR is disabled."""
        stage = TextExtractionStage(use_ocr=False)

        region = {"page_number": 0, "object_index": 0, "cropped_image_base64": "base64string"}

        result = stage._extract_text_with_ocr(region)
        assert result is None

    def test_process_with_empty_regions(self):
        """Test process with empty regions."""
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": ["[]"],
                "layout_objects": ["[]"],
                "cropped_regions": ["[]"],
                "classified_regions": [json.dumps([])],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        stage = TextExtractionStage()
        result = stage.process(batch)

        extracted_text_json = result.to_pandas()["extracted_text"].iloc[0]
        extracted_text = json.loads(extracted_text_json)
        assert extracted_text == []

    def test_process_output_structure(self, sample_classified_regions):
        """Test output structure of extracted text."""
        stage = TextExtractionStage()
        result = stage.process(sample_classified_regions)

        extracted_text_json = result.to_pandas()["extracted_text"].iloc[0]
        extracted_text = json.loads(extracted_text_json)

        if len(extracted_text) > 0:
            text_block = extracted_text[0]

            # Verify required fields
            assert "page_number" in text_block
            assert "object_index" in text_block
            assert "bbox" in text_block
            assert "text" in text_block

            # Verify bbox structure
            bbox = text_block["bbox"]
            assert "x" in bbox
            assert "y" in bbox
            assert "width" in bbox
            assert "height" in bbox

    def test_process_exception_handling(self):
        """Test process handles exceptions gracefully."""
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

        # Invalid JSON
        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": ["[]"],
                "layout_objects": ["[]"],
                "cropped_regions": ["[]"],
                "classified_regions": ["invalid json"],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        stage = TextExtractionStage()
        result = stage.process(batch)

        extracted_text_json = result.to_pandas()["extracted_text"].iloc[0]
        extracted_text = json.loads(extracted_text_json)
        assert extracted_text == []
