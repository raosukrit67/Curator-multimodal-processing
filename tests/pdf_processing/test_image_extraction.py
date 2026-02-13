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

"""Tests for ImageExtractionStage."""

import json

import pandas as pd
import pytest

from nemo_curator.stages.pdf.image_extraction import ImageExtractionStage
from nemo_curator.tasks import DocumentBatch


class TestImageExtractionStage:
    """Test cases for ImageExtractionStage."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = ImageExtractionStage()
        assert stage.classified_regions_field == "classified_regions"
        assert stage.output_field == "extracted_images"
        assert stage.image_types == ["image", "figure", "chart"]
        assert stage.name == "image_extraction"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        stage = ImageExtractionStage(
            classified_regions_field="custom_classified",
            output_field="custom_images",
            image_types=["figure", "chart"],
        )
        assert stage.classified_regions_field == "custom_classified"
        assert stage.output_field == "custom_images"
        assert stage.image_types == ["figure", "chart"]

    def test_inputs_outputs(self):
        """Test inputs and outputs configuration."""
        stage = ImageExtractionStage()
        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["classified_regions"])
        assert outputs == (["data"], ["classified_regions", "extracted_images"])

    def test_process_filters_and_extracts_image_types(self, sample_classified_regions):
        """Test process filters and extracts image/figure/chart types."""
        stage = ImageExtractionStage()
        result = stage.process(sample_classified_regions)

        # Verify
        assert "extracted_images" in result.to_pandas().columns
        extracted_images_json = result.to_pandas()["extracted_images"].iloc[0]
        extracted_images = json.loads(extracted_images_json)

        # Should extract the figure region (object_index 2)
        assert len(extracted_images) == 1
        image = extracted_images[0]

        assert image["page_number"] == 0
        assert image["object_index"] == 2
        assert image["type"] == "figure"
        assert "bbox" in image
        assert "image_base64" in image

    def test_filtering_logic_correctly_identifies_content_types(self, sample_base64_image):
        """Test filtering logic correctly identifies content types."""
        # Create regions with different types
        classified_data = [
            {
                "page_number": 0,
                "object_index": 0,
                "classified_type": "text",
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2},
                "cropped_image_base64": sample_base64_image,
            },
            {
                "page_number": 0,
                "object_index": 1,
                "classified_type": "image",
                "bbox": {"x": 0.1, "y": 0.3, "width": 0.8, "height": 0.3},
                "cropped_image_base64": sample_base64_image,
            },
            {
                "page_number": 0,
                "object_index": 2,
                "classified_type": "figure",
                "bbox": {"x": 0.1, "y": 0.6, "width": 0.8, "height": 0.3},
                "cropped_image_base64": sample_base64_image,
            },
            {
                "page_number": 0,
                "object_index": 3,
                "classified_type": "chart",
                "bbox": {"x": 0.1, "y": 0.9, "width": 0.8, "height": 0.3},
                "cropped_image_base64": sample_base64_image,
            },
            {
                "page_number": 0,
                "object_index": 4,
                "classified_type": "table",
                "bbox": {"x": 0.1, "y": 1.2, "width": 0.8, "height": 0.3},
                "cropped_image_base64": sample_base64_image,
            },
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

        stage = ImageExtractionStage()
        result = stage.process(batch)

        extracted_images_json = result.to_pandas()["extracted_images"].iloc[0]
        extracted_images = json.loads(extracted_images_json)

        # Should extract only image, figure, and chart (not text or table)
        assert len(extracted_images) == 3

        extracted_types = [img["type"] for img in extracted_images]
        assert "image" in extracted_types
        assert "figure" in extracted_types
        assert "chart" in extracted_types
        assert "text" not in extracted_types
        assert "table" not in extracted_types

    def test_output_contains_expected_image_data_and_metadata(self, sample_classified_regions):
        """Test output contains expected image data and metadata."""
        stage = ImageExtractionStage()
        result = stage.process(sample_classified_regions)

        extracted_images_json = result.to_pandas()["extracted_images"].iloc[0]
        extracted_images = json.loads(extracted_images_json)

        if len(extracted_images) > 0:
            image = extracted_images[0]

            # Verify required fields
            assert "page_number" in image
            assert "object_index" in image
            assert "type" in image
            assert "bbox" in image
            assert "image_base64" in image

            # Verify bbox structure
            bbox = image["bbox"]
            assert "x" in bbox
            assert "y" in bbox
            assert "width" in bbox
            assert "height" in bbox

            # Verify image data
            assert isinstance(image["image_base64"], str)
            assert len(image["image_base64"]) > 0

    def test_process_with_custom_image_types(self, sample_base64_image):
        """Test process with custom image_types filter."""
        # Create regions
        classified_data = [
            {
                "page_number": 0,
                "object_index": 0,
                "classified_type": "image",
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.3},
                "cropped_image_base64": sample_base64_image,
            },
            {
                "page_number": 0,
                "object_index": 1,
                "classified_type": "figure",
                "bbox": {"x": 0.1, "y": 0.4, "width": 0.8, "height": 0.3},
                "cropped_image_base64": sample_base64_image,
            },
            {
                "page_number": 0,
                "object_index": 2,
                "classified_type": "chart",
                "bbox": {"x": 0.1, "y": 0.7, "width": 0.8, "height": 0.3},
                "cropped_image_base64": sample_base64_image,
            },
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

        # Only extract figures
        stage = ImageExtractionStage(image_types=["figure"])
        result = stage.process(batch)

        extracted_images_json = result.to_pandas()["extracted_images"].iloc[0]
        extracted_images = json.loads(extracted_images_json)

        # Should only extract figure
        assert len(extracted_images) == 1
        assert extracted_images[0]["type"] == "figure"

    def test_process_with_empty_regions(self):
        """Test process with empty regions."""
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

        stage = ImageExtractionStage()
        result = stage.process(batch)

        extracted_images_json = result.to_pandas()["extracted_images"].iloc[0]
        extracted_images = json.loads(extracted_images_json)
        assert extracted_images == []

    def test_process_skips_regions_without_image_data(self, sample_base64_image):
        """Test process skips regions without cropped_image_base64."""
        classified_data = [
            {
                "page_number": 0,
                "object_index": 0,
                "classified_type": "image",
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.3},
                # Missing cropped_image_base64
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

        stage = ImageExtractionStage()
        result = stage.process(batch)

        # Should skip region without image data
        extracted_images_json = result.to_pandas()["extracted_images"].iloc[0]
        extracted_images = json.loads(extracted_images_json)
        assert len(extracted_images) == 0

    def test_process_exception_handling(self):
        """Test process handles exceptions gracefully."""
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

        stage = ImageExtractionStage()
        result = stage.process(batch)

        extracted_images_json = result.to_pandas()["extracted_images"].iloc[0]
        extracted_images = json.loads(extracted_images_json)
        assert extracted_images == []
