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

"""Tests for BoundingBoxExtractionStage."""

import json

import pandas as pd
import pytest

from nemo_curator.stages.pdf.bbox_extraction import BoundingBoxExtractionStage
from nemo_curator.tasks import DocumentBatch


class TestBoundingBoxExtractionStage:
    """Test cases for BoundingBoxExtractionStage."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = BoundingBoxExtractionStage()
        assert stage.page_images_field == "page_images"
        assert stage.layout_objects_field == "layout_objects"
        assert stage.output_field == "cropped_regions"
        assert stage.name == "bbox_extraction"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        stage = BoundingBoxExtractionStage(
            page_images_field="custom_images",
            layout_objects_field="custom_layout",
            output_field="custom_cropped",
        )
        assert stage.page_images_field == "custom_images"
        assert stage.layout_objects_field == "custom_layout"
        assert stage.output_field == "custom_cropped"

    def test_inputs_outputs(self):
        """Test inputs and outputs configuration."""
        stage = BoundingBoxExtractionStage()
        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["page_images", "layout_objects"])
        assert outputs == (["data"], ["page_images", "layout_objects", "cropped_regions"])

    def test_process_crops_images_correctly(self, sample_page_images, sample_layout_objects, sample_base64_image):
        """Test process crops images correctly using layout bounding boxes."""
        # Combine page_images and layout_objects
        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": [sample_page_images.to_pandas()["page_images"].iloc[0]],
                "layout_objects": [sample_layout_objects.to_pandas()["layout_objects"].iloc[0]],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = BoundingBoxExtractionStage()
        result = stage.process(batch)

        # Verify
        assert "cropped_regions" in result.to_pandas().columns
        cropped_regions_json = result.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)

        # Should have cropped regions for each object
        assert len(cropped_regions) == 3  # text, table, image
        assert all("cropped_image_base64" in region for region in cropped_regions)
        assert all("bbox" in region for region in cropped_regions)
        assert all("type" in region for region in cropped_regions)

    def test_cropping_with_normalized_coordinates(self, sample_page_images, sample_base64_image):
        """Test cropping with normalized bbox coordinates."""
        # Create layout with normalized coordinates
        layout_data = [
            {
                "page_number": 0,
                "width": 100,
                "height": 100,
                "objects": [{"type": "text", "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}, "content": "Full page"}],
            }
        ]

        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": [sample_page_images.to_pandas()["page_images"].iloc[0]],
                "layout_objects": [json.dumps(layout_data)],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = BoundingBoxExtractionStage()
        result = stage.process(batch)

        # Verify cropped region exists
        cropped_regions_json = result.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)

        assert len(cropped_regions) == 1
        assert cropped_regions[0]["type"] == "text"
        assert "cropped_image_base64" in cropped_regions[0]

    def test_handling_missing_layout_objects(self, sample_page_images):
        """Test handling missing layout objects gracefully."""
        # Create batch with empty layout objects
        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": [sample_page_images.to_pandas()["page_images"].iloc[0]],
                "layout_objects": [json.dumps([])],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = BoundingBoxExtractionStage()
        result = stage.process(batch)

        # Should return empty cropped regions
        cropped_regions_json = result.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)
        assert cropped_regions == []

    def test_handling_mismatched_page_numbers(self, sample_base64_image):
        """Test handling mismatched page numbers between images and layouts."""
        # Page images for page 0
        page_images = json.dumps([{"page_number": 0, "width": 100, "height": 100, "image_base64": sample_base64_image}])

        # Layout objects for page 1 (mismatch)
        layout_data = [
            {
                "page_number": 1,
                "width": 100,
                "height": 100,
                "objects": [{"type": "text", "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2}, "content": "Text"}],
            }
        ]

        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": [page_images],
                "layout_objects": [json.dumps(layout_data)],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = BoundingBoxExtractionStage()
        result = stage.process(batch)

        # Should handle mismatch gracefully
        cropped_regions_json = result.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)
        assert cropped_regions == []  # No matching pages

    def test_cropped_regions_output_structure(self, sample_page_images, sample_layout_objects):
        """Test cropped_regions output structure."""
        # Combine data
        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": [sample_page_images.to_pandas()["page_images"].iloc[0]],
                "layout_objects": [sample_layout_objects.to_pandas()["layout_objects"].iloc[0]],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = BoundingBoxExtractionStage()
        result = stage.process(batch)

        # Verify structure
        cropped_regions_json = result.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)

        assert len(cropped_regions) > 0

        # Check required fields
        region = cropped_regions[0]
        assert "page_number" in region
        assert "object_index" in region
        assert "type" in region
        assert "bbox" in region
        assert "content" in region
        assert "cropped_image_base64" in region

        # Verify bbox structure
        bbox = region["bbox"]
        assert "x" in bbox
        assert "y" in bbox
        assert "width" in bbox
        assert "height" in bbox

    def test_process_with_objects_without_bbox(self, sample_page_images, sample_base64_image):
        """Test process skips objects without bbox."""
        # Create layout with object missing bbox
        layout_data = [
            {
                "page_number": 0,
                "width": 100,
                "height": 100,
                "objects": [
                    {"type": "text", "content": "No bbox here"},  # Missing bbox
                    {
                        "type": "table",
                        "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2},
                        "content": "Has bbox",
                    },
                ],
            }
        ]

        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": [sample_page_images.to_pandas()["page_images"].iloc[0]],
                "layout_objects": [json.dumps(layout_data)],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process
        stage = BoundingBoxExtractionStage()
        result = stage.process(batch)

        # Should only crop the object with bbox
        cropped_regions_json = result.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)

        assert len(cropped_regions) == 1
        assert cropped_regions[0]["type"] == "table"

    def test_process_exception_handling(self):
        """Test process handles exceptions gracefully."""
        # Create invalid batch
        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": ["invalid json"],
                "layout_objects": ["invalid json"],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        # Process should not crash
        stage = BoundingBoxExtractionStage()
        result = stage.process(batch)

        # Should return empty on error
        cropped_regions_json = result.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)
        assert cropped_regions == []
