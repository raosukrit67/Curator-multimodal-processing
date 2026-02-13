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

"""Tests for ContentTypeClassificationStage."""

import json

import pytest

from nemo_curator.stages.pdf.content_classification import ContentTypeClassificationStage


class TestContentTypeClassificationStage:
    """Test cases for ContentTypeClassificationStage."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = ContentTypeClassificationStage()
        assert stage.cropped_regions_field == "cropped_regions"
        assert stage.output_field == "classified_regions"
        assert stage.min_table_aspect_ratio == 1.5
        assert stage.min_text_area == 0.01
        assert stage.name == "content_classification"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        stage = ContentTypeClassificationStage(
            cropped_regions_field="custom_cropped",
            output_field="custom_classified",
            min_table_aspect_ratio=2.0,
            min_text_area=0.02,
        )
        assert stage.cropped_regions_field == "custom_cropped"
        assert stage.output_field == "custom_classified"
        assert stage.min_table_aspect_ratio == 2.0
        assert stage.min_text_area == 0.02

    def test_inputs_outputs(self):
        """Test inputs and outputs configuration."""
        stage = ContentTypeClassificationStage()
        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["cropped_regions"])
        assert outputs == (["data"], ["cropped_regions", "classified_regions"])

    def test_classify_region_detects_tables_via_latex(self):
        """Test _classify_region detects tables via LaTeX patterns."""
        stage = ContentTypeClassificationStage()

        # Region with LaTeX table
        region = {
            "type": "unknown",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.3},
            "content": "\\begin{tabular}{ll}\nA & B \\\\\nC & D \\\\\n\\end{tabular}",
        }

        classified_type = stage._classify_region(region)
        assert classified_type == "table"

    def test_classify_region_detects_tables_via_hline(self):
        """Test _classify_region detects tables via \\hline pattern."""
        stage = ContentTypeClassificationStage()

        # Region with \hline
        region = {
            "type": "text",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2},
            "content": "Some content\n\\hline\nMore content",
        }

        classified_type = stage._classify_region(region)
        assert classified_type == "table"

    def test_classify_region_uses_aspect_ratio_heuristics(self):
        """Test _classify_region uses aspect ratio heuristics correctly."""
        stage = ContentTypeClassificationStage(min_table_aspect_ratio=2.0)

        # Wide region (aspect ratio > 2.0, area > 0.05)
        region = {
            "type": "unknown",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2},  # AR = 4.0, area = 0.16
            "content": "",
        }

        classified_type = stage._classify_region(region)
        assert classified_type == "table"

    def test_classify_region_classifies_small_regions_as_text(self):
        """Test _classify_region classifies small regions as text."""
        stage = ContentTypeClassificationStage(min_text_area=0.01)

        # Small region
        region = {
            "type": "unknown",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.05, "height": 0.05},  # area = 0.0025 < 0.01
            "content": "",
        }

        classified_type = stage._classify_region(region)
        assert classified_type == "text"

    def test_classify_region_identifies_images_by_aspect_ratio(self):
        """Test _classify_region identifies images by aspect ratio and area."""
        stage = ContentTypeClassificationStage()

        # Square region with significant area
        region = {
            "type": "image",
            "bbox": {"x": 0.2, "y": 0.2, "width": 0.4, "height": 0.4},  # AR = 1.0, area = 0.16
            "content": "",
        }

        classified_type = stage._classify_region(region)
        assert classified_type == "image"

    def test_classify_region_preserves_figure_type(self):
        """Test _classify_region preserves figure type for image-like regions."""
        stage = ContentTypeClassificationStage()

        # Region originally classified as figure
        region = {
            "type": "figure",
            "bbox": {"x": 0.2, "y": 0.2, "width": 0.5, "height": 0.4},  # AR = 1.25, area = 0.2
            "content": "",
        }

        classified_type = stage._classify_region(region)
        assert classified_type == "figure"

    def test_classify_region_preserves_chart_type(self):
        """Test _classify_region preserves chart type for image-like regions."""
        stage = ContentTypeClassificationStage()

        # Region originally classified as chart
        region = {
            "type": "chart",
            "bbox": {"x": 0.15, "y": 0.15, "width": 0.6, "height": 0.5},  # AR = 1.2, area = 0.3
            "content": "",
        }

        classified_type = stage._classify_region(region)
        assert classified_type == "chart"

    def test_classify_region_defaults_to_original_type(self):
        """Test _classify_region defaults to original type when applicable."""
        stage = ContentTypeClassificationStage()

        # Valid original type, doesn't match other heuristics
        region = {
            "type": "text",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.2},  # area = 0.06
            "content": "Some text",
        }

        classified_type = stage._classify_region(region)
        assert classified_type == "text"

    def test_process_with_mixed_content_types(self, sample_cropped_regions):
        """Test process with mixed content types produces correct classifications."""
        stage = ContentTypeClassificationStage()

        # Process
        result = stage.process(sample_cropped_regions)

        # Verify
        assert "classified_regions" in result.to_pandas().columns
        classified_regions_json = result.to_pandas()["classified_regions"].iloc[0]
        classified_regions = json.loads(classified_regions_json)

        # Should have same number of regions
        cropped_regions_json = sample_cropped_regions.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)
        assert len(classified_regions) == len(cropped_regions)

        # All should have classified_type field
        assert all("classified_type" in region for region in classified_regions)

        # Check specific classifications
        type_map = {region["object_index"]: region["classified_type"] for region in classified_regions}
        assert type_map[0] == "text"  # Text region
        assert type_map[1] == "table"  # Table with LaTeX
        assert type_map[2] in ["image", "figure", "chart"]  # Image-type region

    def test_classification_logic_for_edge_cases(self):
        """Test classification logic for edge cases."""
        stage = ContentTypeClassificationStage()

        # Empty region
        empty_region = {"type": "unknown", "bbox": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}, "content": ""}
        assert stage._classify_region(empty_region) == "text"  # Small area defaults to text

        # Region with no content field
        no_content = {"type": "text", "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.2}}
        assert stage._classify_region(no_content) == "text"

        # Region with unknown type but table content
        table_region = {
            "type": "unknown",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.3},
            "content": "\\begin{tabular}{c}\nData\n\\end{tabular}",
        }
        assert stage._classify_region(table_region) == "table"

    def test_process_with_empty_regions(self):
        """Test process handling regions without content."""
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

        # Create batch with empty regions
        df = pd.DataFrame(
            {
                "pdf_path": ["/tmp/test.pdf"],
                "page_images": ["[]"],
                "layout_objects": ["[]"],
                "cropped_regions": [json.dumps([])],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        stage = ContentTypeClassificationStage()
        result = stage.process(batch)

        # Should return empty
        classified_regions_json = result.to_pandas()["classified_regions"].iloc[0]
        classified_regions = json.loads(classified_regions_json)
        assert classified_regions == []

    def test_process_preserves_region_metadata(self, sample_cropped_regions):
        """Test process preserves all region metadata."""
        stage = ContentTypeClassificationStage()
        result = stage.process(sample_cropped_regions)

        # Get regions
        cropped_regions_json = sample_cropped_regions.to_pandas()["cropped_regions"].iloc[0]
        cropped_regions = json.loads(cropped_regions_json)

        classified_regions_json = result.to_pandas()["classified_regions"].iloc[0]
        classified_regions = json.loads(classified_regions_json)

        # Verify all original fields preserved
        for orig, classified in zip(cropped_regions, classified_regions):
            assert classified["page_number"] == orig["page_number"]
            assert classified["object_index"] == orig["object_index"]
            assert classified["type"] == orig["type"]
            assert classified["bbox"] == orig["bbox"]
            assert classified["content"] == orig["content"]
            assert "classified_type" in classified  # New field added

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
                "cropped_regions": ["invalid json"],
            }
        )
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        stage = ContentTypeClassificationStage()
        result = stage.process(batch)

        # Should return empty on error
        classified_regions_json = result.to_pandas()["classified_regions"].iloc[0]
        classified_regions = json.loads(classified_regions_json)
        assert classified_regions == []
