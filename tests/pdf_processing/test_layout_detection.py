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

"""Tests for LayoutDetectionStage."""

import json

import pytest

from nemo_curator.stages.pdf.layout_detection import LayoutDetectionStage


class TestLayoutDetectionStage:
    """Test cases for LayoutDetectionStage."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = LayoutDetectionStage()
        assert stage.model_identifier == "nvidia/nemoretriever-parse"
        assert stage.page_images_field == "page_images"
        assert stage.output_field == "layout_objects"
        assert stage.max_tokens == 3500
        assert stage.temperature == 0.0
        assert stage.name == "layout_detection"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        stage = LayoutDetectionStage(
            model_identifier="custom/model",
            page_images_field="custom_images",
            output_field="custom_layout",
            max_tokens=2000,
            temperature=0.5,
        )
        assert stage.model_identifier == "custom/model"
        assert stage.page_images_field == "custom_images"
        assert stage.output_field == "custom_layout"
        assert stage.max_tokens == 2000
        assert stage.temperature == 0.5

    def test_inputs_outputs(self):
        """Test inputs and outputs configuration."""
        stage = LayoutDetectionStage()
        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["page_images"])
        assert outputs == (["data"], ["page_images", "layout_objects"])

    def test_prepare_prompt(self, sample_base64_image):
        """Test _prepare_prompt creates correct vision-language prompt format."""
        stage = LayoutDetectionStage()
        prompt = stage._prepare_prompt(sample_base64_image)

        # Verify structure
        assert "prompt" in prompt
        assert "multi_modal_data" in prompt
        assert "image" in prompt["multi_modal_data"]

        # Verify content
        assert "<image>" in prompt["prompt"]
        assert "Extract all document objects" in prompt["prompt"]
        assert "JSON array" in prompt["prompt"]
        assert prompt["multi_modal_data"]["image"] == sample_base64_image

    def test_parse_layout_output_valid_json(self):
        """Test _parse_layout_output parses valid JSON."""
        stage = LayoutDetectionStage()

        # Valid JSON output
        output_text = json.dumps(
            [
                {
                    "type": "text",
                    "bbox": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.1},
                    "content": "Sample text",
                },
                {
                    "type": "table",
                    "bbox": {"x": 0.1, "y": 0.4, "width": 0.8, "height": 0.3},
                    "content": "\\begin{tabular}...",
                },
            ]
        )

        result = stage._parse_layout_output(output_text)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "table"
        assert "bbox" in result[0]
        assert "content" in result[0]

    def test_parse_layout_output_with_markdown_wrapper(self):
        """Test _parse_layout_output handles markdown code blocks."""
        stage = LayoutDetectionStage()

        # JSON wrapped in markdown
        json_data = [{"type": "text", "bbox": {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.1}}]
        output_text = f"```json\n{json.dumps(json_data)}\n```"

        result = stage._parse_layout_output(output_text)

        assert len(result) == 1
        assert result[0]["type"] == "text"

    def test_parse_layout_output_with_plain_markdown(self):
        """Test _parse_layout_output handles plain markdown code blocks."""
        stage = LayoutDetectionStage()

        # JSON wrapped in plain markdown
        json_data = [{"type": "image", "bbox": {"x": 0.2, "y": 0.5, "width": 0.6, "height": 0.3}}]
        output_text = f"```\n{json.dumps(json_data)}\n```"

        result = stage._parse_layout_output(output_text)

        assert len(result) == 1
        assert result[0]["type"] == "image"

    def test_parse_layout_output_malformed_json(self):
        """Test _parse_layout_output handles malformed JSON gracefully."""
        stage = LayoutDetectionStage()

        # Invalid JSON
        output_text = '{"invalid": "json", missing bracket'

        result = stage._parse_layout_output(output_text)

        assert result == []  # Should return empty list on error

    def test_parse_layout_output_non_list(self):
        """Test _parse_layout_output handles non-list JSON."""
        stage = LayoutDetectionStage()

        # Valid JSON but not a list
        output_text = json.dumps({"type": "text", "bbox": {}})

        result = stage._parse_layout_output(output_text)

        assert result == []  # Should return empty list for non-list

    def test_process_with_mocked_vllm(self, sample_page_images, mock_vllm_model):
        """Test process with mocked vLLM returns expected layout structure."""
        stage = LayoutDetectionStage()
        stage.model = mock_vllm_model

        # Mock sampling params
        from unittest.mock import MagicMock

        stage.sampling_params = MagicMock()

        # Process
        result = stage.process(sample_page_images)

        # Verify
        assert "layout_objects" in result.to_pandas().columns
        layout_objects_json = result.to_pandas()["layout_objects"].iloc[0]
        layout_objects = json.loads(layout_objects_json)

        assert len(layout_objects) == 1
        assert "page_number" in layout_objects[0]
        assert "width" in layout_objects[0]
        assert "height" in layout_objects[0]
        assert "objects" in layout_objects[0]

    def test_layout_output_contains_bboxes_and_types(self, sample_page_images, mock_vllm_model):
        """Test layout output contains bounding boxes, types, and metadata."""
        stage = LayoutDetectionStage()
        stage.model = mock_vllm_model

        from unittest.mock import MagicMock

        stage.sampling_params = MagicMock()

        # Process
        result = stage.process(sample_page_images)

        # Parse output
        layout_objects_json = result.to_pandas()["layout_objects"].iloc[0]
        layout_objects = json.loads(layout_objects_json)

        # Verify structure
        assert len(layout_objects) > 0
        page_layout = layout_objects[0]

        assert "page_number" in page_layout
        assert "objects" in page_layout

        # Check objects have required fields
        if page_layout["objects"]:
            obj = page_layout["objects"][0]
            assert "type" in obj
            assert "bbox" in obj
            assert "content" in obj

    def test_process_with_empty_page_images(self):
        """Test process with empty page images."""
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

        # Create batch with empty page images
        df = pd.DataFrame({"pdf_path": ["/tmp/test.pdf"], "page_images": [json.dumps([])]})
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        stage = LayoutDetectionStage()
        stage.model = None  # Don't initialize model

        # Process should handle empty gracefully
        result = stage.process(batch)

        layout_objects_json = result.to_pandas()["layout_objects"].iloc[0]
        layout_objects = json.loads(layout_objects_json)
        assert layout_objects == []

    def test_process_exception_handling(self, sample_page_images):
        """Test process handles exceptions gracefully."""
        stage = LayoutDetectionStage()

        # Set model to None to trigger exception
        stage.model = None

        # Process should not crash
        result = stage.process(sample_page_images)

        # Should return empty layout on error
        layout_objects_json = result.to_pandas()["layout_objects"].iloc[0]
        layout_objects = json.loads(layout_objects_json)
        assert layout_objects == []
