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

"""Tests for DeepAnalysisStage."""

import json

import pytest

from nemo_curator.stages.pdf.deep_analysis import DeepAnalysisStage


class TestDeepAnalysisStage:
    """Test cases for DeepAnalysisStage."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = DeepAnalysisStage()
        assert stage.model_identifier == "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
        assert stage.classified_regions_field == "classified_regions"
        assert stage.output_field == "analysis_results"
        assert stage.max_tokens == 1024
        assert stage.temperature == 0.2
        assert stage.top_p == 0.7
        assert stage.analyze_types == ["table", "image", "figure", "chart"]
        assert stage.name == "deep_analysis"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        stage = DeepAnalysisStage(
            model_identifier="custom/model",
            classified_regions_field="custom_classified",
            output_field="custom_analysis",
            max_tokens=512,
            temperature=0.5,
            top_p=0.9,
            analyze_types=["table", "chart"],
        )
        assert stage.model_identifier == "custom/model"
        assert stage.classified_regions_field == "custom_classified"
        assert stage.output_field == "custom_analysis"
        assert stage.max_tokens == 512
        assert stage.temperature == 0.5
        assert stage.top_p == 0.9
        assert stage.analyze_types == ["table", "chart"]

    def test_inputs_outputs(self):
        """Test inputs and outputs configuration."""
        stage = DeepAnalysisStage()
        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["classified_regions"])
        assert outputs == (["data"], ["classified_regions", "analysis_results"])

    def test_prepare_analysis_prompt_for_table(self, sample_base64_image):
        """Test _prepare_analysis_prompt creates type-specific prompts for tables."""
        stage = DeepAnalysisStage()

        region = {
            "classified_type": "table",
            "cropped_image_base64": sample_base64_image,
        }

        prompt = stage._prepare_analysis_prompt(region)

        # Verify structure
        assert "prompt" in prompt
        assert "multi_modal_data" in prompt
        assert "image" in prompt["multi_modal_data"]

        # Verify table-specific content
        assert "<image>" in prompt["prompt"]
        assert "table" in prompt["prompt"].lower()
        assert "structure" in prompt["prompt"].lower()
        assert prompt["multi_modal_data"]["image"] == sample_base64_image

    def test_prepare_analysis_prompt_for_image(self, sample_base64_image):
        """Test _prepare_analysis_prompt creates type-specific prompts for images."""
        stage = DeepAnalysisStage()

        region = {
            "classified_type": "image",
            "cropped_image_base64": sample_base64_image,
        }

        prompt = stage._prepare_analysis_prompt(region)

        # Verify image-specific content
        assert "figure/image" in prompt["prompt"].lower() or "image" in prompt["prompt"].lower()
        assert "description" in prompt["prompt"].lower()

    def test_prepare_analysis_prompt_for_figure(self, sample_base64_image):
        """Test _prepare_analysis_prompt creates type-specific prompts for figures."""
        stage = DeepAnalysisStage()

        region = {
            "classified_type": "figure",
            "cropped_image_base64": sample_base64_image,
        }

        prompt = stage._prepare_analysis_prompt(region)

        # Verify figure-specific content
        assert "figure" in prompt["prompt"].lower() or "image" in prompt["prompt"].lower()

    def test_prepare_analysis_prompt_for_chart(self, sample_base64_image):
        """Test _prepare_analysis_prompt creates type-specific prompts for charts."""
        stage = DeepAnalysisStage()

        region = {
            "classified_type": "chart",
            "cropped_image_base64": sample_base64_image,
        }

        prompt = stage._prepare_analysis_prompt(region)

        # Verify chart-specific content
        assert "chart" in prompt["prompt"].lower()
        assert "trends" in prompt["prompt"].lower() or "patterns" in prompt["prompt"].lower()

    def test_process_with_mocked_vllm(self, sample_classified_regions, mock_vllm_model):
        """Test process with mocked vLLM produces analysis results."""
        stage = DeepAnalysisStage()
        stage.model = mock_vllm_model

        # Mock sampling params
        from unittest.mock import MagicMock

        stage.sampling_params = MagicMock()

        # Mock output for analysis
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "This is a detailed analysis of the content."
        mock_vllm_model.generate.return_value = [mock_output]

        # Process
        result = stage.process(sample_classified_regions)

        # Verify
        assert "analysis_results" in result.to_pandas().columns
        analysis_results_json = result.to_pandas()["analysis_results"].iloc[0]
        analysis_results = json.loads(analysis_results_json)

        # Should analyze table and figure regions (not text)
        assert len(analysis_results) == 2  # table and figure

    def test_analysis_output_structure(self, sample_classified_regions, mock_vllm_model):
        """Test analysis output structure contains type, bbox, and analysis text."""
        stage = DeepAnalysisStage()
        stage.model = mock_vllm_model

        from unittest.mock import MagicMock

        stage.sampling_params = MagicMock()

        # Mock output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Analysis text here"
        mock_vllm_model.generate.return_value = [mock_output]

        # Process
        result = stage.process(sample_classified_regions)

        analysis_results_json = result.to_pandas()["analysis_results"].iloc[0]
        analysis_results = json.loads(analysis_results_json)

        if len(analysis_results) > 0:
            analysis = analysis_results[0]

            # Verify required fields
            assert "page_number" in analysis
            assert "object_index" in analysis
            assert "type" in analysis
            assert "bbox" in analysis
            assert "analysis" in analysis

            # Verify bbox structure
            bbox = analysis["bbox"]
            assert "x" in bbox
            assert "y" in bbox
            assert "width" in bbox
            assert "height" in bbox

    def test_process_filters_by_analyze_types(self, sample_base64_image):
        """Test process only analyzes specified content types."""
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

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
                "classified_type": "table",
                "bbox": {"x": 0.1, "y": 0.3, "width": 0.8, "height": 0.3},
                "cropped_image_base64": sample_base64_image,
            },
            {
                "page_number": 0,
                "object_index": 2,
                "classified_type": "image",
                "bbox": {"x": 0.1, "y": 0.6, "width": 0.8, "height": 0.3},
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

        # Only analyze tables
        stage = DeepAnalysisStage(analyze_types=["table"])
        stage.model = MagicMock()
        stage.sampling_params = MagicMock()

        from unittest.mock import MagicMock

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "Table analysis"
        stage.model.generate.return_value = [mock_output]

        result = stage.process(batch)

        analysis_results_json = result.to_pandas()["analysis_results"].iloc[0]
        analysis_results = json.loads(analysis_results_json)

        # Should only analyze table
        assert len(analysis_results) == 1
        assert analysis_results[0]["type"] == "table"

    def test_process_skips_regions_without_image(self, sample_base64_image):
        """Test process skips regions without cropped_image_base64."""
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

        classified_data = [
            {
                "page_number": 0,
                "object_index": 0,
                "classified_type": "table",
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

        stage = DeepAnalysisStage()
        stage.model = MagicMock()
        stage.sampling_params = MagicMock()

        result = stage.process(batch)

        # Should skip region without image
        analysis_results_json = result.to_pandas()["analysis_results"].iloc[0]
        analysis_results = json.loads(analysis_results_json)
        assert len(analysis_results) == 0

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

        stage = DeepAnalysisStage()
        stage.model = None

        result = stage.process(batch)

        analysis_results_json = result.to_pandas()["analysis_results"].iloc[0]
        analysis_results = json.loads(analysis_results_json)
        assert analysis_results == []

    def test_process_exception_handling(self, sample_classified_regions):
        """Test process handles exceptions gracefully."""
        stage = DeepAnalysisStage()
        stage.model = None  # Will trigger exception

        result = stage.process(sample_classified_regions)

        # Should return empty on error
        analysis_results_json = result.to_pandas()["analysis_results"].iloc[0]
        analysis_results = json.loads(analysis_results_json)
        assert analysis_results == []
