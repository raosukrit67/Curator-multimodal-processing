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

"""Tests for LayoutDetectionStage with mocked vLLM."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

from nemo_curator.stages.pdf.layout_detection import LayoutDetectionStage
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.pdf_utils import image_to_base64

MOCK_PARSE_OUTPUT = (
    "<x_0.05><y_0.02>"
    "# Test Title"
    "<x_0.95><y_0.08><class_Title>"
    "<x_0.05><y_0.10>"
    "Body text paragraph here."
    "<x_0.95><y_0.30><class_Text>"
    "<x_0.10><y_0.35>"
    "<x_0.90><y_0.70><class_Picture>"
)


@pytest.fixture
def sample_page_images_json():
    """Create a sample page_images JSON string."""
    img = Image.new("RGB", (800, 1000), color="white")
    b64 = image_to_base64(img)
    page_data = [{
        "page_number": 0,
        "width": 800,
        "height": 1000,
        "image_base64": b64,
    }]
    return json.dumps(page_data)


@pytest.fixture
def mock_vlm_model():
    """Create a mocked VLLMModel."""
    model = MagicMock()
    model.generate.return_value = [MOCK_PARSE_OUTPUT]
    return model


class TestLayoutDetectionStage:
    def test_init_defaults(self):
        stage = LayoutDetectionStage()
        assert stage.model_identifier == "nvidia/NVIDIA-Nemotron-Parse-v1.1"
        assert stage.name == "layout_detection"
        assert stage.temperature == 0.0

    def test_inputs_outputs(self):
        stage = LayoutDetectionStage()
        inputs = stage.inputs()
        outputs = stage.outputs()
        assert inputs == (["data"], ["page_images"])
        assert "layout_regions" in outputs[1]

    @patch("nemo_curator.stages.pdf.layout_detection.VLLMModel")
    def test_process_parses_output(self, mock_vlm_cls, sample_page_images_json, mock_vlm_model):
        """Test that process correctly parses Nemotron Parse output."""
        mock_vlm_cls.return_value = mock_vlm_model

        stage = LayoutDetectionStage()
        stage._model = mock_vlm_model

        df = pd.DataFrame({
            "page_images": [sample_page_images_json],
        })
        batch = DocumentBatch(
            task_id="test_0", dataset_name="test", data=df,
        )

        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "layout_regions" in result_df.columns
        layout = json.loads(result_df["layout_regions"].iloc[0])
        assert len(layout) == 1  # One page

        page = layout[0]
        assert page["page_number"] == 0
        regions = page["regions"]
        assert len(regions) == 3  # Title, Text, Picture

        assert regions[0]["class_name"] == "Title"
        assert regions[1]["class_name"] == "Text"
        assert regions[2]["class_name"] == "Picture"
        assert regions[2]["needs_vl"] is True
        assert regions[0]["needs_vl"] is False

    @patch("nemo_curator.stages.pdf.layout_detection.VLLMModel")
    def test_process_empty_pages(self, mock_vlm_cls):
        """Test handling of empty page images."""
        stage = LayoutDetectionStage()
        stage._model = MagicMock()

        df = pd.DataFrame({
            "page_images": [json.dumps([])],
        })
        batch = DocumentBatch(task_id="test_0", dataset_name="test", data=df)

        result = stage.process(batch)
        result_df = result.to_pandas()
        layout = json.loads(result_df["layout_regions"].iloc[0])
        assert layout == []

    @patch("nemo_curator.stages.pdf.layout_detection.VLLMModel")
    def test_process_handles_model_error(self, mock_vlm_cls, sample_page_images_json):
        """Test graceful handling of model inference errors."""
        mock_model = MagicMock()
        mock_model.generate.side_effect = RuntimeError("GPU OOM")

        stage = LayoutDetectionStage()
        stage._model = mock_model

        df = pd.DataFrame({"page_images": [sample_page_images_json]})
        batch = DocumentBatch(task_id="test_0", dataset_name="test", data=df)

        result = stage.process(batch)
        result_df = result.to_pandas()
        layout = json.loads(result_df["layout_regions"].iloc[0])
        assert layout == []  # Graceful fallback
