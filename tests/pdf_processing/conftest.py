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

"""Shared fixtures for PDF processing tests."""

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from PIL import Image

from nemo_curator.tasks import DocumentBatch


@pytest.fixture
def tutorial_pdf_path():
    """Path to tutorial PDF source directory."""
    return Path(__file__).parents[2] / "tutorials" / "pdf_processing" / "source"


@pytest.fixture
def sample_base64_image():
    """Base64-encoded sample image."""
    # Create a simple 100x100 white image
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@pytest.fixture
def sample_pdf_document():
    """Mock DocumentBatch with pdf_path."""
    df = pd.DataFrame({"pdf_path": ["/tmp/sample.pdf"]})
    return DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=df)


@pytest.fixture
def sample_page_images(sample_base64_image):
    """Mock DocumentBatch with page_images JSON."""
    page_data = [
        {
            "page_number": 0,
            "width": 100,
            "height": 100,
            "image_base64": sample_base64_image,
        }
    ]
    df = pd.DataFrame(
        {"pdf_path": ["/tmp/sample.pdf"], "page_images": [json.dumps(page_data)]}
    )
    return DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=df)


@pytest.fixture
def sample_layout_objects():
    """Mock DocumentBatch with layout_objects JSON."""
    layout_data = [
        {
            "page_number": 0,
            "width": 100,
            "height": 100,
            "objects": [
                {
                    "type": "text",
                    "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2},
                    "content": "Sample text content",
                },
                {
                    "type": "table",
                    "bbox": {"x": 0.1, "y": 0.4, "width": 0.8, "height": 0.3},
                    "content": "\\begin{tabular}{ll}\nA & B \\\\\nC & D \\\\\n\\end{tabular}",
                },
                {
                    "type": "image",
                    "bbox": {"x": 0.2, "y": 0.8, "width": 0.6, "height": 0.15},
                    "content": "",
                },
            ],
        }
    ]
    df = pd.DataFrame(
        {
            "pdf_path": ["/tmp/sample.pdf"],
            "page_images": ["[]"],
            "layout_objects": [json.dumps(layout_data)],
        }
    )
    return DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=df)


@pytest.fixture
def sample_cropped_regions(sample_base64_image):
    """Mock DocumentBatch with cropped_regions JSON."""
    cropped_data = [
        {
            "page_number": 0,
            "object_index": 0,
            "type": "text",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2},
            "content": "Sample text content",
            "cropped_image_base64": sample_base64_image,
        },
        {
            "page_number": 0,
            "object_index": 1,
            "type": "table",
            "bbox": {"x": 0.1, "y": 0.4, "width": 0.8, "height": 0.3},
            "content": "\\begin{tabular}{ll}\nA & B \\\\\nC & D \\\\\n\\end{tabular}",
            "cropped_image_base64": sample_base64_image,
        },
        {
            "page_number": 0,
            "object_index": 2,
            "type": "image",
            "bbox": {"x": 0.2, "y": 0.8, "width": 0.6, "height": 0.15},
            "content": "",
            "cropped_image_base64": sample_base64_image,
        },
    ]
    df = pd.DataFrame(
        {
            "pdf_path": ["/tmp/sample.pdf"],
            "page_images": ["[]"],
            "layout_objects": ["[]"],
            "cropped_regions": [json.dumps(cropped_data)],
        }
    )
    return DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=df)


@pytest.fixture
def sample_classified_regions(sample_base64_image):
    """Mock DocumentBatch with classified_regions JSON."""
    classified_data = [
        {
            "page_number": 0,
            "object_index": 0,
            "type": "text",
            "classified_type": "text",
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2},
            "content": "Sample text content",
            "cropped_image_base64": sample_base64_image,
        },
        {
            "page_number": 0,
            "object_index": 1,
            "type": "table",
            "classified_type": "table",
            "bbox": {"x": 0.1, "y": 0.4, "width": 0.8, "height": 0.3},
            "content": "\\begin{tabular}{ll}\nA & B \\\\\nC & D \\\\\n\\end{tabular}",
            "cropped_image_base64": sample_base64_image,
        },
        {
            "page_number": 0,
            "object_index": 2,
            "type": "image",
            "classified_type": "figure",
            "bbox": {"x": 0.2, "y": 0.8, "width": 0.6, "height": 0.15},
            "content": "",
            "cropped_image_base64": sample_base64_image,
        },
    ]
    df = pd.DataFrame(
        {
            "pdf_path": ["/tmp/sample.pdf"],
            "page_images": ["[]"],
            "layout_objects": ["[]"],
            "cropped_regions": ["[]"],
            "classified_regions": [json.dumps(classified_data)],
        }
    )
    return DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=df)


@pytest.fixture
def mock_vllm_model():
    """Mock VLLMModel for testing without GPU."""
    mock_model = MagicMock()

    # Mock generate method to return layout detection results
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock()]
    mock_output.outputs[
        0
    ].text = '[{"type": "text", "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.2}, "content": "Sample text"}]'

    mock_model.generate.return_value = [mock_output]

    return mock_model
