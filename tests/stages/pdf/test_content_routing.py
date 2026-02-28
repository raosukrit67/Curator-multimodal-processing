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

"""Tests for ContentRoutingStage."""

import json

import pandas as pd
import pytest
from PIL import Image

from nemo_curator.stages.pdf.content_routing import ContentRoutingStage
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.pdf_utils import image_to_base64


@pytest.fixture
def sample_data():
    """Create sample layout_regions and page_images for testing."""
    img = Image.new("RGB", (100, 200), color="white")
    b64 = image_to_base64(img)

    page_images = [{
        "page_number": 0,
        "width": 100,
        "height": 200,
        "image_base64": b64,
    }]

    layout_regions = [{
        "page_number": 0,
        "width": 100,
        "height": 200,
        "regions": [
            {"class_name": "Title", "bbox": (5, 2, 95, 15), "text": "# Title"},
            {"class_name": "Text", "bbox": (5, 20, 95, 80), "text": "Body text."},
            {"class_name": "Picture", "bbox": (10, 90, 90, 180), "text": ""},
            {"class_name": "Table", "bbox": (5, 82, 95, 88), "text": "\\begin{tabular}..."},
        ],
    }]

    return {
        "layout_regions": json.dumps(layout_regions),
        "page_images": json.dumps(page_images),
    }


class TestContentRoutingStage:
    def test_routes_text_and_visual(self, sample_data):
        stage = ContentRoutingStage()

        df = pd.DataFrame({
            "layout_regions": [sample_data["layout_regions"]],
            "page_images": [sample_data["page_images"]],
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        result_df = result.to_pandas()
        routed = json.loads(result_df["routed_content"].iloc[0])

        assert len(routed) == 1
        page = routed[0]

        # Title, Text, Table should be text regions
        assert len(page["text_regions"]) == 3
        # Picture should be VL region
        assert len(page["vl_regions"]) == 1
        assert page["vl_regions"][0]["class_name"] == "Picture"
        assert "cropped_image_base64" in page["vl_regions"][0]

    def test_include_tables_for_vl(self, sample_data):
        stage = ContentRoutingStage(include_tables_for_vl=True)

        df = pd.DataFrame({
            "layout_regions": [sample_data["layout_regions"]],
            "page_images": [sample_data["page_images"]],
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        routed = json.loads(result.to_pandas()["routed_content"].iloc[0])

        page = routed[0]
        # With include_tables_for_vl, Table also goes to VL
        vl_classes = [r["class_name"] for r in page["vl_regions"]]
        assert "Picture" in vl_classes
        assert "Table" in vl_classes

    def test_empty_layout(self):
        stage = ContentRoutingStage()

        df = pd.DataFrame({
            "layout_regions": [json.dumps([])],
            "page_images": [json.dumps([])],
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        routed = json.loads(result.to_pandas()["routed_content"].iloc[0])
        assert routed == []
