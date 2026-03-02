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

"""Tests for TextAssemblyStage."""

import json

import pandas as pd
import pytest

from nemo_curator.stages.pdf.postprocess import TextAssemblyStage
from nemo_curator.tasks import DocumentBatch


@pytest.fixture
def sample_routed_and_analysis():
    """Create sample routed content and VL analysis results."""
    routed = [{
        "page_number": 0,
        "text_regions": [
            {"class_name": "Title", "bbox": [5, 2, 95, 15], "text": "# Title"},
            {"class_name": "Text", "bbox": [5, 20, 95, 80], "text": "Body text here."},
        ],
        "vl_regions": [
            {
                "class_name": "Picture",
                "bbox": [10, 90, 90, 180],
                "text": "",
                "cropped_image_base64": "abc123",
            },
        ],
    }]

    analyses = [{
        "page_number": 0,
        "class_name": "Picture",
        "bbox": [10, 90, 90, 180],
        "description": "A diagram showing the NeMo Curator architecture.",
    }]

    return {
        "routed_content": json.dumps(routed),
        "analysis_results": json.dumps(analyses),
    }


class TestTextAssemblyStage:
    def test_assembles_all_modalities(self, sample_routed_and_analysis):
        stage = TextAssemblyStage()

        df = pd.DataFrame({
            "routed_content": [sample_routed_and_analysis["routed_content"]],
            "analysis_results": [sample_routed_and_analysis["analysis_results"]],
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        pages = json.loads(result.to_pandas()["pages"].iloc[0])

        assert len(pages) == 1

        page = pages[0]
        assert page["page_number"] == 0

        # Text blocks
        assert len(page["text_blocks"]) == 2
        assert page["text_blocks"][0]["class_name"] == "Title"
        assert page["text_blocks"][1]["text"] == "Body text here."

        # Figures with VL description
        assert len(page["figures"]) == 1
        assert "architecture" in page["figures"][0]["description"]

        # Full text should contain all content
        assert "Title" in page["full_text"]
        assert "Body text" in page["full_text"]
        assert "architecture" in page["full_text"]

    def test_empty_input(self):
        stage = TextAssemblyStage()

        df = pd.DataFrame({
            "routed_content": [json.dumps([])],
            "analysis_results": [json.dumps([])],
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        pages = json.loads(result.to_pandas()["pages"].iloc[0])
        assert pages == []

    def test_no_vl_results(self):
        """Assembly should work even when VL model returned no results."""
        stage = TextAssemblyStage()

        routed = [{
            "page_number": 0,
            "text_regions": [
                {"class_name": "Text", "bbox": [5, 5, 95, 50], "text": "Some text."},
            ],
            "vl_regions": [
                {"class_name": "Picture", "bbox": [10, 60, 90, 90], "text": ""},
            ],
        }]

        df = pd.DataFrame({
            "routed_content": [json.dumps(routed)],
            "analysis_results": [json.dumps([])],  # No VL results
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        pages = json.loads(result.to_pandas()["pages"].iloc[0])

        page = pages[0]
        assert len(page["text_blocks"]) == 1
        assert len(page["figures"]) == 1
        assert page["figures"][0]["description"] == ""  # No VL description

    def test_produces_text_column(self, sample_routed_and_analysis):
        """TextAssemblyStage should produce a flat 'text' column for dedup."""
        stage = TextAssemblyStage()

        df = pd.DataFrame({
            "pdf_path": ["test.pdf"],
            "routed_content": [sample_routed_and_analysis["routed_content"]],
            "analysis_results": [sample_routed_and_analysis["analysis_results"]],
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "text" in result_df.columns
        text = result_df["text"].iloc[0]
        assert "Title" in text
        assert "Body text" in text
        assert "architecture" in text

    def test_produces_pages_column(self, sample_routed_and_analysis):
        """TextAssemblyStage should produce a 'pages' column (JSON string)."""
        stage = TextAssemblyStage()

        df = pd.DataFrame({
            "pdf_path": ["test.pdf"],
            "routed_content": [sample_routed_and_analysis["routed_content"]],
            "analysis_results": [sample_routed_and_analysis["analysis_results"]],
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "pages" in result_df.columns
        pages = json.loads(result_df["pages"].iloc[0])
        assert len(pages) == 1
        assert pages[0]["page_number"] == 0

    def test_drops_intermediate_columns(self, sample_routed_and_analysis):
        """TextAssemblyStage should drop intermediate columns."""
        stage = TextAssemblyStage()

        df = pd.DataFrame({
            "pdf_path": ["test.pdf"],
            "page_images": ["[img_data]"],
            "layout_regions": ["[layout_data]"],
            "routed_content": [sample_routed_and_analysis["routed_content"]],
            "analysis_results": [sample_routed_and_analysis["analysis_results"]],
        })
        batch = DocumentBatch(task_id="test", dataset_name="test", data=df)

        result = stage.process(batch)
        result_df = result.to_pandas()

        # Intermediate columns should be dropped
        assert "page_images" not in result_df.columns
        assert "layout_regions" not in result_df.columns
        assert "routed_content" not in result_df.columns
        assert "analysis_results" not in result_df.columns
        assert "assembled_content" not in result_df.columns

        # Clean output columns should remain
        assert "pdf_path" in result_df.columns
        assert "pages" in result_df.columns
        assert "text" in result_df.columns
