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

"""Tests for TableExtractionStage."""

import json

import pytest

from nemo_curator.stages.pdf.table_extraction import TableExtractionStage


class TestTableExtractionStage:
    """Test cases for TableExtractionStage."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        stage = TableExtractionStage()
        assert stage.classified_regions_field == "classified_regions"
        assert stage.output_field == "extracted_tables"
        assert stage.name == "table_extraction"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        stage = TableExtractionStage(classified_regions_field="custom_classified", output_field="custom_tables")
        assert stage.classified_regions_field == "custom_classified"
        assert stage.output_field == "custom_tables"

    def test_inputs_outputs(self):
        """Test inputs and outputs configuration."""
        stage = TableExtractionStage()
        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["classified_regions"])
        assert outputs == (["data"], ["classified_regions", "extracted_tables"])

    def test_latex_to_html_simple_table(self):
        """Test _latex_to_html converts simple LaTeX tables to HTML correctly."""
        stage = TableExtractionStage()

        latex_content = r"""
        \begin{tabular}{ll}
        Header1 & Header2 \\
        Row1Col1 & Row1Col2 \\
        Row2Col1 & Row2Col2 \\
        \end{tabular}
        """

        html = stage._latex_to_html(latex_content)

        assert html is not None
        assert "<table>" in html
        assert "</table>" in html
        assert "<th>Header1</th>" in html
        assert "<th>Header2</th>" in html
        assert "<td>Row1Col1</td>" in html
        assert "<td>Row2Col1</td>" in html

    def test_latex_to_html_with_hline(self):
        """Test _latex_to_html handles \\hline correctly."""
        stage = TableExtractionStage()

        latex_content = r"""
        \begin{tabular}{cc}
        \hline
        A & B \\
        \hline
        C & D \\
        \hline
        \end{tabular}
        """

        html = stage._latex_to_html(latex_content)

        assert html is not None
        assert "<table>" in html
        assert "<th>A</th>" in html
        assert "<td>C</td>" in html
        # \hline should be removed
        assert "hline" not in html

    def test_latex_to_html_malformed_latex(self):
        """Test _latex_to_html handles malformed LaTeX gracefully."""
        stage = TableExtractionStage()

        # Missing \end{tabular}
        latex_content = r"\begin{tabular}{cc} A & B \\"

        html = stage._latex_to_html(latex_content)
        assert html is None  # Should return None on error

    def test_latex_to_html_no_tabular_environment(self):
        """Test _latex_to_html returns None for content without tabular."""
        stage = TableExtractionStage()

        latex_content = "Just some text without tabular environment"

        html = stage._latex_to_html(latex_content)
        assert html is None

    def test_latex_to_html_complex_table(self):
        """Test _latex_to_html handles more complex LaTeX structures."""
        stage = TableExtractionStage()

        latex_content = r"""
        \begin{tabular}{|c|c|c|}
        \hline
        Col1 & Col2 & Col3 \\
        \hline
        Data1 & Data2 & Data3 \\
        Data4 & Data5 & Data6 \\
        \hline
        \end{tabular}
        """

        html = stage._latex_to_html(latex_content)

        assert html is not None
        assert "<table>" in html
        assert "<th>Col1</th>" in html
        assert "<td>Data1</td>" in html
        # Should have 3 rows (1 header + 2 data)
        assert html.count("<tr>") == 3

    def test_process_extracts_table_regions(self, sample_classified_regions):
        """Test process extracts table regions and produces HTML output."""
        stage = TableExtractionStage()
        result = stage.process(sample_classified_regions)

        # Verify
        assert "extracted_tables" in result.to_pandas().columns
        extracted_tables_json = result.to_pandas()["extracted_tables"].iloc[0]
        extracted_tables = json.loads(extracted_tables_json)

        # Should have extracted the table
        assert len(extracted_tables) == 1
        table = extracted_tables[0]

        assert table["page_number"] == 0
        assert table["object_index"] == 1
        assert "bbox" in table
        assert "html" in table
        assert "<table>" in table["html"]

    def test_process_filters_non_table_regions(self, sample_classified_regions):
        """Test process only extracts from table-classified regions."""
        stage = TableExtractionStage()
        result = stage.process(sample_classified_regions)

        # Get classified regions
        classified_regions_json = sample_classified_regions.to_pandas()["classified_regions"].iloc[0]
        classified_regions = json.loads(classified_regions_json)

        # Count table regions
        table_regions = [r for r in classified_regions if r.get("classified_type") == "table"]

        # Extracted tables should match table regions count
        extracted_tables_json = result.to_pandas()["extracted_tables"].iloc[0]
        extracted_tables = json.loads(extracted_tables_json)

        assert len(extracted_tables) == len(table_regions)

    def test_extract_table_from_image_not_implemented(self):
        """Test _extract_table_from_image returns None (not yet implemented)."""
        stage = TableExtractionStage()

        region = {"cropped_image_base64": "base64string"}

        result = stage._extract_table_from_image(region)
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

        stage = TableExtractionStage()
        result = stage.process(batch)

        extracted_tables_json = result.to_pandas()["extracted_tables"].iloc[0]
        extracted_tables = json.loads(extracted_tables_json)
        assert extracted_tables == []

    def test_process_output_structure(self, sample_classified_regions):
        """Test output structure of extracted tables."""
        stage = TableExtractionStage()
        result = stage.process(sample_classified_regions)

        extracted_tables_json = result.to_pandas()["extracted_tables"].iloc[0]
        extracted_tables = json.loads(extracted_tables_json)

        if len(extracted_tables) > 0:
            table = extracted_tables[0]

            # Verify required fields
            assert "page_number" in table
            assert "object_index" in table
            assert "bbox" in table
            assert "html" in table

            # Verify bbox structure
            bbox = table["bbox"]
            assert "x" in bbox
            assert "y" in bbox
            assert "width" in bbox
            assert "height" in bbox

            # Verify HTML is valid
            assert isinstance(table["html"], str)
            assert "<table>" in table["html"]

    def test_process_with_region_without_latex(self, sample_base64_image):
        """Test process with table region without LaTeX content."""
        import pandas as pd

        from nemo_curator.tasks import DocumentBatch

        # Table region without LaTeX
        classified_data = [
            {
                "page_number": 0,
                "object_index": 0,
                "type": "table",
                "classified_type": "table",
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.3},
                "content": "No LaTeX here",  # No LaTeX content
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

        stage = TableExtractionStage()
        result = stage.process(batch)

        # Should skip table without LaTeX (image extraction not implemented)
        extracted_tables_json = result.to_pandas()["extracted_tables"].iloc[0]
        extracted_tables = json.loads(extracted_tables_json)
        assert len(extracted_tables) == 0

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

        stage = TableExtractionStage()
        result = stage.process(batch)

        extracted_tables_json = result.to_pandas()["extracted_tables"].iloc[0]
        extracted_tables = json.loads(extracted_tables_json)
        assert extracted_tables == []
