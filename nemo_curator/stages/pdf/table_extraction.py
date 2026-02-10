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

"""Stage for extracting tables from classified regions."""

import json
import re

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class TableExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract tables from classified regions and convert to HTML.

    This stage processes regions classified as tables, extracting structure
    from LaTeX content if available or using OCR/image analysis as fallback.
    Tables are converted to HTML format for easy consumption.

    Args:
        classified_regions_field: Column containing classified regions (default: "classified_regions")
        output_field: Column for storing extracted tables (default: "extracted_tables")
    """

    def __init__(
        self,
        classified_regions_field: str = "classified_regions",
        output_field: str = "extracted_tables",
    ):
        self.classified_regions_field = classified_regions_field
        self.output_field = output_field

        self.name = "table_extraction"
        self.resources = Resources(cpus=4.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.classified_regions_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.classified_regions_field, self.output_field]

    def _latex_to_html(self, latex_content: str) -> str | None:
        """Convert LaTeX table to HTML.

        Args:
            latex_content: LaTeX table content

        Returns:
            HTML table string or None if conversion fails
        """
        try:
            # Extract tabular environment
            tabular_match = re.search(r"\\begin{tabular}{.*?}(.*?)\\end{tabular}", latex_content, re.DOTALL)

            if not tabular_match:
                return None

            table_content = tabular_match.group(1)

            # Split into rows
            rows = [row.strip() for row in table_content.split("\\\\") if row.strip()]

            # Build HTML table
            html_parts = ["<table>"]

            for i, row in enumerate(rows):
                # Remove \hline and other commands
                row = re.sub(r"\\hline", "", row)
                row = row.strip()

                if not row:
                    continue

                # Split into cells
                cells = [cell.strip() for cell in row.split("&")]

                # First row is typically header
                tag = "th" if i == 0 else "td"
                html_parts.append("  <tr>")
                for cell in cells:
                    html_parts.append(f"    <{tag}>{cell}</{tag}>")
                html_parts.append("  </tr>")

            html_parts.append("</table>")

            return "\n".join(html_parts)

        except Exception as e:
            logger.error(f"Failed to convert LaTeX to HTML: {e}")
            return None

    def _extract_table_from_image(self, region: dict) -> str | None:
        """Extract table from image using OCR or other methods.

        Args:
            region: Region dictionary with cropped image

        Returns:
            HTML table string or None if extraction fails
        """
        # Placeholder for image-based table extraction
        # Could use OCR + table structure recognition
        # For now, return a simple representation
        logger.warning("Image-based table extraction not yet implemented")
        return None

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        extracted_tables_list = []

        for classified_regions_json in df[self.classified_regions_field]:
            try:
                classified_regions = json.loads(classified_regions_json)

                if not classified_regions:
                    extracted_tables_list.append(json.dumps([]))
                    continue

                tables = []

                for region in classified_regions:
                    # Only process table regions
                    if region.get("classified_type") != "table":
                        continue

                    content = region.get("content", "")
                    html_table = None

                    # Try LaTeX conversion first
                    if "\\begin{tabular}" in content:
                        html_table = self._latex_to_html(content)

                    # Fallback to image-based extraction
                    if html_table is None and "cropped_image_base64" in region:
                        html_table = self._extract_table_from_image(region)

                    if html_table:
                        tables.append(
                            {
                                "page_number": region["page_number"],
                                "object_index": region["object_index"],
                                "bbox": region["bbox"],
                                "html": html_table,
                            }
                        )

                extracted_tables_list.append(json.dumps(tables))
                logger.info(f"Extracted {len(tables)} tables")

            except Exception as e:
                logger.error(f"Failed to extract tables: {e}")
                extracted_tables_list.append(json.dumps([]))

        df[self.output_field] = extracted_tables_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
