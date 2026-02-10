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

"""Stage for classifying content types from cropped regions."""

import json

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class ContentTypeClassificationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Classify content types from cropped regions.

    This stage refines the content type classification of cropped regions using
    heuristics based on size, aspect ratio, and content. It adds a 'classified_type'
    field that may differ from the initial 'type' from layout detection.

    Args:
        cropped_regions_field: Column containing cropped regions (default: "cropped_regions")
        output_field: Column for storing classified regions (default: "classified_regions")
        min_table_aspect_ratio: Minimum aspect ratio for table detection (default: 1.5)
        min_text_area: Minimum area for text blocks (default: 0.01)
    """

    def __init__(
        self,
        cropped_regions_field: str = "cropped_regions",
        output_field: str = "classified_regions",
        min_table_aspect_ratio: float = 1.5,
        min_text_area: float = 0.01,
    ):
        self.cropped_regions_field = cropped_regions_field
        self.output_field = output_field
        self.min_table_aspect_ratio = min_table_aspect_ratio
        self.min_text_area = min_text_area

        self.name = "content_classification"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.cropped_regions_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.cropped_regions_field, self.output_field]

    def _classify_region(self, region: dict) -> str:
        """Classify a region based on heuristics.

        Args:
            region: Region dictionary with bbox and type

        Returns:
            Classified type (text|table|image|figure|chart)
        """
        # Start with original type from layout detection
        original_type = region.get("type", "unknown")
        bbox = region.get("bbox", {})

        # Calculate region properties
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)
        area = width * height
        aspect_ratio = width / height if height > 0 else 0

        # Check if region has LaTeX content (likely table)
        content = region.get("content", "")
        has_latex = "\\begin{tabular}" in content or "\\hline" in content

        # Classification heuristics
        if has_latex or original_type == "table":
            return "table"

        # Tables often have wider aspect ratios
        if aspect_ratio > self.min_table_aspect_ratio and area > 0.05:
            return "table"

        # Small regions are likely text
        if area < self.min_text_area:
            return "text"

        # Square/portrait regions with significant area are likely images
        if 0.5 < aspect_ratio < 2.0 and area > 0.1:
            if original_type in ["image", "figure", "chart"]:
                return original_type
            return "image"

        # Default to original type or text
        if original_type in ["text", "table", "image", "figure", "chart"]:
            return original_type

        return "text"

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        classified_regions_list = []

        for cropped_regions_json in df[self.cropped_regions_field]:
            try:
                cropped_regions = json.loads(cropped_regions_json)

                if not cropped_regions:
                    classified_regions_list.append(json.dumps([]))
                    continue

                classified_regions = []

                for region in cropped_regions:
                    # Classify region
                    classified_type = self._classify_region(region)

                    # Add classification to region
                    region["classified_type"] = classified_type
                    classified_regions.append(region)

                classified_regions_list.append(json.dumps(classified_regions))

                # Log classification statistics
                type_counts = {}
                for r in classified_regions:
                    t = r["classified_type"]
                    type_counts[t] = type_counts.get(t, 0) + 1

                logger.info(f"Classified {len(classified_regions)} regions: {type_counts}")

            except Exception as e:
                logger.error(f"Failed to classify content types: {e}")
                classified_regions_list.append(json.dumps([]))

        df[self.output_field] = classified_regions_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
