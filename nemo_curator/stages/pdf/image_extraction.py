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

"""Stage for extracting and saving images from classified regions."""

import json

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class ImageExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract images from classified regions.

    This stage processes regions classified as images/figures/charts and stores
    them with metadata for further processing or analysis.

    Args:
        classified_regions_field: Column containing classified regions (default: "classified_regions")
        output_field: Column for storing extracted images (default: "extracted_images")
        image_types: Types of regions to extract as images (default: ["image", "figure", "chart"])
    """

    def __init__(
        self,
        classified_regions_field: str = "classified_regions",
        output_field: str = "extracted_images",
        image_types: list[str] | None = None,
    ):
        self.classified_regions_field = classified_regions_field
        self.output_field = output_field
        self.image_types = image_types or ["image", "figure", "chart"]

        self.name = "image_extraction"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.classified_regions_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.classified_regions_field, self.output_field]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        extracted_images_list = []

        for classified_regions_json in df[self.classified_regions_field]:
            try:
                classified_regions = json.loads(classified_regions_json)

                if not classified_regions:
                    extracted_images_list.append(json.dumps([]))
                    continue

                images = []

                for region in classified_regions:
                    # Only process image-type regions
                    classified_type = region.get("classified_type", "")
                    if classified_type not in self.image_types:
                        continue

                    # Store image with metadata
                    if "cropped_image_base64" in region:
                        images.append(
                            {
                                "page_number": region["page_number"],
                                "object_index": region["object_index"],
                                "type": classified_type,
                                "bbox": region["bbox"],
                                "image_base64": region["cropped_image_base64"],
                            }
                        )

                extracted_images_list.append(json.dumps(images))
                logger.info(f"Extracted {len(images)} images")

            except Exception as e:
                logger.error(f"Failed to extract images: {e}")
                extracted_images_list.append(json.dumps([]))

        df[self.output_field] = extracted_images_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
