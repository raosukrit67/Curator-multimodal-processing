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

"""Stage for routing detected content regions to appropriate processing paths.

After layout detection, regions are classified into types that determine their
processing path:
- Text/Title/Section/etc. → content already extracted by Parse, ready to use
- Picture/Figure/Chart → needs VL model for visual description
- Table → LaTeX already extracted by Parse, optionally sent to VL for interpretation

This stage also crops image regions for content that needs VL processing.
"""

import json

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.pdf_utils import (
    VL_CONTENT_TYPES,
    base64_to_image,
    crop_image_from_bbox_tuple,
    image_to_base64,
)


class ContentRoutingStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Route content regions and crop images for VL processing.

    Examines each region from layout detection and:
    1. Separates regions into text (already extracted) and visual (needs VL)
    2. Crops image regions for visual content to pass to the VL model
    3. Stores cropped images as base64 for the visual analysis stage

    Args:
        layout_regions_field: Column with layout detection output.
        page_images_field: Column with page images (for cropping).
        output_field: Column for storing routed content with cropped images.
        vl_content_types: Set of class names to route to VL model.
        include_tables_for_vl: Whether to also send tables to VL for interpretation.
    """

    def __init__(
        self,
        layout_regions_field: str = "layout_regions",
        page_images_field: str = "page_images",
        output_field: str = "routed_content",
        vl_content_types: set[str] | None = None,
        include_tables_for_vl: bool = False,
    ):
        self.layout_regions_field = layout_regions_field
        self.page_images_field = page_images_field
        self.output_field = output_field
        self.vl_content_types = vl_content_types or VL_CONTENT_TYPES.copy()
        self.include_tables_for_vl = include_tables_for_vl

        if self.include_tables_for_vl:
            self.vl_content_types.add("Table")

        self.name = "content_routing"
        self.resources = Resources(cpus=1.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.layout_regions_field, self.page_images_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.layout_regions_field,
            self.page_images_field,
            self.output_field,
        ]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        routed_content_list = []

        for layout_json, page_images_json in zip(
            df[self.layout_regions_field], df[self.page_images_field], strict=True
        ):
            try:
                page_layouts = json.loads(layout_json)
                page_images = json.loads(page_images_json)

                if not page_layouts:
                    routed_content_list.append(json.dumps([]))
                    continue

                # Build page image lookup
                page_image_map = {}
                for page_data in page_images:
                    page_num = page_data["page_number"]
                    page_image_map[page_num] = base64_to_image(
                        page_data["image_base64"]
                    )

                routed_pages = []
                for page_layout in page_layouts:
                    page_num = page_layout["page_number"]
                    page_image = page_image_map.get(page_num)

                    text_regions = []
                    vl_regions = []

                    for region in page_layout.get("regions", []):
                        cls = region["class_name"]
                        entry = {
                            "class_name": cls,
                            "bbox": region["bbox"],
                            "text": region.get("text", ""),
                        }

                        if cls in self.vl_content_types and page_image is not None:
                            # Crop the image region for VL processing
                            cropped = crop_image_from_bbox_tuple(
                                page_image, tuple(region["bbox"])
                            )
                            entry["cropped_image_base64"] = image_to_base64(cropped)
                            vl_regions.append(entry)
                        else:
                            text_regions.append(entry)

                    routed_pages.append({
                        "page_number": page_num,
                        "text_regions": text_regions,
                        "vl_regions": vl_regions,
                    })

                routed_content_list.append(json.dumps(routed_pages))

                total_text = sum(len(p["text_regions"]) for p in routed_pages)
                total_vl = sum(len(p["vl_regions"]) for p in routed_pages)
                logger.info(
                    f"Routed {total_text} text regions, "
                    f"{total_vl} visual regions for VL processing"
                )

            except Exception as e:
                logger.error(f"Content routing failed: {e}")
                routed_content_list.append(json.dumps([]))

        df[self.output_field] = routed_content_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
