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

"""Stage for extracting cropped regions from images using bounding boxes."""

import json

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.pdf_utils import base64_to_image, crop_image_region, image_to_base64


class BoundingBoxExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract cropped regions from images using bounding boxes.

    This stage takes page images and layout objects (with bounding boxes) and crops
    out each detected region, storing them with metadata for further processing.

    Args:
        page_images_field: Column containing page images (default: "page_images")
        layout_objects_field: Column containing layout detection results (default: "layout_objects")
        output_field: Column for storing cropped regions (default: "cropped_regions")
    """

    def __init__(
        self,
        page_images_field: str = "page_images",
        layout_objects_field: str = "layout_objects",
        output_field: str = "cropped_regions",
    ):
        self.page_images_field = page_images_field
        self.layout_objects_field = layout_objects_field
        self.output_field = output_field

        self.name = "bbox_extraction"
        self.resources = Resources(cpus=4.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.page_images_field, self.layout_objects_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.page_images_field, self.layout_objects_field, self.output_field]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        cropped_regions_list = []

        for idx in range(len(df)):
            try:
                page_images_json = df[self.page_images_field].iloc[idx]
                layout_objects_json = df[self.layout_objects_field].iloc[idx]

                page_images = json.loads(page_images_json)
                layout_data = json.loads(layout_objects_json)

                if not page_images or not layout_data:
                    cropped_regions_list.append(json.dumps([]))
                    continue

                # Create mapping of page_number to page image
                page_image_map = {page["page_number"]: page for page in page_images}

                all_cropped_regions = []

                for page_layout in layout_data:
                    page_number = page_layout["page_number"]

                    if page_number not in page_image_map:
                        logger.warning(f"Page {page_number} not found in page images")
                        continue

                    # Get page image
                    page_data = page_image_map[page_number]
                    page_image = base64_to_image(page_data["image_base64"])

                    # Process each object on this page
                    for obj_idx, obj in enumerate(page_layout.get("objects", [])):
                        if "bbox" not in obj:
                            continue

                        bbox = obj["bbox"]
                        obj_type = obj.get("type", "unknown")

                        try:
                            # Crop region
                            cropped_image = crop_image_region(page_image, bbox, normalized=True)

                            # Store cropped region with metadata
                            region_data = {
                                "page_number": page_number,
                                "object_index": obj_idx,
                                "type": obj_type,
                                "bbox": bbox,
                                "content": obj.get("content", ""),
                                "cropped_image_base64": image_to_base64(cropped_image),
                            }

                            all_cropped_regions.append(region_data)

                        except Exception as e:
                            logger.error(f"Failed to crop region {obj_idx} on page {page_number}: {e}")
                            continue

                cropped_regions_list.append(json.dumps(all_cropped_regions))
                logger.info(f"Extracted {len(all_cropped_regions)} regions")

            except Exception as e:
                logger.error(f"Failed to extract bounding boxes: {e}")
                cropped_regions_list.append(json.dumps([]))

        df[self.output_field] = cropped_regions_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
