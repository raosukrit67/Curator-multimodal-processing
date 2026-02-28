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

"""Stage for converting PDF pages to images."""

import json
from pathlib import Path

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.pdf_utils import image_to_base64, pdf_to_images


class PDFToImageStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Convert PDF pages to images.

    This stage reads PDFs from the specified path column and converts each page
    to a PIL Image at the specified DPI. Images are stored as base64-encoded
    PNG strings in the output.

    Args:
        pdf_path_field: Column name containing PDF file paths (default: "pdf_path")
        dpi: Resolution for rendering PDF pages (default: 300)
        image_format: Format for storing images (default: "PNG")
        output_field: Column name for storing page images (default: "page_images")
    """

    def __init__(
        self,
        pdf_path_field: str = "pdf_path",
        dpi: int = 300,
        image_format: str = "PNG",
        output_field: str = "page_images",
    ):
        self.pdf_path_field = pdf_path_field
        self.dpi = dpi
        self.image_format = image_format
        self.output_field = output_field

        self.name = "pdf_to_image"
        self.resources = Resources(cpus=1.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pdf_path_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pdf_path_field, self.output_field]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        page_images_list = []
        for pdf_path in df[self.pdf_path_field]:
            try:
                # Convert PDF to images
                images = pdf_to_images(pdf_path, dpi=self.dpi)

                # Convert images to base64 and store with metadata
                page_data = []
                for page_num, image in enumerate(images):
                    page_data.append(
                        {
                            "page_number": page_num,
                            "width": image.width,
                            "height": image.height,
                            "image_base64": image_to_base64(image, format=self.image_format),
                        }
                    )

                page_images_list.append(json.dumps(page_data))
                logger.info(f"Converted {len(images)} pages from {Path(pdf_path).name}")

            except Exception as e:  # noqa: BLE001, PERF203
                logger.error(f"Failed to convert PDF {pdf_path}: {e}")
                # Store empty list on failure
                page_images_list.append(json.dumps([]))

        df[self.output_field] = page_images_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
