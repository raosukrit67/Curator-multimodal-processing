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

"""Stage for extracting text from classified regions."""

import json

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class TextExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Extract text from classified regions.

    This stage processes regions classified as text, extracting text content
    from the layout detection results or using OCR as a fallback.

    Args:
        classified_regions_field: Column containing classified regions (default: "classified_regions")
        output_field: Column for storing extracted text (default: "extracted_text")
        use_ocr: Whether to use OCR for regions without content (default: False)
    """

    def __init__(
        self,
        classified_regions_field: str = "classified_regions",
        output_field: str = "extracted_text",
        use_ocr: bool = False,
    ):
        self.classified_regions_field = classified_regions_field
        self.output_field = output_field
        self.use_ocr = use_ocr

        self.name = "text_extraction"
        # Allocate GPU memory if OCR is enabled
        if use_ocr:
            self.resources = Resources(cpus=4.0, gpus=0.25, gpu_mem_gb=4.0)
        else:
            self.resources = Resources(cpus=4.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.classified_regions_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.classified_regions_field, self.output_field]

    def _extract_text_with_ocr(self, region: dict) -> str | None:
        """Extract text from image using OCR.

        Args:
            region: Region dictionary with cropped image

        Returns:
            Extracted text or None if OCR fails
        """
        if not self.use_ocr:
            return None

        try:
            import pytesseract

            from nemo_curator.utils.pdf_utils import base64_to_image

            # Get image
            if "cropped_image_base64" not in region:
                return None

            image = base64_to_image(region["cropped_image_base64"])

            # Run OCR
            text = pytesseract.image_to_string(image)
            return text.strip() if text else None

        except ImportError:
            logger.warning("pytesseract not available for OCR. Install with: pip install pytesseract")
            return None
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        extracted_text_list = []

        for classified_regions_json in df[self.classified_regions_field]:
            try:
                classified_regions = json.loads(classified_regions_json)

                if not classified_regions:
                    extracted_text_list.append(json.dumps([]))
                    continue

                text_blocks = []

                for region in classified_regions:
                    # Only process text regions
                    if region.get("classified_type") != "text":
                        continue

                    # Try to get content from layout detection
                    content = region.get("content", "")

                    # Fallback to OCR if content is empty
                    if not content and self.use_ocr:
                        content = self._extract_text_with_ocr(region)

                    if content:
                        text_blocks.append(
                            {
                                "page_number": region["page_number"],
                                "object_index": region["object_index"],
                                "bbox": region["bbox"],
                                "text": content,
                            }
                        )

                extracted_text_list.append(json.dumps(text_blocks))
                logger.info(f"Extracted {len(text_blocks)} text blocks")

            except Exception as e:
                logger.error(f"Failed to extract text: {e}")
                extracted_text_list.append(json.dumps([]))

        df[self.output_field] = extracted_text_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
