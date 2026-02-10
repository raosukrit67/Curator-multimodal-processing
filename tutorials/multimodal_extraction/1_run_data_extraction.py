#!/usr/bin/env python3
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

"""
Multimodal PDF Data Extraction Pipeline

This script implements a complete pipeline for extracting multimodal content
(text, tables, images) from PDF documents using vision-language models with
local vLLM inference.

Pipeline Stages:
1. JSONLReaderStage - Read PDF paths from JSONL
2. PDFToImageStage - Convert PDF pages to images (300 DPI)
3. LayoutDetectionStage - Detect layout with vision-language model (vLLM)
4. BoundingBoxExtractionStage - Crop regions from images
5. ContentTypeClassificationStage - Classify content (text/table/image)
6. TableExtractionStage - Extract tables (LaTeX to HTML)
7. TextExtractionStage - Extract text from regions
8. ImageExtractionStage - Crop and save images
9. DeepAnalysisStage - Deep content analysis (vLLM)
10. JSONLWriterStage - Write results (organized by modality)

Input:
    extraction_results/downloaded_pdfs.jsonl

Output:
    extraction_results/extracted_multimodal_data.jsonl
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.pdf import (
    BoundingBoxExtractionStage,
    ContentTypeClassificationStage,
    DeepAnalysisStage,
    ImageExtractionStage,
    LayoutDetectionStage,
    PDFToImageStage,
    TableExtractionStage,
    TextExtractionStage,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, _EmptyTask


class JSONLReaderStage(ProcessingStage[_EmptyTask, DocumentBatch]):
    """Read PDF paths from JSONL file."""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.name = "jsonl_reader"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pdf_path"]

    def process(self, _: _EmptyTask) -> list[DocumentBatch]:
        """Read JSONL file and create DocumentBatch tasks."""
        logger.info(f"Reading PDF paths from {self.input_path}")

        # Read JSONL file
        pdf_paths = []
        with open(self.input_path) as f:
            for line in f:
                entry = json.loads(line)
                pdf_paths.append(entry["pdf_path"])

        logger.info(f"Loaded {len(pdf_paths)} PDF paths")

        # Create DocumentBatch tasks (one per PDF)
        tasks = []
        for i, pdf_path in enumerate(pdf_paths):
            df = pd.DataFrame({"pdf_path": [pdf_path]})
            task = DocumentBatch(
                task_id=f"pdf_{i}",
                dataset_name="pdfs",
                data=df,
            )
            tasks.append(task)

        return tasks


class JSONLWriterStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Write extracted multimodal data to JSONL file."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.name = "jsonl_writer"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "pdf_path",
            "page_images",
            "layout_objects",
            "extracted_tables",
            "extracted_text",
            "extracted_images",
            "analysis_results",
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup(self, worker_metadata=None) -> None:
        """Create output directory."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def process(self, batch: DocumentBatch) -> None:
        """Write batch to JSONL file."""
        df = batch.to_pandas()

        # Organize results by modality per PDF
        for idx in range(len(df)):
            row = df.iloc[idx]

            # Parse JSON fields
            page_images = json.loads(row.get("page_images", "[]"))
            extracted_tables = json.loads(row.get("extracted_tables", "[]"))
            extracted_text = json.loads(row.get("extracted_text", "[]"))
            extracted_images = json.loads(row.get("extracted_images", "[]"))
            analysis_results = json.loads(row.get("analysis_results", "[]"))

            # Organize by page
            pages_data = {}
            for page_img in page_images:
                page_num = page_img["page_number"]
                if page_num not in pages_data:
                    pages_data[page_num] = {
                        "page_number": page_num,
                        "text": [],
                        "tables": [],
                        "images": [],
                        "analyses": [],
                    }

            # Add text blocks
            for text_block in extracted_text:
                page_num = text_block["page_number"]
                if page_num in pages_data:
                    pages_data[page_num]["text"].append(
                        {"bbox": text_block["bbox"], "text": text_block["text"]}
                    )

            # Add tables
            for table in extracted_tables:
                page_num = table["page_number"]
                if page_num in pages_data:
                    pages_data[page_num]["tables"].append(
                        {"bbox": table["bbox"], "html": table["html"]}
                    )

            # Add images
            for image in extracted_images:
                page_num = image["page_number"]
                if page_num in pages_data:
                    pages_data[page_num]["images"].append(
                        {
                            "bbox": image["bbox"],
                            "type": image["type"],
                            "image_base64": image["image_base64"],
                        }
                    )

            # Add analyses
            for analysis in analysis_results:
                page_num = analysis["page_number"]
                if page_num in pages_data:
                    pages_data[page_num]["analyses"].append(
                        {
                            "type": analysis["type"],
                            "bbox": analysis["bbox"],
                            "analysis": analysis["analysis"],
                        }
                    )

            # Create output entry
            output_entry = {
                "pdf_path": row["pdf_path"],
                "pages": list(pages_data.values()),
            }

            # Append to JSONL file
            with open(self.output_path, "a") as f:
                f.write(json.dumps(output_entry) + "\n")

        logger.info(f"Wrote {len(df)} PDFs to {self.output_path}")

        # Return None to end the pipeline
        return None


def create_extraction_pipeline(
    input_path: str,
    output_path: str,
    model_cache_dir: str | None = None,
) -> Pipeline:
    """Create the multimodal extraction pipeline.

    Args:
        input_path: Path to input JSONL file with PDF paths
        output_path: Path to output JSONL file
        model_cache_dir: Directory for caching model weights

    Returns:
        Configured pipeline
    """
    pipeline = Pipeline(
        name="multimodal_extraction",
        description="Extract multimodal content from PDFs using vision-language models",
    )

    # Stage 1: Read PDF paths
    pipeline.add_stage(JSONLReaderStage(input_path=input_path))

    # Stage 2: Convert PDFs to images
    pipeline.add_stage(
        PDFToImageStage(
            pdf_path_field="pdf_path",
            dpi=300,
            output_field="page_images",
        )
    )

    # Stage 3: Detect layout with vLLM
    pipeline.add_stage(
        LayoutDetectionStage(
            model_identifier="nvidia/nemoretriever-parse",
            page_images_field="page_images",
            output_field="layout_objects",
            max_tokens=3500,
            cache_dir=model_cache_dir,
        )
    )

    # Stage 4: Extract bounding boxes
    pipeline.add_stage(
        BoundingBoxExtractionStage(
            page_images_field="page_images",
            layout_objects_field="layout_objects",
            output_field="cropped_regions",
        )
    )

    # Stage 5: Classify content types
    pipeline.add_stage(
        ContentTypeClassificationStage(
            cropped_regions_field="cropped_regions",
            output_field="classified_regions",
        )
    )

    # Stage 6: Extract tables
    pipeline.add_stage(
        TableExtractionStage(
            classified_regions_field="classified_regions",
            output_field="extracted_tables",
        )
    )

    # Stage 7: Extract text
    pipeline.add_stage(
        TextExtractionStage(
            classified_regions_field="classified_regions",
            output_field="extracted_text",
            use_ocr=False,  # Set to True to enable OCR fallback
        )
    )

    # Stage 8: Extract images
    pipeline.add_stage(
        ImageExtractionStage(
            classified_regions_field="classified_regions",
            output_field="extracted_images",
        )
    )

    # Stage 9: Deep analysis with vLLM
    pipeline.add_stage(
        DeepAnalysisStage(
            model_identifier="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            classified_regions_field="classified_regions",
            output_field="analysis_results",
            max_tokens=1024,
            temperature=0.2,
            top_p=0.7,
            cache_dir=model_cache_dir,
        )
    )

    # Stage 10: Write results
    pipeline.add_stage(JSONLWriterStage(output_path=output_path))

    return pipeline


def main():
    """Run the multimodal extraction pipeline."""
    # Setup paths
    script_dir = Path(__file__).parent
    input_path = script_dir / "extraction_results" / "downloaded_pdfs.jsonl"
    output_path = script_dir / "extraction_results" / "extracted_multimodal_data.jsonl"
    model_cache_dir = script_dir / "model_cache"

    # Clear output file if it exists
    if output_path.exists():
        output_path.unlink()

    logger.info("Starting multimodal PDF extraction pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Create pipeline
    pipeline = create_extraction_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        model_cache_dir=str(model_cache_dir),
    )

    # Print pipeline description
    logger.info("\n" + pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    # Initialize Ray
    ray_client = RayClient()
    ray_client.start()

    try:
        # Create executor
        executor = XennaExecutor()

        # Execute pipeline
        logger.info("Starting pipeline execution...")
        results = pipeline.run(executor)

        # Print results
        logger.info("\nPipeline completed!")
        logger.info(f"Results written to: {output_path}")

    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
