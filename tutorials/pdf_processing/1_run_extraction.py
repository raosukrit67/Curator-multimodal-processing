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

"""
Multimodal PDF Data Extraction Pipeline

Extracts text, tables, and visual content from PDF documents using two models:
1. Nemotron Parse 1.1B - OCR, layout detection, and text extraction
2. Nemotron Nano 12B VL - Visual content description (pictures, figures, charts)

Pipeline Stages:
1. JsonlReader         - Read PDF paths from JSONL manifest
2. PDFToImageStage     - Convert PDF pages to 300 DPI images
3. LayoutDetectionStage - Nemotron Parse: OCR + layout + text extraction
4. ContentRoutingStage  - Route regions: text → direct, visual → VL model
5. VisualAnalysisStage  - Nemotron Nano VL: describe visual content
6. TextAssemblyStage    - Combine all modalities into structured JSON
7. JsonlWriter          - Write results to output directory

Usage:
    python 1_run_extraction.py --input data/raw/pdf_files.jsonl --output data/extracted/

See --help for all options.
"""

import argparse
import os

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.pdf import (
    ContentRoutingStage,
    LayoutDetectionStage,
    PDFToImageStage,
    TextAssemblyStage,
    VisualAnalysisStage,
)
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.utils import prompts


def resolve_prompt(name_or_value: str) -> str:
    """Resolve a prompt by name from prompts module, or return as literal string."""
    try:
        return getattr(prompts, name_or_value)
    except AttributeError:
        return name_or_value


def build_pipeline(  # noqa: PLR0913
    input_path: str,
    output_dir: str,
    parse_model: str = "nvidia/NVIDIA-Nemotron-Parse-v1.1",
    vl_model: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    parse_prompt: str = "NEMOTRON_PARSE_PROMPT",
    model_cache_dir: str | None = None,
    dpi: int = 300,
    include_tables_for_vl: bool = False,
    verbose: bool = False,
) -> Pipeline:
    """Build the multimodal PDF extraction pipeline.

    Args:
        input_path: Path to JSONL manifest with PDF paths.
        output_dir: Directory for output JSONL files.
        parse_model: Nemotron Parse model identifier.
        vl_model: Nemotron Nano VL model identifier.
        parse_prompt: Parse prompt name from prompts.py or literal string.
        model_cache_dir: Cache directory for model weights.
        dpi: DPI for PDF page rendering.
        include_tables_for_vl: Also send tables to VL for interpretation.
        verbose: Enable verbose logging.

    Returns:
        Configured Pipeline.
    """
    pipeline = Pipeline(
        name="pdf_extraction",
        description="Extract multimodal content from PDFs using Nemotron Parse + Nano VL",
    )

    # Stage 1: Read PDF manifest
    pipeline.add_stage(JsonlReader(file_paths=input_path))

    # Stage 2: Convert PDFs to images
    pipeline.add_stage(PDFToImageStage(dpi=dpi))

    # Stage 3: Layout detection + text extraction with Nemotron Parse
    pipeline.add_stage(
        LayoutDetectionStage(
            model_identifier=parse_model,
            prompt=resolve_prompt(parse_prompt),
            cache_dir=model_cache_dir,
            verbose=verbose,
        )
    )

    # Stage 4: Route content (text → direct, visual → VL)
    pipeline.add_stage(
        ContentRoutingStage(
            include_tables_for_vl=include_tables_for_vl,
        )
    )

    # Stage 5: Visual analysis with Nemotron Nano VL
    pipeline.add_stage(
        VisualAnalysisStage(
            model_identifier=vl_model,
            cache_dir=model_cache_dir,
            verbose=verbose,
        )
    )

    # Stage 6: Assemble all modalities + format output
    pipeline.add_stage(TextAssemblyStage())

    # Stage 7: Write results
    pipeline.add_stage(JsonlWriter(
        path=output_dir,
        fields=["pdf_path", "pages", "text"],
    ))

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract multimodal content from PDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "raw", "pdf_files.jsonl"),
        help="Path to JSONL manifest with PDF paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "extracted"),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--parse-model",
        type=str,
        default="nvidia/NVIDIA-Nemotron-Parse-v1.1",
        help="Nemotron Parse model identifier",
    )
    parser.add_argument(
        "--vl-model",
        type=str,
        default="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        help="Nemotron Nano VL model identifier",
    )
    parser.add_argument(
        "--parse-prompt",
        type=str,
        default="NEMOTRON_PARSE_PROMPT",
        help="Parse prompt name from prompts.py or literal prompt string",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=None,
        help="Cache directory for model weights",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF page rendering",
    )
    parser.add_argument(
        "--include-tables-for-vl",
        action="store_true",
        help="Also send tables to VL model for rich interpretation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logger.info("Starting PDF extraction pipeline")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Parse model: {args.parse_model}")
    logger.info(f"VL model: {args.vl_model}")

    pipeline = build_pipeline(
        input_path=args.input,
        output_dir=args.output,
        parse_model=args.parse_model,
        vl_model=args.vl_model,
        parse_prompt=args.parse_prompt,
        model_cache_dir=args.model_cache_dir,
        dpi=args.dpi,
        include_tables_for_vl=args.include_tables_for_vl,
        verbose=args.verbose,
    )

    logger.info("\n" + pipeline.describe())

    ray_client = RayClient()
    ray_client.start()

    try:
        executor = XennaExecutor()
        pipeline.run(executor)
        logger.info(f"Pipeline completed. Results at: {args.output}")
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
