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
Quality Filters for Q&A Dataset Curation from PDFs

Filters extracted PDF content to retain pages suitable for generating
high-quality question-answer pairs. Each filter targets a specific failure
mode that would produce bad Q&A data:

1. ExtractionCompletenessFilter - Drop docs where Parse mostly failed
2. BoilerplateFilter            - Drop pages that are only headers/footers/ToC
3. QAReadinessFilter            - Keep pages with answerable content
                                  (substantive text, data tables, or described figures)

Usage:
    python 3_run_quality_filters.py --input data/extracted/ \
                                    --output data/filtered/
"""

import argparse
import os

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.pdf.quality_filters import (
    ApplyFiltersStage,
    BoilerplateFilter,
    ExtractionCompletenessFilter,
    QAReadinessFilter,
)
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter


def build_pipeline(
    input_path: str,
    output_dir: str,
    min_extraction_ratio: float = 0.5,
    min_answer_length: int = 80,
    min_qa_pages: int = 1,
) -> Pipeline:
    """Build the quality filtering pipeline for Q&A dataset curation."""
    pipeline = Pipeline(
        name="pdf_qa_quality_filtering",
        description="Filter extracted PDF data for Q&A dataset generation",
    )

    pipeline.add_stage(JsonlReader(file_paths=input_path))
    pipeline.add_stage(ExtractionCompletenessFilter(
        min_extraction_ratio=min_extraction_ratio,
    ))
    pipeline.add_stage(BoilerplateFilter())
    pipeline.add_stage(QAReadinessFilter(
        min_answer_length=min_answer_length,
        min_qa_pages=min_qa_pages,
    ))
    pipeline.add_stage(ApplyFiltersStage())
    pipeline.add_stage(JsonlWriter(
        path=output_dir,
        fields=["pdf_path", "pages"],
    ))

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter extracted PDF data for Q&A dataset curation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "extracted"),
        help="Input directory of JSONL files from extraction pipeline",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "filtered"),
        help="Output directory for filtered JSONL files",
    )
    parser.add_argument("--min-extraction-ratio", type=float, default=0.5,
                        help="Min fraction of pages with extracted content")
    parser.add_argument("--min-answer-length", type=int, default=80,
                        help="Min chars in a text block to be Q&A-worthy")
    parser.add_argument("--min-qa-pages", type=int, default=1,
                        help="Min Q&A-ready pages per document")

    args = parser.parse_args()

    logger.info("Starting Q&A quality filtering pipeline")

    pipeline = build_pipeline(
        input_path=args.input,
        output_dir=args.output,
        min_extraction_ratio=args.min_extraction_ratio,
        min_answer_length=args.min_answer_length,
        min_qa_pages=args.min_qa_pages,
    )

    logger.info("\n" + pipeline.describe())

    ray_client = RayClient()
    ray_client.start()

    try:
        executor = XennaExecutor()
        pipeline.run(executor)
        logger.info(f"Quality filtering complete. Output: {args.output}")
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
