#!/usr/bin/env python3
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
    python 4_run_quality_filters.py --input data/extracted/extracted_data.jsonl \
                                    --output data/filtered/filtered_data.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, _EmptyTask

# Parse classes that are structural noise, not Q&A-worthy content
_BOILERPLATE_CLASSES = {"Page-Header", "Page-Footer", "Index"}


class QualityReaderStage(ProcessingStage[_EmptyTask, DocumentBatch]):
    """Read extraction output JSONL for quality filtering."""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.name = "quality_reader"
        self.resources = Resources(cpus=1.0)
        self.batch_size = 1

    def inputs(self):
        return [], []

    def outputs(self):
        return ["data"], ["pdf_path", "pages"]

    def process(self, _: _EmptyTask) -> list[DocumentBatch]:
        logger.info(f"Reading extracted data from {self.input_path}")
        tasks = []
        with open(self.input_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                df = pd.DataFrame({
                    "pdf_path": [entry.get("pdf_path", "")],
                    "pages": [json.dumps(entry.get("pages", []))],
                })
                tasks.append(DocumentBatch(
                    task_id=f"doc_{i}",
                    dataset_name="quality_filter",
                    data=df,
                ))
        logger.info(f"Loaded {len(tasks)} documents for quality filtering")
        return tasks


class ExtractionCompletenessFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Drop documents where the extraction pipeline largely failed.

    If Parse returned nothing for most pages, the PDF was likely
    scanned without OCR, encrypted, or corrupt. These documents
    cannot produce Q&A pairs.

    Args:
        min_extraction_ratio: Minimum fraction of pages that must have
            at least one text_block, table, or figure.
    """

    def __init__(self, min_extraction_ratio: float = 0.5):
        self.min_extraction_ratio = min_extraction_ratio
        self.name = "extraction_completeness_filter"
        self.resources = Resources(cpus=0.5)
        self.batch_size = 1

    def inputs(self):
        return ["data"], ["pages"]

    def outputs(self):
        return ["data"], ["pages", "extraction_ok"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        flags = []

        for pages_json in df["pages"]:
            pages = json.loads(pages_json)
            if not pages:
                flags.append(False)
                continue

            pages_with_content = sum(
                1 for p in pages
                if p.get("text_blocks") or p.get("tables") or p.get("figures")
            )
            ratio = pages_with_content / len(pages)
            flags.append(ratio >= self.min_extraction_ratio)

        df["extraction_ok"] = flags
        failed = sum(1 for f in flags if not f)
        logger.info(f"ExtractionCompleteness: {failed}/{len(flags)} documents failed")

        return DocumentBatch(
            task_id=batch.task_id, dataset_name=batch.dataset_name,
            data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf,
        )


class BoilerplateFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Remove pages that consist entirely of boilerplate content.

    Pages with only Page-Header, Page-Footer, or Index blocks cannot
    produce meaningful Q&A pairs. This filter strips those pages from
    the document (rather than dropping the whole document).

    Also removes pages that are effectively empty after stripping
    boilerplate (e.g., cover pages with only a title and no body text).

    Args:
        min_substantive_blocks: Minimum number of non-boilerplate text
            blocks a page must have (tables and figures always count).
    """

    def __init__(self, min_substantive_blocks: int = 1):
        self.min_substantive_blocks = min_substantive_blocks
        self.name = "boilerplate_filter"
        self.resources = Resources(cpus=0.5)
        self.batch_size = 1

    def inputs(self):
        return ["data"], ["pages", "extraction_ok"]

    def outputs(self):
        return ["data"], ["pages", "extraction_ok", "pages_removed"]

    def _is_substantive(self, page: dict) -> bool:
        """Check if a page has content beyond boilerplate."""
        # Tables and figures are always substantive
        if page.get("tables") or page.get("figures"):
            return True

        # Count non-boilerplate text blocks
        substantive = 0
        for block in page.get("text_blocks", []):
            if block.get("class_name") not in _BOILERPLATE_CLASSES:
                substantive += 1

        return substantive >= self.min_substantive_blocks

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        filtered_pages_list = []
        pages_removed_list = []

        for pages_json in df["pages"]:
            pages = json.loads(pages_json)
            kept = [p for p in pages if self._is_substantive(p)]
            removed = len(pages) - len(kept)

            # Rebuild full_text for kept pages (strip boilerplate blocks)
            for page in kept:
                page["text_blocks"] = [
                    b for b in page.get("text_blocks", [])
                    if b.get("class_name") not in _BOILERPLATE_CLASSES
                ]
                # Rebuild full_text without boilerplate
                parts = []
                for b in page["text_blocks"]:
                    text = b.get("text", "").strip()
                    if text:
                        parts.append(text)
                for t in page.get("tables", []):
                    latex = t.get("latex", "").strip()
                    if latex:
                        parts.append(f"[Table: {latex}]")
                for f in page.get("figures", []):
                    desc = f.get("description", "").strip()
                    if desc:
                        parts.append(f"[{f.get('class_name', 'Figure')}: {desc}]")
                page["full_text"] = "\n\n".join(parts)

            filtered_pages_list.append(json.dumps(kept))
            pages_removed_list.append(removed)

        df["pages"] = filtered_pages_list
        df["pages_removed"] = pages_removed_list

        total_removed = sum(pages_removed_list)
        logger.info(f"Boilerplate: removed {total_removed} pages across {len(df)} documents")

        return DocumentBatch(
            task_id=batch.task_id, dataset_name=batch.dataset_name,
            data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf,
        )


class QAReadinessFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Drop documents that cannot produce useful Q&A pairs.

    A document is Q&A-ready if it has at least one page with answerable
    content. A page is answerable if it contains any of:
    - A text block with >= min_answer_length characters (enough context
      to form a question and answer)
    - A table (structured data → factual Q&A pairs)
    - A figure with a non-empty VL description (visual Q&A)

    Args:
        min_answer_length: Minimum character length for a text block
            to be considered sufficient for Q&A.
        min_qa_pages: Minimum number of Q&A-ready pages per document.
    """

    def __init__(self, min_answer_length: int = 80, min_qa_pages: int = 1):
        self.min_answer_length = min_answer_length
        self.min_qa_pages = min_qa_pages
        self.name = "qa_readiness_filter"
        self.resources = Resources(cpus=0.5)
        self.batch_size = 1

    def inputs(self):
        return ["data"], ["pages", "extraction_ok", "pages_removed"]

    def outputs(self):
        return ["data"], ["pages", "extraction_ok", "pages_removed", "qa_ready"]

    def _page_is_qa_ready(self, page: dict) -> bool:
        # Tables are high-value for factual Q&A
        if page.get("tables"):
            return True

        # Figures with descriptions enable visual Q&A
        for fig in page.get("figures", []):
            if fig.get("description", "").strip():
                return True

        # Text blocks must be long enough to contain an answer
        for block in page.get("text_blocks", []):
            if len(block.get("text", "")) >= self.min_answer_length:
                return True

        return False

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        flags = []

        for pages_json in df["pages"]:
            pages = json.loads(pages_json)
            qa_pages = sum(1 for p in pages if self._page_is_qa_ready(p))
            flags.append(qa_pages >= self.min_qa_pages)

        df["qa_ready"] = flags
        failed = sum(1 for f in flags if not f)
        logger.info(f"QAReadiness: {failed}/{len(flags)} documents failed")

        return DocumentBatch(
            task_id=batch.task_id, dataset_name=batch.dataset_name,
            data=df, _metadata=batch._metadata, _stage_perf=batch._stage_perf,
        )


class ApplyFiltersStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Apply all quality filter flags and remove failing documents."""

    FILTER_COLUMNS = ["extraction_ok", "qa_ready"]
    DROP_COLUMNS = ["extraction_ok", "pages_removed", "qa_ready"]

    def __init__(self):
        self.name = "apply_filters"
        self.resources = Resources(cpus=0.5)
        self.batch_size = 1

    def inputs(self):
        return ["data"], ["pages", "pdf_path"] + self.FILTER_COLUMNS

    def outputs(self):
        return ["data"], ["pages", "pdf_path"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        mask = pd.Series(True, index=df.index)
        for col in self.FILTER_COLUMNS:
            if col in df.columns:
                mask &= df[col].astype(bool)

        filtered = df[mask].drop(columns=self.DROP_COLUMNS, errors="ignore").copy()

        logger.info(
            f"Applied filters: {len(df)} → {len(filtered)} documents "
            f"({len(df) - len(filtered)} removed)"
        )

        return DocumentBatch(
            task_id=batch.task_id, dataset_name=batch.dataset_name,
            data=filtered, _metadata=batch._metadata, _stage_perf=batch._stage_perf,
        )


class QualityWriterStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Write filtered data to JSONL."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.name = "quality_writer"
        self.resources = Resources(cpus=1.0)
        self.batch_size = 1

    def inputs(self):
        return ["data"], ["pdf_path", "pages"]

    def outputs(self):
        return [], []

    def setup(self, worker_metadata=None):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def process(self, batch: DocumentBatch) -> None:
        df = batch.to_pandas()
        for _, row in df.iterrows():
            entry = {
                "pdf_path": row["pdf_path"],
                "pages": json.loads(row["pages"]),
            }
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Wrote {len(df)} filtered documents to {self.output_path}")
        return None


def build_quality_pipeline(
    input_path: str,
    output_path: str,
    min_extraction_ratio: float = 0.5,
    min_answer_length: int = 80,
    min_qa_pages: int = 1,
) -> Pipeline:
    """Build the quality filtering pipeline for Q&A dataset curation."""
    pipeline = Pipeline(
        name="pdf_qa_quality_filtering",
        description="Filter extracted PDF data for Q&A dataset generation",
    )

    pipeline.add_stage(QualityReaderStage(input_path=input_path))
    pipeline.add_stage(ExtractionCompletenessFilter(
        min_extraction_ratio=min_extraction_ratio,
    ))
    pipeline.add_stage(BoilerplateFilter())
    pipeline.add_stage(QAReadinessFilter(
        min_answer_length=min_answer_length,
        min_qa_pages=min_qa_pages,
    ))
    pipeline.add_stage(ApplyFiltersStage())
    pipeline.add_stage(QualityWriterStage(output_path=output_path))

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter extracted PDF data for Q&A dataset curation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "extracted", "extracted_data.jsonl"),
        help="Input JSONL from extraction pipeline",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "filtered", "filtered_data.jsonl"),
        help="Output JSONL for filtered data",
    )
    parser.add_argument("--min-extraction-ratio", type=float, default=0.5,
                        help="Min fraction of pages with extracted content")
    parser.add_argument("--min-answer-length", type=int, default=80,
                        help="Min chars in a text block to be Q&A-worthy")
    parser.add_argument("--min-qa-pages", type=int, default=1,
                        help="Min Q&A-ready pages per document")

    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    logger.info("Starting Q&A quality filtering pipeline")

    pipeline = build_quality_pipeline(
        input_path=args.input,
        output_path=args.output,
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
