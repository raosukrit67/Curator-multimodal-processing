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
Apply Quality Filters to Deduplicated Data

This script applies quality filtering to deduplicated multimodal content using
a Pipeline with custom filter stages designed for multimodal PDF data.

Quality Filter Stages:
1. BoundingBoxValidationStage - Validate bounding box coordinates
2. ContentCompletenessStage - Ensure minimum content per page
3. ExtractionQualityStage - Filter pages with failed extractions
4. ModalityBalanceStage - Ensure sufficient modality diversity
5. TextQualityStage - Apply basic text quality checks

Input:
    dedup_results/deduplicated_data.jsonl

Output:
    quality_results/filtered_data.jsonl (final curated dataset)
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.tasks import DocumentBatch


class BoundingBoxValidationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Validate bounding box coordinates in multimodal data."""

    def __init__(self):
        self.name = "bbox_validation"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages", "bbox_valid"]

    def _validate_bbox(self, bbox: dict) -> bool:
        """Validate bounding box has valid coordinates."""
        try:
            x = float(bbox.get("x", -1))
            y = float(bbox.get("y", -1))
            width = float(bbox.get("width", -1))
            height = float(bbox.get("height", -1))

            # Check all values are non-negative and within [0, 1] for normalized coords
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                return False
            if x > 1 or y > 1 or width > 1 or height > 1:
                return False
            if x + width > 1.01 or y + height > 1.01:  # Small tolerance
                return False

            return True
        except (ValueError, TypeError):
            return False

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        valid_flags = []
        invalid_count = 0

        for pages_json in df["pages"]:
            try:
                pages = json.loads(pages_json) if isinstance(pages_json, str) else pages_json

                is_valid = True
                for page in pages:
                    # Validate text bboxes
                    for text_block in page.get("text", []):
                        if not self._validate_bbox(text_block.get("bbox", {})):
                            is_valid = False
                            break

                    # Validate table bboxes
                    for table in page.get("tables", []):
                        if not self._validate_bbox(table.get("bbox", {})):
                            is_valid = False
                            break

                    # Validate image bboxes
                    for image in page.get("images", []):
                        if not self._validate_bbox(image.get("bbox", {})):
                            is_valid = False
                            break

                    if not is_valid:
                        break

                valid_flags.append(is_valid)
                if not is_valid:
                    invalid_count += 1

            except Exception as e:
                logger.warning(f"Error validating bboxes: {e}")
                valid_flags.append(False)
                invalid_count += 1

        df["bbox_valid"] = valid_flags

        logger.info(f"BBox validation: {invalid_count} invalid documents out of {len(df)}")

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class ContentCompletenessStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Ensure minimum content per page."""

    def __init__(self, min_text_length: int = 50, min_pages: int = 1):
        self.min_text_length = min_text_length
        self.min_pages = min_pages
        self.name = "content_completeness"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages", "bbox_valid"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages", "bbox_valid", "content_complete"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        complete_flags = []
        incomplete_count = 0

        for pages_json in df["pages"]:
            try:
                pages = json.loads(pages_json) if isinstance(pages_json, str) else pages_json

                # Check minimum number of pages
                if len(pages) < self.min_pages:
                    complete_flags.append(False)
                    incomplete_count += 1
                    continue

                # Check each page has sufficient content
                is_complete = True
                for page in pages:
                    text_blocks = page.get("text", [])
                    tables = page.get("tables", [])
                    images = page.get("images", [])

                    # Calculate total text length
                    total_text_length = sum(
                        len(block.get("text", "")) for block in text_blocks
                    )

                    # Page should have either:
                    # - Sufficient text, OR
                    # - At least one table or image
                    has_content = (
                        total_text_length >= self.min_text_length
                        or len(tables) > 0
                        or len(images) > 0
                    )

                    if not has_content:
                        is_complete = False
                        break

                complete_flags.append(is_complete)
                if not is_complete:
                    incomplete_count += 1

            except Exception as e:
                logger.warning(f"Error checking completeness: {e}")
                complete_flags.append(False)
                incomplete_count += 1

        df["content_complete"] = complete_flags

        logger.info(f"Content completeness: {incomplete_count} incomplete documents out of {len(df)}")

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class ModalityBalanceStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Ensure documents have sufficient modality diversity."""

    def __init__(self, min_modalities: int = 1):
        self.min_modalities = min_modalities
        self.name = "modality_balance"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages", "bbox_valid", "content_complete"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages", "bbox_valid", "content_complete", "modality_balanced"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        balanced_flags = []
        unbalanced_count = 0

        for pages_json in df["pages"]:
            try:
                pages = json.loads(pages_json) if isinstance(pages_json, str) else pages_json

                # Count modalities present in document
                has_text = False
                has_tables = False
                has_images = False

                for page in pages:
                    if page.get("text"):
                        has_text = True
                    if page.get("tables"):
                        has_tables = True
                    if page.get("images"):
                        has_images = True

                modality_count = sum([has_text, has_tables, has_images])
                is_balanced = modality_count >= self.min_modalities

                balanced_flags.append(is_balanced)
                if not is_balanced:
                    unbalanced_count += 1

            except Exception as e:
                logger.warning(f"Error checking modality balance: {e}")
                balanced_flags.append(False)
                unbalanced_count += 1

        df["modality_balanced"] = balanced_flags

        logger.info(f"Modality balance: {unbalanced_count} unbalanced documents out of {len(df)}")

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class TextQualityStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Apply basic text quality checks."""

    def __init__(
        self,
        min_avg_word_length: int = 3,
        max_avg_word_length: int = 15,
        min_alpha_ratio: float = 0.7,
    ):
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.min_alpha_ratio = min_alpha_ratio
        self.name = "text_quality"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages", "bbox_valid", "content_complete", "modality_balanced"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages", "bbox_valid", "content_complete", "modality_balanced", "text_quality_passed"]

    def _check_text_quality(self, text: str) -> bool:
        """Check if text meets quality criteria."""
        if not text or len(text) < 10:
            return True  # Skip very short text

        # Calculate average word length
        words = text.split()
        if not words:
            return False

        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < self.min_avg_word_length or avg_word_length > self.max_avg_word_length:
            return False

        # Calculate alphanumeric ratio
        alpha_count = sum(c.isalnum() for c in text)
        total_count = len(text)
        alpha_ratio = alpha_count / total_count if total_count > 0 else 0

        if alpha_ratio < self.min_alpha_ratio:
            return False

        return True

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        quality_flags = []
        low_quality_count = 0

        for pages_json in df["pages"]:
            try:
                pages = json.loads(pages_json) if isinstance(pages_json, str) else pages_json

                is_quality = True
                for page in pages:
                    # Check text quality
                    for text_block in page.get("text", []):
                        text = text_block.get("text", "")
                        if not self._check_text_quality(text):
                            is_quality = False
                            break

                    if not is_quality:
                        break

                quality_flags.append(is_quality)
                if not is_quality:
                    low_quality_count += 1

            except Exception as e:
                logger.warning(f"Error checking text quality: {e}")
                quality_flags.append(False)
                low_quality_count += 1

        df["text_quality_passed"] = quality_flags

        logger.info(f"Text quality: {low_quality_count} low-quality documents out of {len(df)}")

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class QualityFilterStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Filter documents based on all quality checks."""

    def __init__(self):
        self.name = "quality_filter"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "pages",
            "bbox_valid",
            "content_complete",
            "modality_balanced",
            "text_quality_passed",
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pages"]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        # Filter based on all quality checks
        filtered_df = df[
            df["bbox_valid"]
            & df["content_complete"]
            & df["modality_balanced"]
            & df["text_quality_passed"]
        ].copy()

        # Remove quality check columns
        filtered_df = filtered_df.drop(
            columns=[
                "bbox_valid",
                "content_complete",
                "modality_balanced",
                "text_quality_passed",
            ]
        )

        initial_count = len(df)
        final_count = len(filtered_df)
        removed_count = initial_count - final_count

        logger.info(f"Quality filter: Removed {removed_count} documents, kept {final_count}")

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=filtered_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def create_quality_filtering_pipeline(
    input_path: str,
    output_path: str,
    min_text_length: int = 50,
    min_pages: int = 1,
    min_modalities: int = 1,
) -> Pipeline:
    """Create the quality filtering pipeline.

    Args:
        input_path: Path to deduplicated data
        output_path: Path to write filtered data
        min_text_length: Minimum text length per page
        min_pages: Minimum number of pages
        min_modalities: Minimum number of modalities (text/table/image)

    Returns:
        Configured pipeline
    """
    pipeline = Pipeline(
        name="quality_filtering",
        description="Filter multimodal PDF data based on quality metrics",
    )

    # Stage 1: Read deduplicated data
    pipeline.add_stage(JsonlReader(input_path=input_path))

    # Stage 2: Validate bounding boxes
    pipeline.add_stage(BoundingBoxValidationStage())

    # Stage 3: Check content completeness
    pipeline.add_stage(
        ContentCompletenessStage(
            min_text_length=min_text_length,
            min_pages=min_pages,
        )
    )

    # Stage 4: Check modality balance
    pipeline.add_stage(ModalityBalanceStage(min_modalities=min_modalities))

    # Stage 5: Check text quality
    pipeline.add_stage(TextQualityStage())

    # Stage 6: Apply filters
    pipeline.add_stage(QualityFilterStage())

    # Stage 7: Write filtered data
    pipeline.add_stage(JsonlWriter(output_path=output_path))

    return pipeline


def main():
    """Run quality filtering pipeline."""
    # Setup paths
    script_dir = Path(__file__).parent
    input_path = script_dir / "dedup_results" / "deduplicated_data.jsonl"
    output_dir = script_dir / "quality_results"
    output_path = output_dir / "filtered_data.jsonl"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear output file if it exists
    if output_path.exists():
        output_path.unlink()

    logger.info("Starting quality filtering pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Check input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run 2_remove_duplicates.py first")
        return

    # Create pipeline
    pipeline = create_quality_filtering_pipeline(
        input_path=str(input_path),
        output_path=str(output_path),
        min_text_length=50,
        min_pages=1,
        min_modalities=1,  # At least one modality (text, table, or image)
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
        logger.info("Final curated dataset ready!")

    except Exception as e:
        logger.error(f"Quality filtering failed: {e}")
        logger.exception("Full traceback:")
        raise

    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
