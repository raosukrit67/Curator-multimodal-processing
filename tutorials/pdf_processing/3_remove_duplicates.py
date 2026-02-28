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
Remove Duplicates from Extracted Multimodal Data

This script applies deduplication to the extracted multimodal content using
Curator's built-in fuzzy deduplication workflow.

Deduplication Method:
- Fuzzy Deduplication - MinHash + LSH for near-duplicate detection

The script:
1. Preprocesses multimodal data to extract text from all pages
2. Runs FuzzyDeduplicationWorkflow to identify duplicates
3. Filters out duplicates from the original data
4. Writes deduplicated results

Input:
    data/extracted/extracted_data.jsonl

Output:
    data/dedup/deduplicated_data.jsonl
    data/dedup/duplicate_ids.parquet (intermediate)
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow


def preprocess_for_deduplication(input_path: str, temp_path: str) -> None:
    """Preprocess multimodal data for deduplication.

    Extracts text from all pages and modalities to create a text field
    for deduplication.

    Args:
        input_path: Path to extracted multimodal data JSONL
        temp_path: Path to write preprocessed data with text field
    """
    logger.info("Preprocessing multimodal data for deduplication...")

    temp_path_obj = Path(temp_path)
    temp_path_obj.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    with open(input_path) as infile, open(temp_path, "w") as outfile:
        for line_num, line in enumerate(infile):
            try:
                entry = json.loads(line)
                pdf_path = entry.get("pdf_path", "")
                pages = entry.get("pages", [])

                # Extract all text from all pages and modalities
                text_parts = []

                for page in pages:
                    # Use full_text if available (from TextAssemblyStage)
                    full_text = page.get("full_text", "").strip()
                    if full_text:
                        text_parts.append(full_text)
                        continue

                    # Fallback: extract from text_blocks
                    for text_block in page.get("text_blocks", []):
                        text = text_block.get("text", "").strip()
                        if text:
                            text_parts.append(text)

                    # Extract table content (LaTeX from Parse)
                    for table in page.get("tables", []):
                        latex = table.get("latex", "").strip()
                        if latex:
                            text_parts.append(latex)
                        desc = table.get("description", "").strip()
                        if desc:
                            text_parts.append(desc)

                    # Extract figure descriptions (from VL model)
                    for figure in page.get("figures", []):
                        desc = figure.get("description", "").strip()
                        if desc:
                            text_parts.append(desc)

                # Combine all text
                combined_text = " ".join(text_parts)

                # Create entry with id and text field for deduplication
                dedup_entry = {
                    "id": f"doc_{line_num}",  # Unique document ID
                    "pdf_path": pdf_path,
                    "text": combined_text,
                }

                outfile.write(json.dumps(dedup_entry) + "\n")
                processed_count += 1

            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                continue

    logger.info(f"Preprocessed {processed_count} documents for deduplication")


def apply_deduplication_results(
    input_path: str, duplicate_ids_path: str, output_path: str
) -> None:
    """Filter out duplicates from original data.

    Args:
        input_path: Path to original extracted data
        duplicate_ids_path: Path to duplicate IDs parquet from dedup workflow
        output_path: Path to write deduplicated data
    """
    logger.info("Applying deduplication results...")

    # Load duplicate IDs
    try:
        duplicate_df = pd.read_parquet(duplicate_ids_path)
        duplicate_ids = set(duplicate_df["id"].tolist())
        logger.info(f"Found {len(duplicate_ids)} duplicate documents to remove")
    except Exception as e:
        logger.warning(f"Could not load duplicate IDs: {e}")
        logger.info("Proceeding without removing duplicates")
        duplicate_ids = set()

    # Filter original data
    kept_count = 0
    removed_count = 0

    with open(input_path) as infile, open(output_path, "w") as outfile:
        for line_num, line in enumerate(infile):
            doc_id = f"doc_{line_num}"

            if doc_id not in duplicate_ids:
                outfile.write(line)
                kept_count += 1
            else:
                removed_count += 1

    logger.info(f"Kept {kept_count} documents, removed {removed_count} duplicates")


def main():
    """Run deduplication pipeline."""
    # Setup paths
    script_dir = Path(__file__).parent
    input_path = script_dir / "data" / "extracted" / "extracted_data.jsonl"
    output_dir = script_dir / "data" / "dedup"
    output_path = output_dir / "deduplicated_data.jsonl"

    # Intermediate paths
    cache_dir = output_dir / "cache"
    temp_preprocessed = output_dir / "temp_preprocessed.jsonl"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting deduplication pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Check input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run 2_run_extraction.py first")
        return

    # Step 1: Preprocess data for deduplication
    preprocess_for_deduplication(str(input_path), str(temp_preprocessed))

    # Step 2: Run fuzzy deduplication workflow
    logger.info("Running fuzzy deduplication workflow...")

    # Initialize Ray
    ray_client = RayClient()
    ray_client.start()

    try:
        # Create fuzzy deduplication workflow
        fuzzy_workflow = FuzzyDeduplicationWorkflow(
            input_path=str(temp_preprocessed),
            cache_path=str(cache_dir),
            output_path=str(output_dir),
            input_filetype="jsonl",
            text_field="text",
            perform_removal=False,  # We'll handle removal ourselves
            # MinHash + LSH parameters
            seed=42,
            char_ngrams=24,
            num_bands=20,
            minhashes_per_band=13,
            use_64_bit_hash=False,
        )

        # Create executor
        executor = XennaExecutor()

        # Run workflow
        logger.info("Executing fuzzy deduplication workflow...")
        fuzzy_workflow.run(executor)

        logger.info("Fuzzy deduplication workflow completed")

        # Step 3: Apply deduplication results to original data
        duplicate_ids_path = output_dir / "duplicates.parquet"
        apply_deduplication_results(
            str(input_path), str(duplicate_ids_path), str(output_path)
        )

        logger.info(f"Deduplication complete! Results written to {output_path}")

    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        logger.exception("Full traceback:")
        raise

    finally:
        ray_client.stop()

        # Clean up temporary files
        if temp_preprocessed.exists():
            temp_preprocessed.unlink()
            logger.debug("Cleaned up temporary preprocessed file")


if __name__ == "__main__":
    main()
