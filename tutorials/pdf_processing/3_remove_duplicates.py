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
Remove Duplicates from Extracted PDF Data

Applies two stages of deduplication:
1. Fuzzy Deduplication (MinHash + LSH) — catches near-duplicate text
2. Semantic Deduplication (embeddings + clustering) — catches documents
   that say the same thing in different words

For Q&A curation, semantic dedup is important because two datasheets
describing the same GPU with different wording would produce redundant
Q&A pairs.

Usage:
    # Run both fuzzy and semantic dedup (default)
    python 3_remove_duplicates.py

    # Fuzzy dedup only (no GPU required)
    python 3_remove_duplicates.py --skip-semantic

    # Custom paths
    python 3_remove_duplicates.py --input data/extracted/extracted_data.jsonl \
                                  --output data/dedup/deduplicated_data.jsonl

Input:
    data/extracted/extracted_data.jsonl

Output:
    data/dedup/deduplicated_data.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow

SCRIPT_DIR = Path(__file__).resolve().parent


def extract_text_from_entry(entry: dict) -> str:
    """Extract combined text from all pages and modalities of a document."""
    text_parts = []
    for page in entry.get("pages", []):
        full_text = page.get("full_text", "").strip()
        if full_text:
            text_parts.append(full_text)
            continue

        for block in page.get("text_blocks", []):
            text = block.get("text", "").strip()
            if text:
                text_parts.append(text)

        for table in page.get("tables", []):
            for field in ("latex", "description"):
                val = table.get(field, "").strip()
                if val:
                    text_parts.append(val)

        for figure in page.get("figures", []):
            desc = figure.get("description", "").strip()
            if desc:
                text_parts.append(desc)

    return " ".join(text_parts)


def preprocess_to_jsonl(input_path: str, output_path: str) -> int:
    """Preprocess extracted data into a flat JSONL with id + text fields.

    Returns:
        Number of documents processed.
    """
    logger.info("Preprocessing extracted data for deduplication...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path) as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = extract_text_from_entry(entry)
            outfile.write(json.dumps({
                "id": f"doc_{i}",
                "pdf_path": entry.get("pdf_path", ""),
                "text": text,
            }) + "\n")
            count += 1

    logger.info(f"Preprocessed {count} documents")
    return count


def preprocess_to_parquet(input_path: str, output_path: str) -> int:
    """Preprocess extracted data into parquet with id + text fields.

    Required for semantic dedup which reads parquet.

    Returns:
        Number of documents processed.
    """
    logger.info("Preprocessing extracted data to parquet for semantic dedup...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ids, pdf_paths, texts = [], [], []
    with open(input_path) as infile:
        for i, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ids.append(f"doc_{i}")
            pdf_paths.append(entry.get("pdf_path", ""))
            texts.append(extract_text_from_entry(entry))

    table = pa.table({"id": ids, "pdf_path": pdf_paths, "text": texts})
    pq.write_table(table, output_path)

    logger.info(f"Wrote {len(ids)} documents to {output_path}")
    return len(ids)


def collect_duplicate_ids(*parquet_paths: str) -> set[str]:
    """Collect duplicate IDs from one or more parquet files."""
    all_ids: set[str] = set()
    for path in parquet_paths:
        try:
            df = pd.read_parquet(path)
            all_ids.update(df["id"].tolist())
            logger.info(f"Loaded {len(df)} duplicate IDs from {path}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not load {path}: {e}")
    return all_ids


def apply_deduplication(
    input_path: str, duplicate_ids: set[str], output_path: str
) -> None:
    """Filter out duplicates from the original extracted data."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    removed = 0
    with open(input_path) as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            if f"doc_{i}" not in duplicate_ids:
                outfile.write(line)
                kept += 1
            else:
                removed += 1

    logger.info(f"Dedup result: kept {kept}, removed {removed}")


def run_fuzzy_dedup(
    preprocessed_jsonl: str,
    cache_dir: str,
    output_dir: str,
    executor: XennaExecutor,
) -> str:
    """Run fuzzy deduplication. Returns path to duplicates parquet."""
    logger.info("Running fuzzy deduplication (MinHash + LSH)...")

    fuzzy_output = os.path.join(output_dir, "fuzzy")
    fuzzy_cache = os.path.join(cache_dir, "fuzzy")

    workflow = FuzzyDeduplicationWorkflow(
        input_path=preprocessed_jsonl,
        cache_path=fuzzy_cache,
        output_path=fuzzy_output,
        input_filetype="jsonl",
        text_field="text",
        perform_removal=False,
        seed=42,
        char_ngrams=24,
        num_bands=20,
        minhashes_per_band=13,
        use_64_bit_hash=False,
    )
    workflow.run(executor)

    duplicates_path = os.path.join(fuzzy_output, "duplicates.parquet")
    logger.info("Fuzzy deduplication complete")
    return duplicates_path


def run_semantic_dedup(
    preprocessed_parquet: str,
    cache_dir: str,
    output_dir: str,
    executor: XennaExecutor,
    embedding_model: str = "intfloat/e5-base-v2",
    n_clusters: int = 5,
    eps: float = 0.1,
) -> str:
    """Run semantic deduplication. Returns path to duplicates directory."""
    logger.info("Running semantic deduplication (embeddings + clustering)...")

    sem_output = os.path.join(output_dir, "semantic")
    sem_cache = os.path.join(cache_dir, "semantic")

    # Step 1: Generate embeddings using a vLLM embedding pipeline
    logger.info(f"Generating embeddings with {embedding_model}...")
    embeddings_dir = os.path.join(sem_cache, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage
    from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage
    from nemo_curator.stages.text.io.writer.parquet import ParquetWriterStage

    embed_pipeline = Pipeline(
        name="embedding_generation",
        description="Generate text embeddings for semantic dedup",
    )
    embed_pipeline.add_stage(ParquetReaderStage(input_path=preprocessed_parquet))
    embed_pipeline.add_stage(VLLMEmbeddingModelStage(
        model_identifier=embedding_model,
        text_field="text",
        embedding_field="embeddings",
    ))
    embed_pipeline.add_stage(ParquetWriterStage(output_path=embeddings_dir))
    embed_pipeline.run(executor)

    logger.info("Embeddings generated, running semantic clustering...")

    # Step 2: Run semantic dedup workflow on embeddings
    workflow = SemanticDeduplicationWorkflow(
        input_path=embeddings_dir,
        output_path=sem_output,
        cache_path=sem_cache,
        n_clusters=n_clusters,
        id_field="id",
        embedding_field="embeddings",
        eps=eps,
        input_filetype="parquet",
        which_to_keep="hard",
    )
    workflow.run(executor)

    duplicates_path = os.path.join(sem_output, "duplicates")
    logger.info("Semantic deduplication complete")
    return duplicates_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicates from extracted PDF data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default=str(SCRIPT_DIR / "data" / "extracted" / "extracted_data.jsonl"),
        help="Input JSONL from extraction pipeline",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(SCRIPT_DIR / "data" / "dedup" / "deduplicated_data.jsonl"),
        help="Output JSONL for deduplicated data",
    )
    parser.add_argument(
        "--skip-semantic", action="store_true",
        help="Skip semantic dedup (only run fuzzy dedup, no GPU needed)",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="intfloat/e5-base-v2",
        help="Embedding model for semantic dedup",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5,
        help="Number of clusters for semantic dedup (increase for larger datasets)",
    )
    parser.add_argument(
        "--sem-eps", type=float, default=0.1,
        help="Epsilon threshold for semantic duplicate identification",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run 2_run_extraction.py first")
        return

    output_path = Path(args.output)
    output_dir = output_path.parent
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess
    temp_jsonl = str(output_dir / "temp_preprocessed.jsonl")
    temp_parquet = str(output_dir / "temp_preprocessed.parquet")

    doc_count = preprocess_to_jsonl(str(input_path), temp_jsonl)
    if doc_count == 0:
        logger.error("No documents to deduplicate")
        return

    logger.info(f"Starting deduplication pipeline ({doc_count} documents)")

    ray_client = RayClient()
    ray_client.start()

    try:
        executor = XennaExecutor()
        all_duplicate_ids: set[str] = set()

        # Stage 1: Fuzzy dedup
        fuzzy_dupes_path = run_fuzzy_dedup(
            temp_jsonl, str(cache_dir), str(output_dir), executor,
        )
        all_duplicate_ids.update(collect_duplicate_ids(fuzzy_dupes_path))

        # Stage 2: Semantic dedup (optional)
        if not args.skip_semantic:
            preprocess_to_parquet(str(input_path), temp_parquet)
            sem_dupes_path = run_semantic_dedup(
                temp_parquet, str(cache_dir), str(output_dir), executor,
                embedding_model=args.embedding_model,
                n_clusters=args.n_clusters,
                eps=args.sem_eps,
            )
            # Semantic dedup outputs a directory of parquet files
            sem_dupes_dir = Path(sem_dupes_path)
            if sem_dupes_dir.exists():
                for pq_file in sem_dupes_dir.glob("*.parquet"):
                    all_duplicate_ids.update(collect_duplicate_ids(str(pq_file)))

        # Apply combined results
        logger.info(f"Total unique duplicate IDs: {len(all_duplicate_ids)}")
        apply_deduplication(str(input_path), all_duplicate_ids, str(output_path))
        logger.info(f"Deduplicated data written to {output_path}")

    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        logger.exception("Full traceback:")
        raise

    finally:
        ray_client.stop()
        # Cleanup temp files
        for temp in (temp_jsonl, temp_parquet):
            if Path(temp).exists():
                Path(temp).unlink()


if __name__ == "__main__":
    main()
