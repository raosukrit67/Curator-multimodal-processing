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

The extraction pipeline (1_run_extraction.py) already produces a ``text``
column suitable for dedup, so no preprocessing is needed.

Usage:
    # Run fuzzy dedup (default)
    python 2_remove_duplicates.py --input data/extracted/ --output data/dedup/

    # With semantic dedup
    python 2_remove_duplicates.py --input data/extracted/ --output data/dedup/ \
                                  --semantic

    # Custom dedup parameters
    python 2_remove_duplicates.py --input data/extracted/ --output data/dedup/ \
                                  --char-ngrams 24 --num-bands 20

Input:
    data/extracted/  (directory of JSONL files from 1_run_extraction.py)

Output:
    data/dedup/      (directory of deduplicated JSONL files)
"""

import argparse
import os
import shutil
from pathlib import Path

from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

SCRIPT_DIR = Path(__file__).resolve().parent


def run_fuzzy_dedup(  # noqa: PLR0913
    input_path: str,
    cache_dir: str,
    duplicate_ids_dir: str,
    output_path: str,
    text_field: str = "text",
    char_ngrams: int = 24,
    num_bands: int = 20,
    minhashes_per_band: int = 13,
    seed: int = 42,
) -> None:
    """Run fuzzy deduplication following the math pipeline pattern.

    1. FuzzyDeduplicationWorkflow identifies duplicate IDs.
    2. If duplicates found, TextDuplicatesRemovalWorkflow removes them.
    3. If no duplicates, copies input to output.
    """
    logger.info("Running fuzzy deduplication (MinHash + LSH)...")

    fuzzy_workflow = FuzzyDeduplicationWorkflow(
        input_path=input_path,
        cache_path=cache_dir,
        output_path=duplicate_ids_dir,
        input_filetype="jsonl",
        text_field=text_field,
        perform_removal=False,
        seed=seed,
        char_ngrams=char_ngrams,
        num_bands=num_bands,
        minhashes_per_band=minhashes_per_band,
        use_64_bit_hash=False,
    )
    fuzzy_workflow.run()

    duplicate_ids_path = os.path.join(duplicate_ids_dir, "FuzzyDuplicateIds")
    id_generator_path = os.path.join(duplicate_ids_dir, "fuzzy_id_generator.json")

    if not os.path.exists(duplicate_ids_path):
        logger.info("No duplicates found. Copying input to output directory...")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        shutil.copytree(input_path, output_path)
        logger.info(f"All documents are unique. Copied {input_path} → {output_path}")
    else:
        logger.info("Running text duplicates removal workflow...")
        removal_workflow = TextDuplicatesRemovalWorkflow(
            input_path=input_path,
            ids_to_remove_path=duplicate_ids_path,
            output_path=output_path,
            input_filetype="jsonl",
            id_field="_curator_dedup_id",
            duplicate_id_field="_curator_dedup_id",
            output_filetype="jsonl",
            id_generator_path=id_generator_path,
        )
        removal_workflow.run()

    logger.info("Fuzzy deduplication complete")


def run_semantic_dedup(  # noqa: PLR0913
    input_path: str,
    cache_dir: str,
    output_dir: str,
    embedding_model: str = "intfloat/e5-base-v2",
    n_clusters: int = 5,
    eps: float = 0.1,
) -> None:
    """Run semantic deduplication using embeddings + clustering."""
    logger.info("Running semantic deduplication (embeddings + clustering)...")

    from nemo_curator.backends.xenna import XennaExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow
    from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage
    from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
    from nemo_curator.stages.text.io.writer.parquet import ParquetWriter

    sem_cache = os.path.join(cache_dir, "semantic")
    embeddings_dir = os.path.join(sem_cache, "embeddings")
    sem_output = os.path.join(output_dir, "semantic")
    os.makedirs(embeddings_dir, exist_ok=True)

    # Step 1: Generate embeddings
    logger.info(f"Generating embeddings with {embedding_model}...")
    embed_pipeline = Pipeline(
        name="embedding_generation",
        description="Generate text embeddings for semantic dedup",
    )
    embed_pipeline.add_stage(JsonlReader(file_paths=input_path))
    embed_pipeline.add_stage(VLLMEmbeddingModelStage(
        model_identifier=embedding_model,
        text_field="text",
        embedding_field="embeddings",
    ))
    embed_pipeline.add_stage(ParquetWriter(path=embeddings_dir))

    executor = XennaExecutor()
    embed_pipeline.run(executor)

    # Step 2: Semantic clustering
    logger.info("Embeddings generated, running semantic clustering...")
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

    logger.info("Semantic deduplication complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicates from extracted PDF data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default=str(SCRIPT_DIR / "data" / "extracted"),
        help="Input directory of JSONL files from extraction pipeline",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(SCRIPT_DIR / "data" / "dedup"),
        help="Output directory for deduplicated data",
    )
    parser.add_argument(
        "--semantic", action="store_true",
        help="Also run semantic dedup (requires GPU for embeddings)",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="intfloat/e5-base-v2",
        help="Embedding model for semantic dedup",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5,
        help="Number of clusters for semantic dedup",
    )
    parser.add_argument(
        "--sem-eps", type=float, default=0.1,
        help="Epsilon threshold for semantic duplicate identification",
    )
    parser.add_argument(
        "--char-ngrams", type=int, default=24,
        help="Size of character n-grams for MinHash",
    )
    parser.add_argument(
        "--num-bands", type=int, default=20,
        help="Number of bands for LSH",
    )
    parser.add_argument(
        "--minhashes-per-band", type=int, default=13,
        help="Number of hashes per band",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for minhash permutations",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        logger.info("Please run 1_run_extraction.py first")
        return

    output_path = Path(args.output)
    cache_dir = str(output_path / "cache")
    duplicate_ids_dir = str(output_path / "duplicate_ids")
    fuzzy_output = str(output_path / "fuzzy")
    output_path.mkdir(parents=True, exist_ok=True)

    ray_client = RayClient()
    ray_client.start()

    try:
        # Stage 1: Fuzzy dedup
        run_fuzzy_dedup(
            input_path=str(input_path),
            cache_dir=cache_dir,
            duplicate_ids_dir=duplicate_ids_dir,
            output_path=fuzzy_output,
            char_ngrams=args.char_ngrams,
            num_bands=args.num_bands,
            minhashes_per_band=args.minhashes_per_band,
            seed=args.seed,
        )

        # Stage 2: Semantic dedup (optional)
        if args.semantic:
            run_semantic_dedup(
                input_path=fuzzy_output,
                cache_dir=cache_dir,
                output_dir=str(output_path),
                embedding_model=args.embedding_model,
                n_clusters=args.n_clusters,
                eps=args.sem_eps,
            )

        logger.info(f"Deduplication complete. Results at: {output_path}")

    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        logger.exception("Full traceback:")
        raise

    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
