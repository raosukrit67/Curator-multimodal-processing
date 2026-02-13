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
Download PDFs from URLs using Curator Pipeline

This script downloads PDF files from URLs using a Pipeline with custom stages
and creates a manifest of local file paths for use in the extraction pipeline.

Pipeline Stages:
1. URLReaderStage - Read URLs from JSONL file
2. URLDownloadStage - Download PDFs from URLs
3. PDFManifestWriterStage - Write manifest of downloaded PDF paths

Input:
    source/pdf_urls.jsonl - JSONL file with entries like:
        {"url": "https://example.com/paper.pdf", "filename": "paper.pdf"}

Output:
    source/pdfs/ - Directory containing downloaded PDF files
    extraction_results/downloaded_pdfs.jsonl - Manifest of local file paths
"""

import json
import os
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, _EmptyTask


class URLReaderStage(ProcessingStage[_EmptyTask, DocumentBatch]):
    """Read URLs from JSONL file."""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.name = "url_reader"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "filename"]

    def process(self, _: _EmptyTask) -> list[DocumentBatch]:
        """Read URLs from JSONL file and create DocumentBatch tasks."""
        logger.info(f"Reading URLs from {self.input_path}")

        # Read URLs from JSONL
        urls = []
        with open(self.input_path) as f:
            for line in f:
                entry = json.loads(line)
                url = entry["url"]
                filename = entry.get("filename", url.split("/")[-1])
                urls.append({"url": url, "filename": filename})

        logger.info(f"Loaded {len(urls)} URLs to download")

        # Create DocumentBatch tasks (one per URL)
        tasks = []
        for i, entry in enumerate(urls):
            df = pd.DataFrame([entry])
            task = DocumentBatch(
                task_id=f"url_{i}",
                dataset_name="urls",
                data=df,
            )
            tasks.append(task)

        return tasks


class URLDownloadStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Download PDFs from URLs."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.name = "url_download"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "filename"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "filename", "pdf_path", "success"]

    def setup(self, worker_metadata=None) -> None:
        """Create output directory."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _download_pdf(self, url: str, output_path: str) -> bool:
        """Download a PDF from URL to local path.

        Args:
            url: URL to download from
            output_path: Local path to save file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file already exists
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"File already exists: {output_path}")
                return True

            logger.info(f"Downloading {url} to {output_path}")

            # Download file
            with urlopen(url) as response:
                data = response.read()

            # Write to file
            with open(output_path, "wb") as f:
                f.write(data)

            logger.info(f"Successfully downloaded {output_path} ({len(data)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Download PDFs for URLs in batch."""
        df = batch.to_pandas()

        pdf_paths = []
        success_flags = []

        for _, row in df.iterrows():
            url = row["url"]
            filename = row["filename"]
            output_path = Path(self.output_dir) / filename

            success = self._download_pdf(url, str(output_path))
            pdf_paths.append(str(output_path.absolute()) if success else "")
            success_flags.append(success)

        df["pdf_path"] = pdf_paths
        df["success"] = success_flags

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class PDFManifestWriterStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Write manifest of downloaded PDF paths."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.name = "manifest_writer"
        self.resources = Resources(cpus=2.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["pdf_path", "success"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup(self, worker_metadata=None) -> None:
        """Create output directory and clear existing manifest."""
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear existing manifest
        if output_path.exists():
            output_path.unlink()

    def process(self, batch: DocumentBatch) -> None:
        """Write successful downloads to manifest."""
        df = batch.to_pandas()

        # Filter successful downloads
        successful_df = df[df["success"]]

        # Write to manifest
        with open(self.output_path, "a") as f:
            for _, row in successful_df.iterrows():
                entry = {"pdf_path": row["pdf_path"]}
                f.write(json.dumps(entry) + "\n")

        logger.info(
            f"Wrote {len(successful_df)} successful downloads to manifest "
            f"({len(df) - len(successful_df)} failed)"
        )

        # Return None to end the pipeline
        return None


def create_download_pipeline(
    input_path: str,
    output_dir: str,
    manifest_path: str,
) -> Pipeline:
    """Create the PDF download pipeline.

    Args:
        input_path: Path to input JSONL file with URLs
        output_dir: Directory to save downloaded PDFs
        manifest_path: Path to output manifest file

    Returns:
        Configured pipeline
    """
    pipeline = Pipeline(
        name="pdf_download",
        description="Download PDFs from URLs and create manifest",
    )

    # Stage 1: Read URLs
    pipeline.add_stage(URLReaderStage(input_path=input_path))

    # Stage 2: Download PDFs
    pipeline.add_stage(URLDownloadStage(output_dir=output_dir))

    # Stage 3: Write manifest
    pipeline.add_stage(PDFManifestWriterStage(output_path=manifest_path))

    return pipeline


def main():
    """Run the PDF download pipeline."""
    # Setup paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "source" / "pdf_urls.jsonl"
    output_dir = script_dir / "source" / "pdfs"
    manifest_file = script_dir / "extraction_results" / "downloaded_pdfs.jsonl"

    # Check input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please create source/pdf_urls.jsonl with PDF URLs")
        logger.info('Example entry: {"url": "https://example.com/paper.pdf", "filename": "paper.pdf"}')
        return

    logger.info("Starting PDF download pipeline")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Manifest: {manifest_file}")

    # Create pipeline
    pipeline = create_download_pipeline(
        input_path=str(input_file),
        output_dir=str(output_dir),
        manifest_path=str(manifest_file),
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
        logger.info(f"Manifest written to: {manifest_file}")

    except Exception as e:
        logger.error(f"Download pipeline failed: {e}")
        logger.exception("Full traceback:")
        raise

    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
