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
Download PDF files and prepare a JSONL manifest for the extraction pipeline.

Two modes:
  1. URL download: fetch PDFs from datasets.json using NeMo Curator's
     download pipeline (URLGenerationStage → DocumentDownloadStage).
  2. Local PDFs: scan a directory and write the manifest directly
     (no Ray needed).

Both modes produce the same output: a JSONL manifest with one entry
per PDF ({"pdf_path": "/absolute/path/to.pdf"}).

Usage:
    # Download from configured URL source
    python 0_download_and_prepare_data.py --dataset NVIDIA_DATASHEETS

    # Use local PDFs instead
    python 0_download_and_prepare_data.py --pdf-dir /path/to/my/pdfs

    # List available datasets
    python 0_download_and_prepare_data.py --list

    # Download only a few files (for testing)
    python 0_download_and_prepare_data.py --dataset NVIDIA_DATASHEETS --max-files 2

Output:
    data/raw/pdf_files.jsonl  -  JSONL manifest for 1_run_extraction.py
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.pdf.download import PDFDownloader, PDFURLGenerator
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_CONFIG_PATH = SCRIPT_DIR / "datasets.json"


def list_datasets(config_path: Path) -> None:
    """Print available datasets from datasets.json."""
    with open(config_path) as f:
        config = json.load(f)

    print("\nAvailable PDF datasets:\n")
    for name, info in config.items():
        source = info.get("source", "url")
        if source != "url":
            continue
        n_urls = len(info.get("urls", []))
        desc = info.get("description", "")
        print(f"  {name}")
        print(f"    Files: {n_urls}")
        print(f"    {desc}\n")


def write_manifest(pdf_paths: list[str], manifest_path: Path) -> None:
    """Write a JSONL manifest from a list of PDF file paths."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        for pdf_path in sorted(pdf_paths):
            entry = {"pdf_path": str(Path(pdf_path).resolve())}
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Manifest written: {manifest_path} ({len(pdf_paths)} PDFs)")


def scan_local_dir(pdf_dir: Path) -> list[str]:
    """Scan a directory for PDF files and return their absolute paths."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")
    for pdf_file in pdf_files:
        size_kb = pdf_file.stat().st_size / 1024
        logger.info(f"  {pdf_file.name} ({size_kb:.0f} KB)")

    return [str(p.resolve()) for p in pdf_files]


def build_pipeline(
    config_path: str,
    dataset_name: str,
    download_dir: str,
    max_files: int | None = None,
) -> Pipeline:
    """Build the PDF download pipeline using NeMo Curator stages.

    Args:
        config_path: Path to datasets.json.
        dataset_name: Dataset key in datasets.json.
        download_dir: Directory to save downloaded PDFs.
        max_files: Optional limit on files to download.

    Returns:
        Configured Pipeline.
    """
    pipeline = Pipeline(
        name="pdf_download",
        description="Download PDFs from configured URL sources",
    )

    pipeline.add_stage(URLGenerationStage(
        url_generator=PDFURLGenerator(
            config_path=config_path,
            dataset_name=dataset_name,
            max_files=max_files,
        ),
    ))

    pipeline.add_stage(DocumentDownloadStage(
        downloader=PDFDownloader(
            download_dir=download_dir,
            verbose=True,
        ),
    ))

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PDFs and prepare JSONL manifest for extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="NVIDIA_DATASHEETS",
        help="Dataset name from datasets.json",
    )
    parser.add_argument(
        "--pdf-dir", type=str, default=None,
        help="Use local PDFs from this directory instead of downloading",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(SCRIPT_DIR / "data" / "raw" / "pdfs"),
        help="Directory to save downloaded PDFs",
    )
    parser.add_argument(
        "--manifest", type=str,
        default=str(SCRIPT_DIR / "data" / "raw" / "pdf_files.jsonl"),
        help="Output path for the JSONL manifest",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(DATASETS_CONFIG_PATH),
        help="Path to datasets.json config file",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of files to download (for testing)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets(Path(args.config))
        return

    manifest_path = Path(args.manifest)

    if args.pdf_dir:
        # Local mode: scan directory, write manifest directly
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            sys.exit(1)

        pdf_paths = scan_local_dir(pdf_dir)
        write_manifest(pdf_paths, manifest_path)
    else:
        # Download mode: use NeMo Curator pipeline
        ray_client = RayClient()
        ray_client.start()

        try:
            pipeline = build_pipeline(
                config_path=args.config,
                dataset_name=args.dataset,
                download_dir=args.output_dir,
                max_files=args.max_files,
            )

            logger.info("\n" + pipeline.describe())
            result_tasks = pipeline.run()

            # Collect downloaded file paths from FileGroupTask results
            pdf_paths = []
            for task in result_tasks or []:
                pdf_paths.extend(task.data)

            if not pdf_paths:
                logger.error("No PDFs downloaded")
                sys.exit(1)

            write_manifest(pdf_paths, manifest_path)
        finally:
            ray_client.stop()


if __name__ == "__main__":
    main()
