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
Download PDF files from configured URL sources.

This step is OPTIONAL. Skip it if you already have PDFs locally.

PDF sources are configured in datasets.json. Each entry has a name,
description, and a list of direct-download PDF URLs. Users can add
their own datasets by editing datasets.json.

Usage:
    # List available datasets
    python 0_download.py --list

    # Download the default dataset (NVIDIA datasheets)
    python 0_download.py --dataset NVIDIA_DATASHEETS

    # Download to a custom directory
    python 0_download.py --dataset NVIDIA_DATASHEETS --output-dir /data/pdfs

    # Download only a few files (for testing)
    python 0_download.py --dataset NVIDIA_DATASHEETS --max-files 2

Output:
    <output-dir>/*.pdf  -  Downloaded PDF files
"""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_CONFIG_PATH = SCRIPT_DIR / "datasets.json"


def load_datasets_config(config_path: Path | None = None) -> dict:
    """Load dataset configuration from JSON file."""
    path = config_path or DATASETS_CONFIG_PATH
    if not path.exists():
        logger.error(f"Dataset config not found: {path}")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


def list_datasets(config: dict) -> None:
    """Print available datasets."""
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


def download_pdf(url: str, output_path: str) -> bool:
    """Download a single PDF file."""
    try:
        urllib.request.urlretrieve(url, output_path)  # noqa: S310
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"Downloaded ({size_kb:.0f} KB): {Path(output_path).name}")
    return True


def download_dataset(
    dataset_name: str,
    config: dict,
    output_dir: Path,
    max_files: int | None = None,
    workers: int = 4,
) -> list[str]:
    """Download PDFs for a named dataset.

    Args:
        dataset_name: Key in datasets.json.
        config: Full datasets config dict.
        output_dir: Directory to save PDFs into.
        max_files: Limit number of files to download.
        workers: Number of parallel download threads.

    Returns:
        List of successfully downloaded file paths.
    """
    if dataset_name not in config:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available: {', '.join(config.keys())}")
        sys.exit(1)

    dataset = config[dataset_name]
    if dataset.get("source") != "url":
        logger.error(f"Dataset '{dataset_name}' is not a URL source. Use --pdf-dir with 1_prepare_data.py instead.")
        sys.exit(1)

    urls = dataset.get("urls", [])
    if not urls:
        logger.warning(f"No URLs configured for {dataset_name}")
        return []

    if max_files is not None:
        urls = urls[:max_files]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {len(urls)} PDFs from {dataset_name} to {output_dir}")

    # Build download tasks (skip existing)
    tasks = []
    for url in urls:
        filename = url.split("/")[-1].split("?")[0]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        output_path = str(output_dir / filename)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Already exists: {filename}")
            tasks.append((url, output_path, True))
        else:
            tasks.append((url, output_path, False))

    downloaded = [path for _, path, done in tasks if done]
    to_download = [(url, path) for url, path, done in tasks if not done]

    if to_download:
        with ThreadPoolExecutor(max_workers=min(workers, len(to_download))) as executor:
            futures = {
                executor.submit(download_pdf, url, path): path
                for url, path in to_download
            }
            for future in as_completed(futures):
                path = futures[future]
                if future.result():
                    downloaded.append(path)

    logger.info(f"Downloaded {len(downloaded)}/{len(urls)} PDFs")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PDF files from configured URL sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="NVIDIA_DATASHEETS",
        help="Dataset name from datasets.json",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(SCRIPT_DIR / "data" / "raw" / "pdfs"),
        help="Directory to save downloaded PDFs",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to datasets.json config file",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of files to download (for testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download threads",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_datasets_config(config_path)

    if args.list:
        list_datasets(config)
        return

    downloaded = download_dataset(
        args.dataset, config, Path(args.output_dir),
        max_files=args.max_files, workers=args.workers,
    )
    if not downloaded:
        logger.error("No PDFs downloaded")
        sys.exit(1)


if __name__ == "__main__":
    main()
