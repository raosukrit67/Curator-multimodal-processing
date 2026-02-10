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
Download PDFs from URLs listed in source/pdf_urls.jsonl.

This script downloads PDF files from URLs and creates a manifest of local file paths
for use in the multimodal extraction pipeline.

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

from loguru import logger


def download_pdf(url: str, output_path: str) -> bool:
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


def main():
    """Download PDFs and create manifest."""
    # Setup paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "source" / "pdf_urls.jsonl"
    output_dir = script_dir / "source" / "pdfs"
    manifest_dir = script_dir / "extraction_results"
    manifest_file = manifest_dir / "downloaded_pdfs.jsonl"

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Read URLs
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please create source/pdf_urls.jsonl with PDF URLs")
        return

    urls = []
    with open(input_file) as f:
        for line in f:
            urls.append(json.loads(line))

    logger.info(f"Found {len(urls)} URLs to download")

    # Download PDFs
    downloaded_paths = []
    for entry in urls:
        url = entry["url"]
        filename = entry.get("filename", url.split("/")[-1])

        output_path = output_dir / filename

        if download_pdf(url, str(output_path)):
            downloaded_paths.append({"pdf_path": str(output_path.absolute())})

    # Write manifest
    logger.info(f"Writing manifest to {manifest_file}")
    with open(manifest_file, "w") as f:
        for entry in downloaded_paths:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Successfully downloaded {len(downloaded_paths)} / {len(urls)} PDFs")
    logger.info(f"Manifest written to: {manifest_file}")


if __name__ == "__main__":
    main()
