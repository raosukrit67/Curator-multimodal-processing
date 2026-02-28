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
Prepare a PDF directory into a JSONL manifest for the extraction pipeline.

This script takes a directory of PDF files (from 0_download.py or your own
collection) and produces the pdf_files.jsonl expected by 2_run_extraction.py.

Both user paths converge here:
  Path A: 0_download.py → data/raw/pdfs/  → 1_prepare_data.py → pdf_files.jsonl
  Path B: your local PDFs directory        → 1_prepare_data.py → pdf_files.jsonl

Usage:
    # From downloaded PDFs (default directory)
    python 1_prepare_data.py

    # From a custom PDF directory
    python 1_prepare_data.py --pdf-dir /path/to/my/pdfs

    # Custom output location
    python 1_prepare_data.py --pdf-dir ./pdfs --output data/raw/pdf_files.jsonl

Output:
    pdf_files.jsonl with one entry per PDF: {"pdf_path": "/absolute/path/to.pdf"}
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent


def create_manifest(pdf_dir: Path, output_path: Path) -> None:
    """Scan a directory for PDFs and write a JSONL manifest.

    Args:
        pdf_dir: Directory containing PDF files.
        output_path: Path to write the manifest JSONL.
    """
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pdf_file in pdf_files:
            entry = {"pdf_path": str(pdf_file.resolve())}
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Manifest written: {output_path} ({len(pdf_files)} PDFs)")

    for pdf_file in pdf_files:
        size_kb = pdf_file.stat().st_size / 1024
        logger.info(f"  {pdf_file.name} ({size_kb:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSONL manifest from a directory of PDF files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf-dir", type=str,
        default=str(SCRIPT_DIR / "data" / "raw" / "pdfs"),
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(SCRIPT_DIR / "data" / "raw" / "pdf_files.jsonl"),
        help="Output path for the manifest JSONL",
    )

    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        logger.info("Run 0_download.py first, or provide --pdf-dir pointing to your PDFs")
        sys.exit(1)

    create_manifest(pdf_dir, Path(args.output))


if __name__ == "__main__":
    main()
