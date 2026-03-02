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

"""Download stages for PDF documents.

Provides URLGenerator and DocumentDownloader subclasses for downloading
PDFs from URL sources configured in a datasets.json file.
"""

import json
import urllib.request
from pathlib import Path

from loguru import logger

from nemo_curator.stages.text.download.base.download import DocumentDownloader
from nemo_curator.stages.text.download.base.url_generation import URLGenerator


class PDFURLGenerator(URLGenerator):
    """Generate PDF download URLs from a datasets.json config file.

    Reads a JSON config file with named datasets, each containing a list
    of direct-download PDF URLs.

    Args:
        config_path: Path to datasets.json.
        dataset_name: Key in the config file to read URLs from.
        max_files: Optional limit on number of URLs to return.
    """

    def __init__(
        self,
        config_path: str,
        dataset_name: str,
        max_files: int | None = None,
    ):
        self._config_path = config_path
        self._dataset_name = dataset_name
        self._max_files = max_files

    def generate_urls(self) -> list[str]:
        """Read datasets.json and return PDF URLs for the configured dataset."""
        config_path = Path(self._config_path)
        if not config_path.exists():
            msg = f"Dataset config not found: {config_path}"
            raise FileNotFoundError(msg)

        with open(config_path) as f:
            config = json.load(f)

        if self._dataset_name not in config:
            available = ", ".join(config.keys())
            msg = f"Unknown dataset: {self._dataset_name}. Available: {available}"
            raise KeyError(msg)

        dataset = config[self._dataset_name]
        if dataset.get("source") != "url":
            msg = (
                f"Dataset '{self._dataset_name}' is not a URL source. "
                "Use --pdf-dir for local PDFs instead."
            )
            raise ValueError(msg)

        urls = dataset.get("urls", [])
        if self._max_files is not None:
            urls = urls[: self._max_files]

        logger.info(f"Generated {len(urls)} URLs from dataset '{self._dataset_name}'")
        return urls


class PDFDownloader(DocumentDownloader):
    """Download PDF files from direct URLs.

    Uses urllib to download PDFs. Inherits atomic temp-file handling
    and idempotency (skip existing files) from the base class.

    Args:
        download_dir: Directory to store downloaded PDFs.
        verbose: If True, logs detailed download information.
    """

    def _get_output_filename(self, url: str) -> str:
        """Extract filename from URL, ensuring .pdf extension."""
        filename = url.split("/")[-1].split("?")[0]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        return filename

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download a PDF from URL to the specified path."""
        try:
            urllib.request.urlretrieve(url, path)  # noqa: S310
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        return True, None
