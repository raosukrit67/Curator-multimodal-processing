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

"""Tests for Q&A quality filter logic."""

import json

import pandas as pd

from nemo_curator.tasks import DocumentBatch


def _make_batch(pages_list: list[dict]) -> DocumentBatch:
    df = pd.DataFrame({
        "pdf_path": ["test.pdf"],
        "pages": [json.dumps(pages_list)],
    })
    return DocumentBatch(task_id="test", dataset_name="test", data=df)


def _content_page(page_num: int = 0) -> dict:
    """A page with substantive content suitable for Q&A."""
    return {
        "page_number": page_num,
        "text_blocks": [
            {"class_name": "Title", "bbox": [10, 5, 800, 18], "text": "NVIDIA A100 Specifications"},
            {
                "class_name": "Text", "bbox": [10, 20, 800, 300],
                "text": "The NVIDIA A100 Tensor Core GPU delivers unprecedented acceleration "
                        "at every scale for AI, data analytics, and HPC. Based on the NVIDIA "
                        "Ampere architecture, A100 is the engine of the NVIDIA data center platform.",
            },
        ],
        "tables": [{
            "bbox": [10, 310, 800, 500],
            "latex": "\\begin{tabular}{|l|c|}\\hline Spec & Value"
                     " \\\\\\hline Memory & 80 GB \\\\\\hline\\end{tabular}",
        }],
        "figures": [],
        "full_text": "NVIDIA A100 Specifications\n\nThe NVIDIA A100 Tensor Core GPU delivers...",
    }


def _boilerplate_page(page_num: int = 0) -> dict:
    """A page with only headers/footers."""
    return {
        "page_number": page_num,
        "text_blocks": [
            {"class_name": "Page-Header", "bbox": [10, 5, 800, 15], "text": "NVIDIA Corporation"},
            {"class_name": "Page-Footer", "bbox": [10, 990, 800, 1000], "text": "Page 1 of 4"},
        ],
        "tables": [],
        "figures": [],
        "full_text": "NVIDIA Corporation\n\nPage 1 of 4",
    }


def _figure_page(page_num: int = 0) -> dict:
    """A page with a described figure (visual Q&A ready)."""
    return {
        "page_number": page_num,
        "text_blocks": [
            {"class_name": "Caption", "bbox": [10, 400, 800, 420], "text": "Figure 1: Architecture"},
        ],
        "tables": [],
        "figures": [{
            "bbox": [50, 100, 750, 380],
            "class_name": "Figure",
            "description": "Block diagram showing the A100 GPU architecture"
                           " with 8 GPCs and 128 SMs.",
        }],
        "full_text": "Figure 1: Architecture\n\n[Figure: Block diagram showing the A100 GPU architecture...]",
    }


def _empty_page(page_num: int = 0) -> dict:
    return {"page_number": page_num, "text_blocks": [], "tables": [], "figures": [], "full_text": ""}


class TestExtractionCompletenessLogic:
    def test_all_pages_have_content(self):
        pages = [_content_page(), _content_page(1)]
        ratio = sum(1 for p in pages if p["text_blocks"] or p["tables"] or p["figures"]) / len(pages)
        assert ratio >= 0.5

    def test_mostly_empty_fails(self):
        pages = [_content_page(), _empty_page(1), _empty_page(2), _empty_page(3)]
        ratio = sum(1 for p in pages if p["text_blocks"] or p["tables"] or p["figures"]) / len(pages)
        assert ratio < 0.5


class TestBoilerplateLogic:
    BOILERPLATE_CLASSES = {"Page-Header", "Page-Footer", "Index"}

    def _is_substantive(self, page, min_blocks=1):
        if page.get("tables") or page.get("figures"):
            return True
        substantive = sum(
            1 for b in page.get("text_blocks", [])
            if b.get("class_name") not in self.BOILERPLATE_CLASSES
        )
        return substantive >= min_blocks

    def test_content_page_is_substantive(self):
        assert self._is_substantive(_content_page())

    def test_boilerplate_page_is_not_substantive(self):
        assert not self._is_substantive(_boilerplate_page())

    def test_figure_page_is_substantive(self):
        assert self._is_substantive(_figure_page())

    def test_table_only_page_is_substantive(self):
        page = {"text_blocks": [], "tables": [{"bbox": [0, 0, 1, 1], "latex": "..."}], "figures": []}
        assert self._is_substantive(page)


class TestQAReadinessLogic:
    def _page_is_qa_ready(self, page, min_answer_length=80):
        if page.get("tables"):
            return True
        for fig in page.get("figures", []):
            if fig.get("description", "").strip():
                return True
        for block in page.get("text_blocks", []):
            if len(block.get("text", "")) >= min_answer_length:
                return True
        return False

    def test_content_page_is_qa_ready(self):
        assert self._page_is_qa_ready(_content_page())

    def test_table_page_is_qa_ready(self):
        """Tables produce factual Q&A pairs."""
        page = {"text_blocks": [], "tables": [{"latex": "data"}], "figures": []}
        assert self._page_is_qa_ready(page)

    def test_figure_with_description_is_qa_ready(self):
        assert self._page_is_qa_ready(_figure_page())

    def test_figure_without_description_is_not_qa_ready(self):
        page = {"text_blocks": [], "tables": [], "figures": [{"description": ""}]}
        assert not self._page_is_qa_ready(page)

    def test_short_text_is_not_qa_ready(self):
        """Text too short to form a Q&A pair."""
        page = {"text_blocks": [{"text": "OK"}], "tables": [], "figures": []}
        assert not self._page_is_qa_ready(page)

    def test_boilerplate_only_is_not_qa_ready(self):
        assert not self._page_is_qa_ready(_boilerplate_page())

    def test_empty_page_is_not_qa_ready(self):
        assert not self._page_is_qa_ready(_empty_page())
