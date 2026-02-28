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

"""Tests for PDF utility functions including Nemotron Parse output parsing."""


import numpy as np
import pytest
from PIL import Image

from nemo_curator.utils.pdf_utils import (
    PARSE_CLASS_COLORS,
    VL_CONTENT_TYPES,
    base64_to_image,
    bbox_area,
    bbox_iou,
    crop_image_from_bbox_tuple,
    crop_image_region,
    denormalize_bbox,
    extract_classes_bboxes,
    image_to_base64,
    image_to_numpy,
    normalize_bbox,
    numpy_to_image,
    parse_nemotron_output,
    postprocess_text,
    transform_bbox_to_original,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a simple 100x200 RGB test image."""
    return Image.new("RGB", (100, 200), color="red")


@pytest.fixture
def sample_parse_output():
    """Sample Nemotron Parse raw output with multiple region types."""
    return (
        "<x_0.05><y_0.02>"
        "# Introduction to NeMo Curator"
        "<x_0.95><y_0.06><class_Title>"
        "<x_0.05><y_0.08>"
        "NeMo Curator is a scalable data curation toolkit."
        "<x_0.95><y_0.20><class_Text>"
        "<x_0.05><y_0.22>"
        "\\begin{tabular}{|l|c|}\n\\hline\nMethod & Score \\\\\n\\hline\nFuzzy & 0.95 \\\\\n\\hline\n\\end{tabular}"
        "<x_0.95><y_0.45><class_Table>"
        "<x_0.10><y_0.50>"
        "<x_0.90><y_0.80><class_Picture>"
        "<x_0.05><y_0.82>"
        "$E = mc^2$"
        "<x_0.40><y_0.86><class_Formula>"
    )


# =============================================================================
# Image conversion tests
# =============================================================================

class TestImageConversions:
    def test_image_to_base64_roundtrip(self, sample_image):
        b64 = image_to_base64(sample_image)
        assert isinstance(b64, str)
        assert len(b64) > 0

        restored = base64_to_image(b64)
        assert restored.size == sample_image.size

    def test_numpy_roundtrip(self, sample_image):
        arr = image_to_numpy(sample_image)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (200, 100, 3)

        restored = numpy_to_image(arr)
        assert restored.size == sample_image.size


# =============================================================================
# Bounding box tests
# =============================================================================

class TestBoundingBoxOps:
    def test_normalize_bbox(self):
        bbox = {"x": 50, "y": 100, "width": 200, "height": 300}
        result = normalize_bbox(bbox, 1000, 1000)
        assert result == {"x": 0.05, "y": 0.1, "width": 0.2, "height": 0.3}

    def test_denormalize_bbox(self):
        bbox = {"x": 0.05, "y": 0.1, "width": 0.2, "height": 0.3}
        result = denormalize_bbox(bbox, 1000, 1000)
        assert result == {"x": 50, "y": 100, "width": 200, "height": 300}

    def test_bbox_area(self):
        assert bbox_area({"width": 0.5, "height": 0.4}) == pytest.approx(0.2)

    def test_bbox_iou_identical(self):
        bbox = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
        assert bbox_iou(bbox, bbox) == pytest.approx(1.0)

    def test_bbox_iou_no_overlap(self):
        bbox1 = {"x": 0.0, "y": 0.0, "width": 0.2, "height": 0.2}
        bbox2 = {"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.2}
        assert bbox_iou(bbox1, bbox2) == 0.0

    def test_crop_image_region(self, sample_image):
        bbox = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
        cropped = crop_image_region(sample_image, bbox, normalized=True)
        assert cropped.size[0] > 0
        assert cropped.size[1] > 0

    def test_crop_image_from_bbox_tuple(self, sample_image):
        bbox = (10, 20, 80, 150)
        cropped = crop_image_from_bbox_tuple(sample_image, bbox)
        assert cropped.size == (70, 130)

    def test_crop_image_from_bbox_tuple_clamped(self, sample_image):
        """Bbox extending beyond image bounds should be clamped."""
        bbox = (-10, -10, 200, 300)
        cropped = crop_image_from_bbox_tuple(sample_image, bbox)
        assert cropped.size == (100, 200)

    def test_crop_image_from_bbox_tuple_degenerate(self, sample_image):
        """Degenerate bbox (right <= left) should return 1x1 image."""
        bbox = (50, 50, 50, 50)
        cropped = crop_image_from_bbox_tuple(sample_image, bbox)
        assert cropped.size == (1, 1)


# =============================================================================
# Nemotron Parse output parsing tests
# =============================================================================

class TestExtractClassesBboxes:
    def test_basic_parsing(self, sample_parse_output):
        classes, _bboxes, _texts = extract_classes_bboxes(sample_parse_output)

        assert len(classes) == 5
        assert classes[0] == "Title"
        assert classes[1] == "Text"
        assert classes[2] == "Table"
        assert classes[3] == "Picture"
        assert classes[4] == "Formula"

    def test_bbox_values(self, sample_parse_output):
        _classes, bboxes, _texts = extract_classes_bboxes(sample_parse_output)

        # Title bbox
        assert bboxes[0] == pytest.approx((0.05, 0.02, 0.95, 0.06))

    def test_text_content(self, sample_parse_output):
        _classes, _bboxes, texts = extract_classes_bboxes(sample_parse_output)

        assert "Introduction" in texts[0]
        assert "scalable" in texts[1]
        assert "\\begin{tabular}" in texts[2]
        assert texts[3].strip() == ""  # Picture has no text
        assert "$E = mc^2$" in texts[4]

    def test_empty_input(self):
        classes, bboxes, texts = extract_classes_bboxes("")
        assert classes == []
        assert bboxes == []
        assert texts == []

    def test_inline_formula_remapped(self):
        output = "<x_0.1><y_0.1>$x^2$<x_0.5><y_0.2><class_Inline-formula>"
        classes, _bboxes, _texts = extract_classes_bboxes(output)
        assert classes[0] == "Formula"  # Remapped from Inline-formula


class TestTransformBboxToOriginal:
    def test_identity_for_small_image(self):
        """For an image smaller than the target, bbox should roughly map back."""
        # A small image that doesn't need resizing
        bbox = (0.5, 0.5, 0.5, 0.5)
        result = transform_bbox_to_original(bbox, 800, 1000)
        # Should return values in the original image coordinate space
        assert all(isinstance(v, float) for v in result)

    def test_centered_bbox(self):
        """Center of the target should map to center of the original."""
        # For a 1648x2048 image (matches target exactly), center is (0.5, 0.5)
        bbox = (0.5, 0.5, 0.5, 0.5)
        result = transform_bbox_to_original(bbox, 1648, 2048)
        left, top, _right, _bottom = result
        # Should be approximately at the center
        assert abs(left - 824) < 2
        assert abs(top - 1024) < 2

    def test_full_page_bbox(self):
        """A full-page bbox should roughly map to the full original image."""
        # Assuming no padding (image matches target aspect ratio)
        bbox = (0.0, 0.0, 1.0, 1.0)
        result = transform_bbox_to_original(bbox, 1648, 2048)
        left, top, right, bottom = result
        assert left <= 10  # Should be close to 0
        assert top <= 10
        assert right >= 1638
        assert bottom >= 2038


class TestPostprocessText:
    def test_table_latex_passthrough(self):
        latex = "\\begin{tabular}{|l|c|}\\hline A & B \\\\\\hline\\end{tabular}"
        result = postprocess_text(latex, cls="Table", table_format="latex")
        assert result == latex

    def test_table_html_conversion(self):
        latex = "\\begin{tabular}{|l|c|}\\hline A & B \\\\\\hline C & D \\\\\\hline\\end{tabular}"
        result = postprocess_text(latex, cls="Table", table_format="HTML")
        assert "<table>" in result
        assert "<th>" in result or "<td>" in result

    def test_markdown_passthrough(self):
        text = "## Section Title\nSome **bold** text."
        result = postprocess_text(text, cls="Text", text_format="markdown")
        assert result == text

    def test_plain_text_strips_markdown(self):
        text = "## Section Title\nSome **bold** text."
        result = postprocess_text(text, cls="Text", text_format="plain")
        assert "##" not in result
        assert "**" not in result

    def test_blank_text_in_figures(self):
        result = postprocess_text("some text", cls="Picture", blank_text_in_figures=True)
        assert result == ""


class TestParseNemotronOutput:
    def test_full_parse(self, sample_parse_output):
        regions = parse_nemotron_output(
            sample_parse_output,
            image_width=1648,
            image_height=2048,
        )

        assert len(regions) == 5
        assert regions[0]["class_name"] == "Title"
        assert regions[3]["class_name"] == "Picture"
        assert regions[3]["needs_vl"] is True
        assert regions[0]["needs_vl"] is False
        assert regions[1]["needs_vl"] is False

    def test_vl_routing(self, sample_parse_output):
        regions = parse_nemotron_output(
            sample_parse_output,
            image_width=1000,
            image_height=1000,
        )
        vl_regions = [r for r in regions if r["needs_vl"]]
        text_regions = [r for r in regions if not r["needs_vl"]]

        assert len(vl_regions) == 1  # Only Picture
        assert vl_regions[0]["class_name"] == "Picture"
        assert len(text_regions) == 4

    def test_bbox_is_pixel_tuple(self, sample_parse_output):
        regions = parse_nemotron_output(
            sample_parse_output,
            image_width=1000,
            image_height=1000,
        )
        for region in regions:
            bbox = region["bbox"]
            assert isinstance(bbox, tuple)
            assert len(bbox) == 4


# =============================================================================
# Constants tests
# =============================================================================

class TestConstants:
    def test_parse_class_colors_has_main_types(self):
        for cls in ["Title", "Text", "Table", "Picture", "Formula", "Caption"]:
            assert cls in PARSE_CLASS_COLORS

    def test_vl_content_types(self):
        assert "Picture" in VL_CONTENT_TYPES
        assert "Figure" in VL_CONTENT_TYPES
        assert "Text" not in VL_CONTENT_TYPES
        assert "Table" not in VL_CONTENT_TYPES
