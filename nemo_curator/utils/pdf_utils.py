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

"""Utility functions for PDF processing operations."""

import base64
import io
import re
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 300) -> Image.Image:
    """Convert a PDF page to a PIL Image using PyMuPDF (Fitz).

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-indexed)
        dpi: Resolution for rendering (default: 300)

    Returns:
        PIL Image of the page
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        msg = "pymupdf is required for PDF to image conversion. Install with: pip install pymupdf"
        raise ImportError(msg) from exc

    try:
        doc = fitz.open(pdf_path)

        if page_num < 0 or page_num >= len(doc):
            doc.close()
            msg = f"Page {page_num} out of range for {pdf_path} (total pages: {len(doc)})"
            raise ValueError(msg)

        page = doc[page_num]

        # Render page to pixmap at specified DPI
        zoom = dpi / 72  # 72 is default DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        doc.close()
    except Exception as exc:
        msg = f"Failed to convert page {page_num} from {pdf_path}: {exc}"
        raise ValueError(msg) from exc
    return img


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    """Convert all pages of a PDF to PIL Images using PyMuPDF (Fitz).

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (default: 300)

    Returns:
        List of PIL Images, one per page
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        msg = "pymupdf is required for PDF to image conversion. Install with: pip install pymupdf"
        raise ImportError(msg) from exc

    doc = fitz.open(pdf_path)
    images = []

    # Render each page to pixmap at specified DPI
    zoom = dpi / 72  # 72 is default DPI
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image
        format: Image format (default: PNG)

    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image.

    Args:
        base64_str: Base64 encoded image string

    Returns:
        PIL Image
    """
    image_bytes = base64.b64decode(base64_str)
    buffer = io.BytesIO(image_bytes)
    return Image.open(buffer)


def numpy_to_image(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image.

    Args:
        arr: Numpy array with shape (H, W, 3) or (H, W)

    Returns:
        PIL Image
    """
    return Image.fromarray(arr.astype(np.uint8))


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array.

    Args:
        image: PIL Image

    Returns:
        Numpy array with shape (H, W, 3) or (H, W)
    """
    return np.array(image)


def normalize_bbox(bbox: dict[str, float], image_width: float, image_height: float) -> dict[str, float]:
    """Normalize bounding box coordinates to [0, 1] range.

    Args:
        bbox: Bounding box with keys 'x', 'y', 'width', 'height' in pixels
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        Normalized bounding box with values in [0, 1]
    """
    return {
        "x": bbox["x"] / image_width,
        "y": bbox["y"] / image_height,
        "width": bbox["width"] / image_width,
        "height": bbox["height"] / image_height,
    }


def denormalize_bbox(bbox: dict[str, float], image_width: float, image_height: float) -> dict[str, int]:
    """Convert normalized bounding box to pixel coordinates.

    Args:
        bbox: Bounding box with normalized values in [0, 1]
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        Bounding box with pixel coordinates
    """
    return {
        "x": int(bbox["x"] * image_width),
        "y": int(bbox["y"] * image_height),
        "width": int(bbox["width"] * image_width),
        "height": int(bbox["height"] * image_height),
    }


def crop_image_region(image: Image.Image, bbox: dict[str, Any], normalized: bool = True) -> Image.Image:
    """Crop a region from an image using bounding box.

    Args:
        image: PIL Image
        bbox: Bounding box with keys 'x', 'y', 'width', 'height'
        normalized: Whether bbox coordinates are normalized (default: True)

    Returns:
        Cropped PIL Image
    """
    width, height = image.size

    if normalized:
        bbox = denormalize_bbox(bbox, width, height)

    # Convert to (left, top, right, bottom) format for PIL crop
    left = bbox["x"]
    top = bbox["y"]
    right = left + bbox["width"]
    bottom = top + bbox["height"]

    # Clamp coordinates to image bounds
    left = max(0, min(left, width))
    right = max(0, min(right, width))
    top = max(0, min(top, height))
    bottom = max(0, min(bottom, height))

    return image.crop((left, top, right, bottom))


def bbox_area(bbox: dict[str, float]) -> float:
    """Calculate the area of a bounding box.

    Args:
        bbox: Bounding box with keys 'width' and 'height'

    Returns:
        Area of the bounding box
    """
    return bbox["width"] * bbox["height"]


def bbox_iou(bbox1: dict[str, float], bbox2: dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box with keys 'x', 'y', 'width', 'height'
        bbox2: Second bounding box with keys 'x', 'y', 'width', 'height'

    Returns:
        IoU value between 0 and 1
    """
    # Get coordinates
    x1_min = bbox1["x"]
    y1_min = bbox1["y"]
    x1_max = x1_min + bbox1["width"]
    y1_max = y1_min + bbox1["height"]

    x2_min = bbox2["x"]
    y2_min = bbox2["y"]
    x2_max = x2_min + bbox2["width"]
    y2_max = y2_min + bbox2["height"]

    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0

    intersection_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # Calculate union
    bbox1_area = bbox_area(bbox1)
    bbox2_area = bbox_area(bbox2)
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# =============================================================================
# Nemotron Parse output parsing
# =============================================================================
#
# The Parse model ships postprocessing.py and latex2html.py on HuggingFace.
# We try to import from the downloaded model first, falling back to vendored
# implementations if the model isn't available locally.

# Regex for extracting classes, bboxes, and text from Parse output.
# Format: <x_N><y_N>content<x_N><y_N><class_Type>
_RE_EXTRACT_CLASS_BBOX = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>"
    r"(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>"
    r"<class_([^>]+)>",
    re.DOTALL,
)

# Color map for Parse content classes (for visualization)
PARSE_CLASS_COLORS: dict[str, str] = {
    "Title": "#9B59B6",
    "Section": "#2C3E50",
    "Text": "#27AE60",
    "List-Item": "#1ABC9C",
    "Table": "#E74C3C",
    "Picture": "#3498DB",
    "Figure": "#3498DB",
    "Formula": "#E67E22",
    "Caption": "#F1C40F",
    "Page-Header": "#95A5A6",
    "Page-Footer": "#7F8C8D",
    "Bibliography": "#8B4513",
    "Footnote": "#BDC3C7",
    "Index": "#AAB7B8",
}

# Content types that should be routed to the VL model for description
VL_CONTENT_TYPES = {"Picture", "Figure", "Chart"}

# Content types that have text already extracted by Parse
TEXT_CONTENT_TYPES = {
    "Title", "Section", "Text", "List-Item", "Caption",
    "Page-Header", "Page-Footer", "Bibliography", "Footnote",
    "Index", "Formula",
}


def _try_import_model_postprocessing(model_path: str | None = None):
    """Try to import postprocessing from the downloaded Parse model.

    Args:
        model_path: Path to the downloaded model directory.

    Returns:
        Tuple of (extract_classes_bboxes, transform_bbox_to_original, postprocess_text)
        or None if import fails.
    """
    if model_path is None:
        return None

    import importlib.util
    import os

    pp_path = os.path.join(model_path, "postprocessing.py")
    if not os.path.exists(pp_path):
        return None

    try:
        spec = importlib.util.spec_from_file_location("nemotron_postprocessing", pp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        logger.debug(f"Could not import model postprocessing from {pp_path}: {e}")
        return None
    return (
        mod.extract_classes_bboxes,
        mod.transform_bbox_to_original,
        mod.postprocess_text,
    )


def extract_classes_bboxes(text: str) -> tuple[list[str], list[tuple[float, ...]], list[str]]:
    """Extract semantic classes, bounding boxes, and text from Nemotron Parse output.

    Parses the interleaved token format:
        <x_N><y_N>content<x_N><y_N><class_Type>

    Args:
        text: Raw output string from Nemotron Parse model.

    Returns:
        Tuple of (classes, bboxes, texts) where:
            - classes: list of semantic class labels (e.g., "Title", "Text", "Table")
            - bboxes: list of (x1, y1, x2, y2) tuples in the model's normalized space
            - texts: list of extracted text content strings
    """
    classes = []
    bboxes = []
    texts = []

    for m in _RE_EXTRACT_CLASS_BBOX.finditer(text):
        x1, y1, content, x2, y2, cls = m.groups()
        # Remap Inline-formula to Formula (known model quirk)
        if cls == "Inline-formula":
            cls = "Formula"
        classes.append(cls)
        bboxes.append((float(x1), float(y1), float(x2), float(y2)))
        texts.append(content)

    return classes, bboxes, texts


def transform_bbox_to_original(
    bbox: tuple[float, ...],
    original_width: int,
    original_height: int,
    target_w: int = 1648,
    target_h: int = 2048,
) -> tuple[float, float, float, float]:
    """Transform bbox from Parse's internal coordinate space to original image pixels.

    The Parse model internally resizes images to fit within a target_w x target_h grid
    (preserving aspect ratio) and pads to center. Bbox coordinates are normalized
    within this padded grid. This function reverses that transformation.

    Args:
        bbox: (x1, y1, x2, y2) in model's normalized space [0, 1].
        original_width: Width of the original image in pixels.
        original_height: Height of the original image in pixels.
        target_w: Model's internal target width (default: 1648).
        target_h: Model's internal target height (default: 2048).

    Returns:
        (left, top, right, bottom) in original image pixel coordinates.
    """
    aspect_ratio = original_width / original_height
    new_height = original_height
    new_width = original_width

    if original_height > target_h:
        new_height = target_h
        new_width = int(new_height * aspect_ratio)

    if new_width > target_w:
        new_width = target_w
        new_height = int(new_width / aspect_ratio)

    resized_width = new_width
    resized_height = new_height

    # Padding applied to center the resized image in the target grid
    pad_left = (target_w - resized_width) // 2
    pad_top = (target_h - resized_height) // 2

    # Reverse the transformation
    left = ((bbox[0] * target_w) - pad_left) * original_width / resized_width
    right = ((bbox[2] * target_w) - pad_left) * original_width / resized_width
    top = ((bbox[1] * target_h) - pad_top) * original_height / resized_height
    bottom = ((bbox[3] * target_h) - pad_top) * original_height / resized_height

    return left, top, right, bottom


def postprocess_text(
    text: str,
    cls: str = "Text",
    text_format: str = "markdown",
    table_format: str = "latex",
    blank_text_in_figures: bool = False,
) -> str:
    """Post-process extracted text based on its semantic class.

    Args:
        text: Raw extracted text from Parse.
        cls: Semantic class label (e.g., "Text", "Table", "Picture").
        text_format: Output format for text content: "markdown" or "plain".
        table_format: Output format for tables: "latex", "HTML", or "markdown".
        blank_text_in_figures: If True, return empty string for Picture class.

    Returns:
        Post-processed text string.
    """
    if blank_text_in_figures and cls == "Picture":
        return ""

    if cls == "Table":
        if table_format == "HTML":
            text = _latex_table_to_html(text)
        elif table_format == "markdown":
            html = _latex_table_to_html(text)
            text = _html_table_to_markdown(html)
        # else: keep as LaTeX
    elif text_format == "plain":
        text = _convert_mmd_to_plain_text(text)

    return text


def _convert_mmd_to_plain_text(mmd_text: str) -> str:
    """Convert markdown-formatted text to plain text."""
    mmd_text = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", mmd_text, flags=re.DOTALL)
    mmd_text = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", mmd_text, flags=re.DOTALL)
    mmd_text = mmd_text.replace("<br>", "\n")
    mmd_text = re.sub(r"#+\s", "", mmd_text)
    mmd_text = re.sub(r"\*\*(.*?)\*\*", r"\1", mmd_text)
    mmd_text = re.sub(r"\*(.*?)\*", r"\1", mmd_text)
    mmd_text = re.sub(r"(?<!\w)_([^_]+)_", r"\1", mmd_text)
    return mmd_text.strip()


def _latex_table_to_html(latex: str) -> str:
    """Convert LaTeX tabular environment to HTML table.

    Handles basic LaTeX table syntax with \\begin{tabular}, &, \\\\, and \\hline.
    For more complex tables, the model's bundled latex2html.py is preferred.
    """
    # Try importing from model first
    try:
        from latex2html import latex_table_to_html as model_latex_to_html
        return model_latex_to_html(latex)
    except ImportError:
        pass

    # Vendored fallback: basic LaTeX tabular to HTML conversion
    # Extract content between \begin{tabular} and \end{tabular}
    match = re.search(
        r"\\begin\{tabular\}(?:\{[^}]*\})?(.*?)\\end\{tabular\}",
        latex,
        re.DOTALL,
    )
    if not match:
        return f"<p>{latex}</p>"

    content = match.group(1).strip()

    # Split into rows by \\
    rows = re.split(r"\\\\", content)
    html_rows = []
    is_first_data_row = True

    for row in rows:
        row = row.strip()
        if not row or row == "\\hline":
            continue

        # Remove \hline from within the row
        row = row.replace("\\hline", "").strip()
        if not row:
            continue

        cells = [c.strip() for c in row.split("&")]
        tag = "th" if is_first_data_row else "td"
        html_cells = "".join(f"<{tag}>{cell}</{tag}>" for cell in cells)
        html_rows.append(f"<tr>{html_cells}</tr>")

        if is_first_data_row:
            is_first_data_row = False

    return f"<table>{''.join(html_rows)}</table>"


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to markdown table format."""
    # Simple conversion: extract rows and cells
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    if not rows:
        return html

    md_rows = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[hd]>(.*?)</t[hd]>", row, re.DOTALL)
        md_rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join("---" for _ in cells) + " |")

    return "\n".join(md_rows)


def parse_nemotron_output(
    raw_output: str,
    image_width: int,
    image_height: int,
    text_format: str = "markdown",
    table_format: str = "latex",
) -> list[dict[str, Any]]:
    """Parse Nemotron Parse model output into structured regions.

    Convenience function that combines extract_classes_bboxes,
    transform_bbox_to_original, and postprocess_text.

    Args:
        raw_output: Raw output string from the model.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        text_format: Output format for text: "markdown" or "plain".
        table_format: Output format for tables: "latex", "HTML", or "markdown".

    Returns:
        List of region dicts with keys:
            - "class_name": str (e.g., "Title", "Text", "Table", "Picture")
            - "bbox": tuple (left, top, right, bottom) in pixel coordinates
            - "text": str (post-processed text content)
            - "needs_vl": bool (whether this region should be sent to the VL model)
    """
    classes, bboxes, texts = extract_classes_bboxes(raw_output)
    regions = []

    for cls, bbox, text in zip(classes, bboxes, texts, strict=True):
        pixel_bbox = transform_bbox_to_original(bbox, image_width, image_height)
        processed_text = postprocess_text(
            text, cls=cls, text_format=text_format, table_format=table_format
        )
        regions.append({
            "class_name": cls,
            "bbox": pixel_bbox,
            "text": processed_text,
            "needs_vl": cls in VL_CONTENT_TYPES,
        })

    return regions


def crop_image_from_bbox_tuple(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
) -> Image.Image:
    """Crop an image region using a (left, top, right, bottom) pixel bbox tuple.

    Clamps coordinates to image bounds.

    Args:
        image: PIL Image to crop from.
        bbox: (left, top, right, bottom) in pixel coordinates.

    Returns:
        Cropped PIL Image.
    """
    width, height = image.size
    left = max(0, min(int(bbox[0]), width))
    top = max(0, min(int(bbox[1]), height))
    right = max(0, min(int(bbox[2]), width))
    bottom = max(0, min(int(bbox[3]), height))

    if right <= left or bottom <= top:
        return Image.new("RGB", (1, 1))

    return image.crop((left, top, right, bottom))
