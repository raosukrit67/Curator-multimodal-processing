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

"""Utility functions for PDF processing operations."""

import base64
import io
from typing import Any

import numpy as np
from PIL import Image


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 300) -> Image.Image:
    """Convert a PDF page to a PIL Image.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-indexed)
        dpi: Resolution for rendering (default: 300)

    Returns:
        PIL Image of the page
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        msg = "pdf2image is required for PDF to image conversion. Install with: pip install pdf2image"
        raise ImportError(msg) from exc

    # Convert single page
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1)

    if not images:
        msg = f"Failed to convert page {page_num} from {pdf_path}"
        raise ValueError(msg)

    return images[0]


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    """Convert all pages of a PDF to PIL Images.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (default: 300)

    Returns:
        List of PIL Images, one per page
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        msg = "pdf2image is required for PDF to image conversion. Install with: pip install pdf2image"
        raise ImportError(msg) from exc

    return convert_from_path(pdf_path, dpi=dpi)


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
