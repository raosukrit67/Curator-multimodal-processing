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
Visualize layout detection results with color-coded bounding boxes.

Renders each PDF page with bounding boxes and semantic class labels
overlaid on the original page image, matching the style of the
Nemotron Parse demo (https://build.nvidia.com/nvidia/nemotron-parse).

Usage:
    python visualize_layout.py \
        --pdf input.pdf \
        --predictions data/extracted/extracted_data.jsonl \
        --output visualizations/ \
        --format both

Output:
    For each page: page_N_layout.png (annotated image)
    Optionally: combined_layout.pdf (all pages in one PDF)
    Always: legend.png (color legend for all classes)
"""

import argparse
import json
import os
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from nemo_curator.utils.pdf_utils import (
    PARSE_CLASS_COLORS,
    pdf_to_images,
)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def hex_to_rgba(hex_color: str, alpha: int = 40) -> tuple[int, int, int, int]:
    """Convert hex color string to RGBA tuple with transparency."""
    r, g, b = hex_to_rgb(hex_color)
    return (r, g, b, alpha)


def get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font, falling back to default if system fonts unavailable."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


def draw_region_overlay(
    draw: ImageDraw.ImageDraw,
    overlay: Image.Image,
    bbox: tuple[float, float, float, float],
    class_name: str,
    region_index: int | None = None,
    border_width: int = 3,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
) -> None:
    """Draw a single region overlay with bbox, fill, and label.

    Args:
        draw: ImageDraw object for the main image (borders and labels).
        overlay: RGBA image for semi-transparent fills.
        bbox: (left, top, right, bottom) in pixel coordinates.
        class_name: Semantic class name (e.g., "Title", "Text").
        region_index: Optional region number for labeling.
        border_width: Width of the bbox border.
        font: Font for label text.
    """
    left, top, right, bottom = [int(v) for v in bbox]
    color_hex = PARSE_CLASS_COLORS.get(class_name, "#888888")
    color_rgb = hex_to_rgb(color_hex)
    color_rgba = hex_to_rgba(color_hex, alpha=35)

    # Draw semi-transparent fill on the overlay
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([left, top, right, bottom], fill=color_rgba)

    # Draw border on the main image
    draw.rectangle([left, top, right, bottom], outline=color_rgb, width=border_width)

    # Draw label tag
    label = class_name
    if region_index is not None:
        label = f"[{region_index}] {class_name}"

    if font is None:
        font = get_font(12)

    # Measure text
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    padding = 3

    # Label background (solid color)
    label_x = left
    label_y = max(0, top - text_h - padding * 2)
    draw.rectangle(
        [label_x, label_y, label_x + text_w + padding * 2, label_y + text_h + padding * 2],
        fill=color_rgb,
    )
    draw.text(
        (label_x + padding, label_y + padding),
        label,
        fill="white",
        font=font,
    )


def create_legend(output_path: str, width: int = 400) -> None:
    """Create a color legend image showing all content classes."""
    font = get_font(16)
    row_height = 30
    padding = 10
    swatch_size = 20

    classes = list(PARSE_CLASS_COLORS.items())
    height = padding * 2 + len(classes) * row_height + 30

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Title
    title_font = get_font(18)
    draw.text((padding, padding), "Content Classes", fill="black", font=title_font)

    y = padding + 35
    for class_name, color_hex in classes:
        color_rgb = hex_to_rgb(color_hex)

        # Color swatch
        draw.rectangle(
            [padding, y, padding + swatch_size, y + swatch_size],
            fill=color_rgb,
            outline="black",
        )

        # Label
        draw.text(
            (padding + swatch_size + 10, y + 2),
            class_name,
            fill="black",
            font=font,
        )
        y += row_height

    img.save(output_path)
    logger.info(f"Legend saved to {output_path}")


def visualize_document(
    pdf_path: str,
    predictions: dict,
    output_dir: str,
    dpi: int = 300,
    output_format: str = "both",
) -> list[str]:
    """Visualize layout predictions for a single PDF.

    Args:
        pdf_path: Path to the original PDF.
        predictions: Prediction dict with "pages" key.
        output_dir: Directory for output files.
        dpi: DPI for rendering (should match extraction pipeline).
        output_format: "images", "pdf", or "both".

    Returns:
        List of output file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pdf_name = Path(pdf_path).stem
    font = get_font(14)

    # Render PDF pages
    try:
        page_images = pdf_to_images(pdf_path, dpi=dpi)
    except Exception as e:
        logger.error(f"Failed to render PDF {pdf_path}: {e}")
        return []

    pages = predictions.get("pages", [])
    output_files = []
    annotated_images = []

    for page_data in pages:
        page_num = page_data["page_number"]
        if page_num >= len(page_images):
            logger.warning(f"Page {page_num} not found in PDF (has {len(page_images)} pages)")
            continue

        base_image = page_images[page_num].copy().convert("RGBA")
        overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(base_image)

        region_idx = 0
        all_regions = []

        # Collect all regions for this page
        for block in page_data.get("text_blocks", []):
            all_regions.append((block["class_name"], block["bbox"]))
        for table in page_data.get("tables", []):
            all_regions.append(("Table", table["bbox"]))
        for figure in page_data.get("figures", []):
            cls = figure.get("class_name", "Picture")
            all_regions.append((cls, figure["bbox"]))

        # Draw each region
        for class_name, bbox in all_regions:
            draw_region_overlay(
                draw=draw,
                overlay=overlay,
                bbox=tuple(bbox),
                class_name=class_name,
                region_index=region_idx,
                font=font,
            )
            region_idx += 1

        # Composite overlay onto base
        result = Image.alpha_composite(base_image, overlay).convert("RGB")
        annotated_images.append(result)

        if output_format in ("images", "both"):
            img_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num}_layout.png")
            result.save(img_path, quality=95)
            output_files.append(img_path)
            logger.info(f"Saved: {img_path} ({region_idx} regions)")

    # Create combined PDF
    if output_format in ("pdf", "both") and annotated_images:
        pdf_output = os.path.join(output_dir, f"{pdf_name}_layout.pdf")
        annotated_images[0].save(
            pdf_output,
            save_all=True,
            append_images=annotated_images[1:],
            resolution=dpi,
        )
        output_files.append(pdf_output)
        logger.info(f"Saved combined PDF: {pdf_output}")

    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize layout detection results with color-coded bounding boxes",
    )
    parser.add_argument("--pdf", required=True, help="Input PDF file path")
    parser.add_argument(
        "--predictions", required=True,
        help="Path to extraction output JSONL",
    )
    parser.add_argument(
        "--output", default="visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--format", choices=["images", "pdf", "both"], default="both",
        help="Output format",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering")

    args = parser.parse_args()

    # Find predictions for this PDF
    pdf_path = os.path.abspath(args.pdf)
    predictions = None

    with open(args.predictions) as f:
        for line in f:
            entry = json.loads(line.strip())
            if os.path.abspath(entry.get("pdf_path", "")) == pdf_path:
                predictions = entry
                break

    if predictions is None:
        logger.error(f"No predictions found for {pdf_path} in {args.predictions}")
        logger.info("Available PDFs in predictions:")
        with open(args.predictions) as f:
            for line in f:
                entry = json.loads(line.strip())
                logger.info(f"  - {entry.get('pdf_path', 'unknown')}")
        return

    # Create legend
    create_legend(os.path.join(args.output, "legend.png"))

    # Visualize
    output_files = visualize_document(
        pdf_path=pdf_path,
        predictions=predictions,
        output_dir=args.output,
        dpi=args.dpi,
        output_format=args.format,
    )

    logger.info(f"Created {len(output_files)} output files in {args.output}")


if __name__ == "__main__":
    main()
