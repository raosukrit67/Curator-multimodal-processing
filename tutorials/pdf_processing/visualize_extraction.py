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
Visualize text extraction quality.

Three visualization modes:
1. side-by-side: Original page (with numbered boxes) next to extracted text
2. heatmap: Color-coded extraction status overlay (green=ok, red=empty, yellow=VL)

Usage:
    python visualize_extraction.py \
        --pdf input.pdf \
        --predictions data/extracted/extracted_data.jsonl \
        --output visualizations/ \
        --mode side-by-side
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from nemo_curator.utils.pdf_utils import (
    PARSE_CLASS_COLORS,
    pdf_to_images,
)


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


def _collect_all_regions(page_data: dict) -> list[dict]:
    """Collect all regions from a page in order, with a unified schema."""
    regions = []
    for block in page_data.get("text_blocks", []):
        regions.append({
            "class_name": block.get("class_name", "Text"),
            "bbox": block["bbox"],
            "text": block.get("text", ""),
            "source": "parse",
        })
    for table in page_data.get("tables", []):
        desc = table.get("description", "")
        latex = table.get("latex", "")
        display = desc if desc else (latex[:200] + "..." if len(latex) > 200 else latex)
        regions.append({
            "class_name": "Table",
            "bbox": table["bbox"],
            "text": display,
            "source": "vl" if desc else "parse",
        })
    for figure in page_data.get("figures", []):
        regions.append({
            "class_name": figure.get("class_name", "Picture"),
            "bbox": figure["bbox"],
            "text": figure.get("description", ""),
            "source": "vl",
        })
    return regions


def render_side_by_side(
    page_image: Image.Image,
    page_data: dict,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    """Render side-by-side: left=annotated page, right=extracted text."""
    # Left panel: annotated page
    left = page_image.copy().convert("RGB")
    draw_left = ImageDraw.Draw(left)
    regions = _collect_all_regions(page_data)

    # Right panel: text content
    text_panel_width = left.width
    right = Image.new("RGB", (text_panel_width, left.height), "white")
    draw_right = ImageDraw.Draw(right)

    small_font = get_font(11)
    header_font = get_font(13)
    y_cursor = 10
    line_height = 16

    for idx, region in enumerate(regions):
        bbox = region["bbox"]
        cls = region["class_name"]
        text = region["text"]
        color_hex = PARSE_CLASS_COLORS.get(cls, "#888888")

        # Draw numbered box on left panel
        left_px, top_px, right_px, bottom_px = [int(v) for v in bbox]
        draw_left.rectangle(
            [left_px, top_px, right_px, bottom_px],
            outline=color_hex, width=2,
        )
        # Number label
        label = str(idx)
        draw_left.rectangle(
            [left_px, top_px, left_px + 20, top_px + 16],
            fill=color_hex,
        )
        draw_left.text((left_px + 3, top_px + 1), label, fill="white", font=small_font)

        # Draw text on right panel
        if y_cursor > right.height - 50:
            break

        # Header: [N] ClassName
        header = f"[{idx}] {cls}"
        source_tag = f" ({region['source']})" if region["source"] == "vl" else ""
        draw_right.text(
            (10, y_cursor), header + source_tag, fill=color_hex, font=header_font,
        )
        y_cursor += line_height + 2

        # Content (wrapped)
        if text:
            wrapped = textwrap.fill(text[:500], width=60)
            for line in wrapped.split("\n"):
                if y_cursor > right.height - 20:
                    break
                draw_right.text((20, y_cursor), line, fill="#333333", font=small_font)
                y_cursor += line_height
        else:
            draw_right.text((20, y_cursor), "(no content)", fill="#999999", font=small_font)
            y_cursor += line_height

        y_cursor += 8  # Spacing between regions

    # Combine panels
    combined = Image.new("RGB", (left.width + right.width + 4, left.height), "#CCCCCC")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + 4, 0))
    return combined


def render_heatmap(
    page_image: Image.Image,
    page_data: dict,
) -> Image.Image:
    """Render heatmap: green=extracted, red=empty, yellow=VL-processed."""
    base = page_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(base)

    regions = _collect_all_regions(page_data)
    font = get_font(11)

    for region in regions:
        bbox = [int(v) for v in region["bbox"]]
        has_text = bool(region["text"].strip())
        source = region["source"]

        if source == "vl":
            color = (255, 200, 0, 50)  # Yellow for VL-processed
            border_color = "#E6C800"
        elif has_text:
            color = (0, 180, 0, 40)  # Green for extracted OK
            border_color = "#00B400"
        else:
            color = (220, 0, 0, 50)  # Red for empty/failed
            border_color = "#DC0000"

        draw_overlay.rectangle(bbox, fill=color)
        draw_base.rectangle(bbox, outline=border_color, width=2)

        # Status label
        if source == "vl":
            label = "VL"
        elif has_text:
            label = "OK"
        else:
            label = "EMPTY"
        draw_base.text((bbox[0] + 2, bbox[1] + 2), label, fill=border_color, font=font)

    result = Image.alpha_composite(base, overlay).convert("RGB")
    return result


def visualize_extraction(
    pdf_path: str,
    predictions: dict,
    output_dir: str,
    mode: str = "side-by-side",
    dpi: int = 300,
) -> list[str]:
    """Visualize extraction quality for a single PDF."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pdf_name = Path(pdf_path).stem
    font = get_font(14)

    try:
        page_images = pdf_to_images(pdf_path, dpi=dpi)
    except Exception as e:
        logger.error(f"Failed to render PDF: {e}")
        return []

    pages = predictions.get("pages", [])
    output_files = []

    for page_data in pages:
        page_num = page_data["page_number"]
        if page_num >= len(page_images):
            continue

        page_image = page_images[page_num]

        if mode == "side-by-side":
            result = render_side_by_side(page_image, page_data, font)
        elif mode == "heatmap":
            result = render_heatmap(page_image, page_data)
        else:
            logger.error(f"Unknown mode: {mode}")
            continue

        img_path = os.path.join(
            output_dir, f"{pdf_name}_page_{page_num}_{mode}.png"
        )
        result.save(img_path, quality=95)
        output_files.append(img_path)
        logger.info(f"Saved: {img_path}")

    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize text extraction quality",
    )
    parser.add_argument("--pdf", required=True, help="Input PDF file path")
    parser.add_argument(
        "--predictions", required=True,
        help="Path to extraction output JSONL",
    )
    parser.add_argument(
        "--output", default="visualizations",
        help="Output directory",
    )
    parser.add_argument(
        "--mode",
        choices=["side-by-side", "heatmap"],
        default="side-by-side",
        help="Visualization mode",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendering")

    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    predictions = None

    with open(args.predictions) as f:
        for line in f:
            entry = json.loads(line.strip())
            if os.path.abspath(entry.get("pdf_path", "")) == pdf_path:
                predictions = entry
                break

    if predictions is None:
        logger.error(f"No predictions found for {pdf_path}")
        return

    output_files = visualize_extraction(
        pdf_path=pdf_path,
        predictions=predictions,
        output_dir=args.output,
        mode=args.mode,
        dpi=args.dpi,
    )

    logger.info(f"Created {len(output_files)} visualization files")


if __name__ == "__main__":
    main()
