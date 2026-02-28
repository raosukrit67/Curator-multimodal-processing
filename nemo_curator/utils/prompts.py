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

"""Prompt templates for NeMo Curator pipeline stages."""

# =============================================================================
# Nemotron Parse prompts
# =============================================================================

# Default prompt for Nemotron Parse 1.1B: extract bboxes, classes, and markdown text
NEMOTRON_PARSE_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"

# Bbox + classes only (no text extraction) - faster for layout-only tasks
NEMOTRON_PARSE_LAYOUT_ONLY_PROMPT = "</s><s><predict_bbox><predict_classes><output_no_text>"

# =============================================================================
# Nemotron Nano VL prompts for visual content analysis
# =============================================================================

VL_PICTURE_ANALYSIS_PROMPT = (
    "Analyze this image and provide:\n"
    "1. A detailed description of what is shown\n"
    "2. Key visual elements and their relationships\n"
    "3. The image's likely purpose or context\n"
    "Be concise but thorough."
)

VL_TABLE_ANALYSIS_PROMPT = (
    "Analyze this table image and provide:\n"
    "1. A description of the table structure, headers, and content\n"
    "2. Key data points or patterns in the data\n"
    "3. What the table represents and its likely purpose\n"
    "Be concise but thorough."
)

VL_CHART_ANALYSIS_PROMPT = (
    "Analyze this chart and provide:\n"
    "1. The chart type (bar, line, pie, scatter, etc.)\n"
    "2. Axis labels and data ranges\n"
    "3. Key trends, patterns, or outliers\n"
    "4. Main conclusions from the data\n"
    "Be concise but thorough."
)

VL_FIGURE_ANALYSIS_PROMPT = (
    "Analyze this figure/diagram and provide:\n"
    "1. A detailed description of the diagram structure and components\n"
    "2. Key relationships or flows depicted\n"
    "3. The figure's likely purpose or message\n"
    "Be concise but thorough."
)

VL_INFOGRAPHIC_ANALYSIS_PROMPT = (
    "Analyze this infographic and provide:\n"
    "1. A description of the visual layout and key elements\n"
    "2. The main information or data being conveyed\n"
    "3. Key takeaways or conclusions\n"
    "Be concise but thorough."
)

# Map content types to VL prompts
VL_PROMPTS_BY_TYPE = {
    "Picture": VL_PICTURE_ANALYSIS_PROMPT,
    "Figure": VL_FIGURE_ANALYSIS_PROMPT,
    "Chart": VL_CHART_ANALYSIS_PROMPT,
    "Table": VL_TABLE_ANALYSIS_PROMPT,
    "Infographic": VL_INFOGRAPHIC_ANALYSIS_PROMPT,
}

# Default VL prompt for unrecognized content types
VL_DEFAULT_ANALYSIS_PROMPT = (
    "Analyze this content and provide a detailed description of "
    "what it contains and its purpose or significance."
)
