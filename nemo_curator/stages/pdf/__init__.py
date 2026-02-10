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

"""Stages for PDF document processing and multimodal content extraction."""

from nemo_curator.stages.pdf.bbox_extraction import BoundingBoxExtractionStage
from nemo_curator.stages.pdf.content_classification import ContentTypeClassificationStage
from nemo_curator.stages.pdf.deep_analysis import DeepAnalysisStage
from nemo_curator.stages.pdf.image_conversion import PDFToImageStage
from nemo_curator.stages.pdf.image_extraction import ImageExtractionStage
from nemo_curator.stages.pdf.layout_detection import LayoutDetectionStage
from nemo_curator.stages.pdf.table_extraction import TableExtractionStage
from nemo_curator.stages.pdf.text_extraction import TextExtractionStage

__all__ = [
    "PDFToImageStage",
    "LayoutDetectionStage",
    "BoundingBoxExtractionStage",
    "ContentTypeClassificationStage",
    "TableExtractionStage",
    "TextExtractionStage",
    "ImageExtractionStage",
    "DeepAnalysisStage",
]
