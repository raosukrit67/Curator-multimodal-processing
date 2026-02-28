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

"""Stage for layout detection and text extraction using Nemotron Parse.

Nemotron Parse is a specialized encoder-decoder VLM (~885M params) that
simultaneously performs OCR, layout detection, and semantic classification.
Given a page image, it outputs interleaved tokens containing:
- Extracted text as Markdown (with LaTeX for math/tables)
- Bounding boxes for every content region
- Semantic class labels (Title, Text, Table, Picture, Formula, etc.)
"""

import json
import time

from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.vllm_model import VLLMModel
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.pdf_utils import (
    base64_to_image,
    parse_nemotron_output,
)
from nemo_curator.utils.prompts import NEMOTRON_PARSE_PROMPT


class LayoutDetectionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Detect document layout and extract text using Nemotron Parse via vLLM.

    This stage processes page images through Nemotron Parse to detect document
    structure (bounding boxes + semantic classes) and extract text content in
    a single pass.

    The output for each page is a list of region dicts, each containing:
        - class_name: semantic class (Title, Text, Table, Picture, etc.)
        - bbox: (left, top, right, bottom) in original image pixel coordinates
        - text: extracted text content (markdown format)
        - needs_vl: whether this region needs VL model for visual understanding

    Args:
        model_identifier: HuggingFace model ID for Nemotron Parse.
        page_images_field: Column containing page images JSON.
        output_field: Column for storing parsed layout regions.
        prompt: Parse control token prompt.
        max_tokens: Maximum tokens for generation.
        temperature: Sampling temperature (0.0 = deterministic).
        text_format: Output text format: "markdown" or "plain".
        table_format: Output table format: "latex", "HTML", or "markdown".
        cache_dir: Directory for caching model weights.
        hf_token: HuggingFace token for private models.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        model_identifier: str = "nvidia/NVIDIA-Nemotron-Parse-v1.1",
        page_images_field: str = "page_images",
        output_field: str = "layout_regions",
        prompt: str = NEMOTRON_PARSE_PROMPT,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        text_format: str = "markdown",
        table_format: str = "latex",
        cache_dir: str | None = None,
        hf_token: str | None = None,
        verbose: bool = False,
    ):
        self.model_identifier = model_identifier
        self.page_images_field = page_images_field
        self.output_field = output_field
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.text_format = text_format
        self.table_format = table_format
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.verbose = verbose

        self._model: VLLMModel | None = None

        self.name = "layout_detection"
        self.resources = Resources(cpus=1.0, gpus=1.0, gpu_mem_gb=16.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.page_images_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.page_images_field, self.output_field]

    def setup_on_node(
        self,
        node_info: NodeInfo | None = None,
        worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Download model weights on node."""
        if not self.verbose:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()

        snapshot_download(
            self.model_identifier,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            local_files_only=False,
        )
        self._initialize_model()

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the model if not already initialized."""
        if self._model is None:
            self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the VLLMModel wrapper."""
        self._model = VLLMModel(
            model=self.model_identifier,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            cache_dir=self.cache_dir,
            extra_llm_kwargs={
                "disable_log_stats": not self.verbose,
            },
        )
        self._model.setup()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        layout_regions_list = []
        total_inference_time = 0.0

        for page_images_json in df[self.page_images_field]:
            try:
                page_images = json.loads(page_images_json)
                if not page_images:
                    layout_regions_list.append(json.dumps([]))
                    continue

                all_page_regions = []
                for page_data in page_images:
                    image = base64_to_image(page_data["image_base64"])
                    width = page_data["width"]
                    height = page_data["height"]

                    # Prepare prompt for Nemotron Parse
                    prompt_dict = {
                        "prompt": self.prompt,
                        "multi_modal_data": {"image": image},
                    }

                    # Run inference
                    t0 = time.perf_counter()
                    outputs = self._model.generate([prompt_dict])
                    inference_time = time.perf_counter() - t0
                    total_inference_time += inference_time

                    raw_output = outputs[0] if outputs else ""

                    # Parse output using official postprocessing
                    regions = parse_nemotron_output(
                        raw_output,
                        image_width=width,
                        image_height=height,
                        text_format=self.text_format,
                        table_format=self.table_format,
                    )

                    all_page_regions.append({
                        "page_number": page_data["page_number"],
                        "width": width,
                        "height": height,
                        "regions": regions,
                        "raw_output": raw_output if self.verbose else None,
                    })

                layout_regions_list.append(json.dumps(all_page_regions))

                if self.verbose:
                    total_regions = sum(len(p["regions"]) for p in all_page_regions)
                    logger.info(
                        f"Detected {total_regions} regions across "
                        f"{len(all_page_regions)} pages "
                        f"({inference_time:.2f}s)"
                    )

            except Exception as e:
                logger.error(f"Layout detection failed: {e}")
                layout_regions_list.append(json.dumps([]))

        df[self.output_field] = layout_regions_list

        if self.verbose:
            logger.info(f"Total layout detection time: {total_inference_time:.2f}s")

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
