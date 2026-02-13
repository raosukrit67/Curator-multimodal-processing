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

"""Stage for detecting document layout using vision-language models.

This module uses vLLM for inference with vision-language models (VLMs).
Note: vLLM uses the same LLM class for both language-only models and
vision-language models (VLMs) - this is the correct and intended usage.
"""

import json
import time

from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.pdf_utils import base64_to_image

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass


class LayoutDetectionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Detect document layout using vision-language model (VLM) via vLLM.

    This stage processes page images through a vision-language model to detect
    document objects (text blocks, tables, images, etc.) with bounding boxes.

    The model used (nvidia/nemotron-parse) is a Vision-Language Model (VLM) that
    processes both text and images. vLLM's LLM class handles both standard LLMs
    and VLMs with multimodal capabilities.

    Args:
        model_identifier: HuggingFace VLM model ID (default: "nvidia/nemotron-parse")
        page_images_field: Column containing page images (default: "page_images")
        output_field: Column for storing layout objects (default: "layout_objects")
        max_tokens: Maximum tokens for generation (default: 3500)
        temperature: Sampling temperature (default: 0.0)
        cache_dir: Directory for caching model weights
        hf_token: HuggingFace token for private models
        vllm_init_kwargs: Additional kwargs for vLLM initialization
        verbose: Enable verbose logging (default: False)
    """

    def __init__(
        self,
        model_identifier: str = "nvidia/nemotron-parse",
        page_images_field: str = "page_images",
        output_field: str = "layout_objects",
        max_tokens: int = 3500,
        temperature: float = 0.0,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        vllm_init_kwargs: dict | None = None,
        verbose: bool = False,
    ):
        self.model_identifier = model_identifier
        self.page_images_field = page_images_field
        self.output_field = output_field
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.vllm_init_kwargs = vllm_init_kwargs or {}
        self.verbose = verbose

        self.model = None
        self.sampling_params = None

        self.name = "layout_detection"
        self.resources = Resources(cpus=4.0, gpus=1.0, gpu_mem_gb=16.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.page_images_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.page_images_field, self.output_field]

    def _initialize_vllm(self) -> None:
        """Initialize vLLM model for layout detection."""
        if not VLLM_AVAILABLE:
            msg = "vllm is required for LayoutDetectionStage but is not installed. Install with: pip install vllm"
            raise ImportError(msg)

        vllm_init_kwargs = self.vllm_init_kwargs.copy()

        # Set defaults
        if "enforce_eager" not in vllm_init_kwargs:
            vllm_init_kwargs["enforce_eager"] = False
        if self.cache_dir is not None and "download_dir" not in vllm_init_kwargs:
            vllm_init_kwargs["download_dir"] = self.cache_dir

        # Reduce verbosity when not in verbose mode
        if not self.verbose and "disable_log_stats" not in vllm_init_kwargs:
            vllm_init_kwargs["disable_log_stats"] = True

        # vLLM uses the LLM class for both standard LLMs and Vision-Language Models (VLMs)
        # This is the correct usage for multimodal models like nvidia/nemotron-parse
        self.model = LLM(model=self.model_identifier, **vllm_init_kwargs)

        self.sampling_params = SamplingParams(
            temperature=self.temperature, max_tokens=self.max_tokens, stop_token_ids=[]
        )

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        """Download model weights on node."""
        if not self.verbose:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()

        snapshot_download(self.model_identifier, cache_dir=self.cache_dir, token=self.hf_token, local_files_only=False)
        self._initialize_vllm()

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize vLLM model if not already initialized."""
        if self.model is None:
            self._initialize_vllm()

    def _prepare_prompt(self, image_base64: str) -> dict:
        """Prepare prompt for layout detection.

        Args:
            image_base64: Base64-encoded image

        Returns:
            Prompt dictionary with image and text
        """
        prompt_text = (
            "Extract all document objects (text blocks, tables, images, figures, charts) "
            "from this page with their bounding boxes. Return results as a JSON array where "
            "each object has: type (text|table|image|figure|chart), bbox (normalized coordinates "
            "as {x, y, width, height} in range [0, 1]), and any detected text or LaTeX content. "
            "Example format: [{\"type\": \"text\", \"bbox\": {\"x\": 0.1, \"y\": 0.2, \"width\": 0.8, "
            "\"height\": 0.1}, \"content\": \"text here\"}]"
        )

        return {
            "prompt": f"<image>\n{prompt_text}",
            "multi_modal_data": {"image": image_base64},
        }

    def _parse_layout_output(self, output_text: str) -> list[dict]:
        """Parse model output to extract layout objects.

        Args:
            output_text: Raw model output

        Returns:
            List of layout objects with bounding boxes
        """
        try:
            # Try to extract JSON from output
            # Model might wrap JSON in markdown code blocks or add explanation
            text = output_text.strip()

            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            # Parse JSON
            layout_objects = json.loads(text)

            if not isinstance(layout_objects, list):
                logger.warning(f"Expected list of objects, got {type(layout_objects)}")
                return []

            return layout_objects

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse layout detection output as JSON: {e}")
            logger.debug(f"Raw output: {output_text[:500]}")
            return []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        layout_objects_list = []
        metrics = {}
        total_inference_time = 0

        for page_images_json in df[self.page_images_field]:
            try:
                page_images = json.loads(page_images_json)

                if not page_images:
                    layout_objects_list.append(json.dumps([]))
                    continue

                page_layouts = []

                for page_data in page_images:
                    # Prepare prompt with image
                    image_base64 = page_data["image_base64"]
                    prompt = self._prepare_prompt(image_base64)

                    # Run inference
                    t0 = time.perf_counter()
                    outputs = self.model.generate([prompt], sampling_params=self.sampling_params, use_tqdm=False)
                    inference_time = time.perf_counter() - t0
                    total_inference_time += inference_time

                    # Parse output
                    output_text = outputs[0].outputs[0].text
                    layout_objects = self._parse_layout_output(output_text)

                    page_layouts.append(
                        {
                            "page_number": page_data["page_number"],
                            "width": page_data["width"],
                            "height": page_data["height"],
                            "objects": layout_objects,
                        }
                    )

                layout_objects_list.append(json.dumps(page_layouts))

            except Exception as e:
                logger.error(f"Failed to detect layout: {e}")
                layout_objects_list.append(json.dumps([]))

        df[self.output_field] = layout_objects_list

        metrics["layout_detection_time"] = total_inference_time
        self._log_metrics(metrics)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
