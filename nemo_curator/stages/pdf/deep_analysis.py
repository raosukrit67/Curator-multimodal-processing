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

"""Stage for deep content analysis using vision-language models."""

import json
import time

from huggingface_hub import snapshot_download
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass


class DeepAnalysisStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Perform deep content analysis using vision-language model via vLLM.

    This stage analyzes extracted content (tables, images, text) using a
    vision-language model to generate detailed descriptions, classifications,
    and insights.

    Args:
        model_identifier: HuggingFace model ID (default: "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
        classified_regions_field: Column containing classified regions (default: "classified_regions")
        output_field: Column for storing analysis results (default: "analysis_results")
        max_tokens: Maximum tokens for generation (default: 1024)
        temperature: Sampling temperature (default: 0.2)
        top_p: Top-p sampling parameter (default: 0.7)
        cache_dir: Directory for caching model weights
        hf_token: HuggingFace token for private models
        vllm_init_kwargs: Additional kwargs for vLLM initialization
        analyze_types: Types of content to analyze (default: ["table", "image", "figure", "chart"])
        verbose: Enable verbose logging (default: False)
    """

    def __init__(
        self,
        model_identifier: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        classified_regions_field: str = "classified_regions",
        output_field: str = "analysis_results",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.7,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        vllm_init_kwargs: dict | None = None,
        analyze_types: list[str] | None = None,
        verbose: bool = False,
    ):
        self.model_identifier = model_identifier
        self.classified_regions_field = classified_regions_field
        self.output_field = output_field
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.vllm_init_kwargs = vllm_init_kwargs or {}
        self.analyze_types = analyze_types or ["table", "image", "figure", "chart"]
        self.verbose = verbose

        self.model = None
        self.sampling_params = None

        self.name = "deep_analysis"
        self.resources = Resources(cpus=4.0, gpus=1.0, gpu_mem_gb=12.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.classified_regions_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.classified_regions_field, self.output_field]

    def _initialize_vllm(self) -> None:
        """Initialize vLLM model for deep analysis."""
        if not VLLM_AVAILABLE:
            msg = "vllm is required for DeepAnalysisStage but is not installed. Install with: pip install vllm"
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

        self.model = LLM(model=self.model_identifier, **vllm_init_kwargs)

        self.sampling_params = SamplingParams(
            temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens, stop_token_ids=[]
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

    def _prepare_analysis_prompt(self, region: dict) -> dict:
        """Prepare prompt for deep analysis.

        Args:
            region: Region dictionary with cropped image and metadata

        Returns:
            Prompt dictionary with image and text
        """
        content_type = region.get("classified_type", "unknown")

        # Customize prompt based on content type
        if content_type == "table":
            prompt_text = (
                "Analyze this table image and provide:\n"
                "1. A detailed description of the table structure and content\n"
                "2. Key insights or patterns visible in the data\n"
                "3. The table's likely purpose or what it represents\n"
                "Be concise but thorough."
            )
        elif content_type in ["image", "figure"]:
            prompt_text = (
                "Analyze this figure/image and provide:\n"
                "1. A detailed description of what is shown\n"
                "2. Key visual elements and their relationships\n"
                "3. The figure's likely purpose or message\n"
                "Be concise but thorough."
            )
        elif content_type == "chart":
            prompt_text = (
                "Analyze this chart and provide:\n"
                "1. The chart type (bar, line, pie, etc.)\n"
                "2. Key trends or patterns visible in the data\n"
                "3. Main insights or conclusions from the chart\n"
                "Be concise but thorough."
            )
        else:
            prompt_text = (
                "Analyze this content and provide a detailed description of what it contains "
                "and its purpose or significance."
            )

        return {
            "prompt": f"<image>\n{prompt_text}",
            "multi_modal_data": {"image": region["cropped_image_base64"]},
        }

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        analysis_results_list = []
        metrics = {}
        total_inference_time = 0

        for classified_regions_json in df[self.classified_regions_field]:
            try:
                classified_regions = json.loads(classified_regions_json)

                if not classified_regions:
                    analysis_results_list.append(json.dumps([]))
                    continue

                analyses = []

                for region in classified_regions:
                    # Only analyze specified types
                    if region.get("classified_type") not in self.analyze_types:
                        continue

                    # Skip if no image available
                    if "cropped_image_base64" not in region:
                        continue

                    # Prepare prompt
                    prompt = self._prepare_analysis_prompt(region)

                    # Run inference
                    t0 = time.perf_counter()
                    outputs = self.model.generate([prompt], sampling_params=self.sampling_params, use_tqdm=False)
                    inference_time = time.perf_counter() - t0
                    total_inference_time += inference_time

                    # Store analysis result
                    analysis_text = outputs[0].outputs[0].text

                    analyses.append(
                        {
                            "page_number": region["page_number"],
                            "object_index": region["object_index"],
                            "type": region["classified_type"],
                            "bbox": region["bbox"],
                            "analysis": analysis_text,
                        }
                    )

                analysis_results_list.append(json.dumps(analyses))
                logger.info(f"Analyzed {len(analyses)} content items")

            except Exception as e:
                logger.error(f"Failed to perform deep analysis: {e}")
                analysis_results_list.append(json.dumps([]))

        df[self.output_field] = analysis_results_list

        metrics["deep_analysis_time"] = total_inference_time
        self._log_metrics(metrics)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
