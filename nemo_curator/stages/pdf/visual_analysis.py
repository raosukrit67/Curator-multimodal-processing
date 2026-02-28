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

"""Stage for visual content analysis using Nemotron Nano VL.

This stage processes visual content (pictures, figures, charts, infographics)
detected by the layout detection stage, generating natural language descriptions
using the Nemotron Nano VL vision-language model.

Nemotron Nano VL is a 12B parameter VLM that can understand and describe
visual content. It receives cropped image regions and generates detailed
descriptions of charts, diagrams, figures, etc.
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
from nemo_curator.utils.prompts import VL_DEFAULT_ANALYSIS_PROMPT, VL_PROMPTS_BY_TYPE


class VisualAnalysisStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Analyze visual content using Nemotron Nano VL via vLLM.

    Processes visual regions (pictures, figures, charts) that were routed by
    ContentRoutingStage, generating natural language descriptions using the
    VL model's chat API.

    Args:
        model_identifier: HuggingFace model ID for Nemotron Nano VL.
        prompts_by_type: Dict mapping content class names to prompt strings.
            Defaults to VL_PROMPTS_BY_TYPE from prompts.py.
        default_prompt: Fallback prompt for unrecognized content types.
            Defaults to VL_DEFAULT_ANALYSIS_PROMPT from prompts.py.
        routed_content_field: Column with routed content from ContentRoutingStage.
        output_field: Column for storing analysis results.
        max_tokens: Maximum tokens for generation.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        cache_dir: Directory for caching model weights.
        hf_token: HuggingFace token for private models.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        model_identifier: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        prompts_by_type: dict[str, str] | None = None,
        default_prompt: str = VL_DEFAULT_ANALYSIS_PROMPT,
        routed_content_field: str = "routed_content",
        output_field: str = "analysis_results",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_model_len: int | None = None,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        verbose: bool = False,
    ):
        self.model_identifier = model_identifier
        self.prompts_by_type = prompts_by_type if prompts_by_type is not None else VL_PROMPTS_BY_TYPE
        self.default_prompt = default_prompt
        self.routed_content_field = routed_content_field
        self.output_field = output_field
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_model_len = max_model_len
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.verbose = verbose

        self._model: VLLMModel | None = None

        self.name = "visual_analysis"
        self.resources = Resources(cpus=1.0, gpus=1.0, gpu_mem_gb=24.0)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.routed_content_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.routed_content_field, self.output_field]

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
        """Initialize the VLLMModel wrapper for Nemotron Nano VL."""
        self._model = VLLMModel(
            model=self.model_identifier,
            max_model_len=self.max_model_len,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            cache_dir=self.cache_dir,
            allowed_local_media_path="/",
            extra_llm_kwargs={
                "disable_log_stats": not self.verbose,
            },
        )
        self._model.setup()

    def _build_chat_messages(
        self, region: dict
    ) -> list[dict]:
        """Build chat messages for VL model from a visual region.

        Uses the chat API format with image_url content blocks,
        as recommended for Nemotron Nano VL.
        """
        cls = region.get("class_name", "Picture")
        prompt_text = self.prompts_by_type.get(cls, self.default_prompt)
        image_b64 = region["cropped_image_base64"]

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            }
        ]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        analysis_results_list = []
        total_inference_time = 0.0

        for routed_json in df[self.routed_content_field]:
            try:
                routed_pages = json.loads(routed_json)
                if not routed_pages:
                    analysis_results_list.append(json.dumps([]))
                    continue

                page_analyses = []
                for page_data in routed_pages:
                    page_num = page_data["page_number"]
                    vl_regions = page_data.get("vl_regions", [])

                    analyses = []
                    for region in vl_regions:
                        if "cropped_image_base64" not in region:
                            continue

                        messages = self._build_chat_messages(region)

                        t0 = time.perf_counter()
                        results = self._model.chat([messages])
                        inference_time = time.perf_counter() - t0
                        total_inference_time += inference_time

                        description = results[0] if results else ""

                        analyses.append({
                            "page_number": page_num,
                            "class_name": region["class_name"],
                            "bbox": region["bbox"],
                            "description": description,
                        })

                    page_analyses.extend(analyses)

                analysis_results_list.append(json.dumps(page_analyses))

                if self.verbose and page_analyses:
                    logger.info(
                        f"Analyzed {len(page_analyses)} visual regions "
                        f"({total_inference_time:.2f}s)"
                    )

            except Exception as e:
                logger.error(f"Visual analysis failed: {e}")
                analysis_results_list.append(json.dumps([]))

        df[self.output_field] = analysis_results_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
