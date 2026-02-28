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

from typing import Any

from loguru import logger

from nemo_curator.models.base import ModelInterface
from nemo_curator.utils.gpu_utils import get_gpu_count, get_max_model_len_from_config

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class LLM:
        pass

    class SamplingParams:
        pass


class VLLMModel(ModelInterface):
    """Unified vLLM model wrapper for text and vision-language generation.

    Supports three generation modes:
    - ``generate()``: Text-only or multimodal prompts via vLLM's generate API.
      Works with plain strings, chat-template message lists, or dicts with
      ``multi_modal_data`` (e.g., Nemotron Parse).
    - ``chat()``: OpenAI-style chat completions via vLLM's chat API.
      Supports ``image_url`` content blocks (e.g., Nemotron Nano VL).
    - ``get_tokenizer()``: Access the underlying tokenizer.

    Args:
        model: Model identifier (e.g., "microsoft/phi-4" or
            "nvidia/NVIDIA-Nemotron-Parse-v1.1").
        max_model_len: Maximum model context length. Auto-detected from
            HuggingFace config if not specified.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            Auto-detects available GPUs if not specified.
        max_num_batched_tokens: Maximum tokens per batch. Defaults to 4096.
        temperature: Sampling temperature. Defaults to 0.7.
        top_p: Top-p sampling parameter. Defaults to 0.8.
        top_k: Top-k sampling parameter. Defaults to 20.
        min_p: Min-p sampling parameter (for Qwen3). Defaults to 0.0.
        max_tokens: Maximum tokens to generate. Defaults to None
            (uses max_model_len).
        cache_dir: Cache directory for model weights.
        allowed_local_media_path: Path prefix for local media files.
            Required for chat API with local image URLs.
        extra_llm_kwargs: Additional kwargs passed to vLLM LLM constructor
            (e.g., ``mm_processor_kwargs``, ``limit_mm_per_prompt``,
            ``disable_log_stats``).
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        max_model_len: int | None = None,
        tensor_parallel_size: int | None = None,
        max_num_batched_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int | None = None,
        cache_dir: str | None = None,
        allowed_local_media_path: str | None = None,
        extra_llm_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.allowed_local_media_path = allowed_local_media_path
        self.extra_llm_kwargs = extra_llm_kwargs or {}
        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
        self._final_max_model_len: int | None = None
        self._is_qwen3: bool = False

    def model_id_names(self) -> list[str]:
        """Return the model identifier."""
        return [self.model]

    def setup(self) -> None:
        """Set up the vLLM model and sampling parameters."""
        if not VLLM_AVAILABLE:
            msg = (
                "vLLM is required for VLLMModel. "
                "Please install it: pip install vllm"
            )
            raise ImportError(msg)

        # Resolve max_model_len
        if self.max_model_len is not None:
            final_max_model_len = self.max_model_len
        else:
            final_max_model_len = get_max_model_len_from_config(self.model)

        # Resolve tensor_parallel_size
        final_tp_size = (
            self.tensor_parallel_size
            if self.tensor_parallel_size is not None
            else get_gpu_count()
        )

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "enforce_eager": False,
            "trust_remote_code": True,
            "tensor_parallel_size": final_tp_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }

        if final_max_model_len is not None:
            llm_kwargs["max_model_len"] = final_max_model_len

        if self.cache_dir is not None:
            llm_kwargs["download_dir"] = self.cache_dir

        if self.allowed_local_media_path is not None:
            llm_kwargs["allowed_local_media_path"] = self.allowed_local_media_path

        llm_kwargs.update(self.extra_llm_kwargs)

        logger.info(
            f"Initializing vLLM with: model={self.model}, "
            f"max_model_len={final_max_model_len}, "
            f"tensor_parallel_size={final_tp_size}, "
            f"max_num_batched_tokens={self.max_num_batched_tokens}"
        )

        self._llm = LLM(**llm_kwargs)
        self._final_max_model_len = final_max_model_len

        max_gen_tokens = (
            self.max_tokens
            if self.max_tokens is not None
            else final_max_model_len
        )
        is_qwen3 = "Qwen3" in self.model or "qwen3" in self.model.lower()

        sampling_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": max_gen_tokens,
        }

        if is_qwen3:
            sampling_kwargs.update(
                {
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "min_p": self.min_p,
                }
            )
        else:
            sampling_kwargs["top_p"] = self.top_p

        self._sampling_params = SamplingParams(**sampling_kwargs)
        self._is_qwen3 = is_qwen3

    def generate(
        self,
        prompts: list[str] | list[dict[str, Any]] | list[list[dict[str, str]]],
    ) -> list[str]:
        """Generate text from prompts.

        Supports multiple prompt formats:
        - List of strings (text-only generation)
        - List of dicts with "prompt" and "multi_modal_data" keys
          (multimodal generation, e.g., Nemotron Parse)
        - List of message-dict lists (chat template formatting)

        Args:
            prompts: Prompts in any of the supported formats.

        Returns:
            List of generated text strings.

        Raises:
            RuntimeError: If the model is not set up or generation fails.
        """
        if self._llm is None or self._sampling_params is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        try:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )
            return [
                out.outputs[0].text if out.outputs else ""
                for out in outputs
            ]
        except Exception as e:
            msg = f"Error generating text: {e}"
            raise RuntimeError(msg) from e

    def chat(
        self,
        messages_list: list[list[dict[str, Any]]],
        sampling_params: Any | None = None,
    ) -> list[str]:
        """Generate text using the chat API.

        Uses vLLM's OpenAI-compatible chat completions endpoint.
        Supports ``image_url`` content blocks for vision-language models.

        Args:
            messages_list: List of conversation message lists. Each message
                list contains dicts with "role" and "content" keys.
            sampling_params: Optional override for sampling parameters.

        Returns:
            List of generated text strings.
        """
        if self._llm is None or self._sampling_params is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        params = sampling_params if sampling_params is not None else self._sampling_params

        try:
            results = []
            for messages in messages_list:
                outputs = self._llm.chat(
                    messages=messages,
                    sampling_params=params,
                )
                if outputs and outputs[0].outputs:
                    results.append(outputs[0].outputs[0].text)
                else:
                    results.append("")
        except Exception as e:
            msg = f"Error in chat: {e}"
            raise RuntimeError(msg) from e
        return results

    def get_tokenizer(self) -> Any:  # noqa: ANN401
        """Get the tokenizer from the LLM instance."""
        if self._llm is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._llm.get_tokenizer()


# Backward-compatible alias
VLLMVisionModel = VLLMModel
