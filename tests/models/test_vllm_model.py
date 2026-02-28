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

"""Tests for VLLMModel wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.models.vllm_model import VLLMModel, VLLMVisionModel


class TestVLLMModel:
    def test_init_defaults(self):
        model = VLLMModel(model="test/model")
        assert model.model == "test/model"
        assert model.temperature == 0.7
        assert model.max_model_len is None
        assert model.allowed_local_media_path is None
        assert model.extra_llm_kwargs == {}
        assert model._llm is None

    def test_init_with_vision_kwargs(self):
        model = VLLMModel(
            model="nvidia/test-vlm",
            temperature=0.0,
            allowed_local_media_path="/data",
            extra_llm_kwargs={"limit_mm_per_prompt": {"image": 4}},
        )
        assert model.allowed_local_media_path == "/data"
        assert model.extra_llm_kwargs == {"limit_mm_per_prompt": {"image": 4}}
        assert model.temperature == 0.0

    def test_model_id_names(self):
        model = VLLMModel(model="test/model")
        assert model.model_id_names() == ["test/model"]

    def test_generate_without_setup_raises(self):
        model = VLLMModel(model="test/model")
        with pytest.raises(RuntimeError, match="not initialized"):
            model.generate(["hello"])

    def test_chat_without_setup_raises(self):
        model = VLLMModel(model="test/model")
        with pytest.raises(RuntimeError, match="not initialized"):
            model.chat([[{"role": "user", "content": "test"}]])

    def test_get_tokenizer_without_setup_raises(self):
        model = VLLMModel(model="test/model")
        with pytest.raises(RuntimeError, match="not initialized"):
            model.get_tokenizer()

    @patch("nemo_curator.models.vllm_model.VLLM_AVAILABLE", True)
    @patch("nemo_curator.models.vllm_model._get_gpu_count", return_value=1)
    @patch("nemo_curator.models.vllm_model._get_max_model_len_from_config", return_value=4096)
    @patch("nemo_curator.models.vllm_model.LLM")
    def test_generate_with_multimodal_data(self, mock_llm_cls, mock_config, mock_gpu):
        """Test generate() with multimodal prompt dicts."""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated description")]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = [mock_output]
        mock_llm_cls.return_value = mock_llm

        model = VLLMModel(model="nvidia/test-vlm", max_tokens=1024)
        model.setup()

        results = model.generate([
            {"prompt": "Describe this image.", "multi_modal_data": {"image": "base64data"}},
        ])

        assert len(results) == 1
        assert results[0] == "Generated description"
        mock_llm.generate.assert_called_once()

    @patch("nemo_curator.models.vllm_model.VLLM_AVAILABLE", True)
    @patch("nemo_curator.models.vllm_model._get_gpu_count", return_value=1)
    @patch("nemo_curator.models.vllm_model._get_max_model_len_from_config", return_value=4096)
    @patch("nemo_curator.models.vllm_model.LLM")
    def test_chat_with_image_content(self, mock_llm_cls, mock_config, mock_gpu):
        """Test chat() with image_url content blocks."""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="A chart showing trends.")]

        mock_llm = MagicMock()
        mock_llm.chat.return_value = [mock_output]
        mock_llm_cls.return_value = mock_llm

        model = VLLMModel(
            model="nvidia/test-vlm",
            max_tokens=1024,
            allowed_local_media_path="/",
        )
        model.setup()

        messages = [[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this chart."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }]]

        results = model.chat(messages)
        assert len(results) == 1
        assert "chart" in results[0]
        mock_llm.chat.assert_called_once()

    @patch("nemo_curator.models.vllm_model.VLLM_AVAILABLE", True)
    @patch("nemo_curator.models.vllm_model._get_gpu_count", return_value=1)
    @patch("nemo_curator.models.vllm_model._get_max_model_len_from_config", return_value=4096)
    @patch("nemo_curator.models.vllm_model.LLM")
    def test_setup_with_allowed_media_path(self, mock_llm_cls, mock_config, mock_gpu):
        """Test that allowed_local_media_path is passed to vLLM."""
        model = VLLMModel(
            model="nvidia/test-vlm",
            allowed_local_media_path="/data/images",
        )
        model.setup()

        call_kwargs = mock_llm_cls.call_args[1]
        assert call_kwargs["allowed_local_media_path"] == "/data/images"

    @patch("nemo_curator.models.vllm_model.VLLM_AVAILABLE", True)
    @patch("nemo_curator.models.vllm_model._get_gpu_count", return_value=1)
    @patch("nemo_curator.models.vllm_model._get_max_model_len_from_config", return_value=4096)
    @patch("nemo_curator.models.vllm_model.LLM")
    def test_setup_with_extra_llm_kwargs(self, mock_llm_cls, mock_config, mock_gpu):
        """Test that extra_llm_kwargs are passed to vLLM."""
        model = VLLMModel(
            model="nvidia/test-vlm",
            extra_llm_kwargs={"disable_log_stats": True},
        )
        model.setup()

        call_kwargs = mock_llm_cls.call_args[1]
        assert call_kwargs["disable_log_stats"] is True

    @patch("nemo_curator.models.vllm_model.VLLM_AVAILABLE", False)
    def test_setup_raises_without_vllm(self):
        model = VLLMModel(model="nvidia/test-vlm")
        with pytest.raises(ImportError, match="vLLM is required"):
            model.setup()


class TestVLLMVisionModelAlias:
    """Verify backward-compatible alias still works."""

    def test_alias_is_same_class(self):
        assert VLLMVisionModel is VLLMModel

    def test_alias_instantiation(self):
        model = VLLMVisionModel(model="nvidia/test-vlm", temperature=0.0)
        assert isinstance(model, VLLMModel)
        assert model.temperature == 0.0
