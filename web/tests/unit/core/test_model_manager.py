"""Tests for WebModelManager and LLMFactory."""

from unittest.mock import MagicMock, patch

import pytest

from langrag.llm.base import BaseLLM


class TestWebModelManager:
    """Tests for WebModelManager class."""

    def _create_mock_llm(self) -> MagicMock:
        """Create a mock LLM instance."""
        mock = MagicMock(spec=BaseLLM)
        return mock

    def test_init(self):
        """Manager initializes with empty registry."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()

        assert manager._models == {}
        assert manager._default_model_name is None

    def test_register_model(self):
        """register_model adds model to registry."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()

        manager.register_model("test-model", mock_llm)

        assert "test-model" in manager._models
        assert manager._models["test-model"] == mock_llm

    def test_register_model_sets_default(self):
        """First registered model becomes default."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()

        manager.register_model("first-model", mock_llm)

        assert manager._default_model_name == "first-model"

    def test_register_model_explicit_default(self):
        """set_as_default=True overrides existing default."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm1 = self._create_mock_llm()
        mock_llm2 = self._create_mock_llm()

        manager.register_model("first", mock_llm1)
        manager.register_model("second", mock_llm2, set_as_default=True)

        assert manager._default_model_name == "second"

    def test_get_model_by_name(self):
        """get_model returns registered model by name."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()
        manager.register_model("my-model", mock_llm)

        result = manager.get_model("my-model")

        assert result == mock_llm

    def test_get_model_returns_none_for_unknown(self):
        """get_model returns None for unknown model."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()

        result = manager.get_model("nonexistent")

        assert result is None

    def test_get_model_returns_default(self):
        """get_model with no name returns default model."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()
        manager.register_model("default-model", mock_llm)

        result = manager.get_model()

        assert result == mock_llm

    def test_list_models(self):
        """list_models returns all registered model names."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        manager.register_model("model-a", self._create_mock_llm())
        manager.register_model("model-b", self._create_mock_llm())

        names = manager.list_models()

        assert set(names) == {"model-a", "model-b"}

    def test_set_stage_model(self):
        """set_stage_model configures stage."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()
        manager.register_model("chat-model", mock_llm)

        manager.set_stage_model("chat", "chat-model")

        assert manager._stage_config["chat"] == "chat-model"

    def test_set_stage_model_invalid_stage(self):
        """set_stage_model raises for invalid stage."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()
        manager.register_model("model", mock_llm)

        with pytest.raises(ValueError, match="Invalid stage"):
            manager.set_stage_model("invalid-stage", "model")

    def test_set_stage_model_unknown_model(self):
        """set_stage_model raises for unknown model."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()

        with pytest.raises(ValueError, match="not registered"):
            manager.set_stage_model("chat", "nonexistent-model")

    def test_get_stage_model(self):
        """get_stage_model returns model for configured stage."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()
        manager.register_model("router-model", mock_llm)
        manager.set_stage_model("router", "router-model")

        result = manager.get_stage_model("router")

        assert result == mock_llm

    def test_get_stage_model_unconfigured(self):
        """get_stage_model returns None for unconfigured stage."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()

        result = manager.get_stage_model("chat")

        assert result is None

    def test_remove_model(self):
        """remove_model deletes model and clears stage configs."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()
        manager.register_model("to-remove", mock_llm)
        manager.set_stage_model("chat", "to-remove")

        manager.remove_model("to-remove")

        assert "to-remove" not in manager._models
        assert manager._stage_config["chat"] is None

    def test_list_stages(self):
        """list_stages returns all available stages."""
        from web.core.model_manager import WebModelManager
        from langrag.llm.stages import LLMStage

        manager = WebModelManager()

        stages = manager.list_stages()

        assert set(stages) == set(LLMStage.ALL_STAGES)

    def test_get_stage_config(self):
        """get_stage_config returns copy of stage configuration."""
        from web.core.model_manager import WebModelManager

        manager = WebModelManager()
        mock_llm = self._create_mock_llm()
        manager.register_model("model", mock_llm)
        manager.set_stage_model("rewriter", "model")

        config = manager.get_stage_config()

        assert config["rewriter"] == "model"
        # Ensure it's a copy
        config["rewriter"] = "changed"
        assert manager._stage_config["rewriter"] == "model"


class TestLLMFactory:
    """Tests for LLMFactory class."""

    def test_create_mock_llm(self):
        """create with type='mock' returns MockLLM."""
        from web.core.factories.llm_factory import LLMFactory

        llm = LLMFactory.create({"type": "mock"})

        assert llm is not None
        assert "MockLLM" in type(llm).__name__

    @patch("langrag.llm.providers.local.LocalLLM")
    @patch("os.path.exists")
    def test_create_local_llm(self, mock_exists, mock_local_llm_class):
        """create with type='local' returns LocalLLM."""
        from web.core.factories.llm_factory import LLMFactory

        mock_exists.return_value = True
        mock_llm = MagicMock()
        mock_local_llm_class.return_value = mock_llm

        result = LLMFactory.create({
            "type": "local",
            "model_path": "/path/to/model.gguf"
        })

        assert result == mock_llm
        mock_local_llm_class.assert_called_once()

    @patch("openai.AsyncOpenAI")
    @patch("langrag.llm.providers.openai.OpenAILLM")
    def test_create_remote_llm(self, mock_openai_llm_class, mock_async_openai):
        """create with type='remote' returns OpenAILLM."""
        from web.core.factories.llm_factory import LLMFactory

        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client
        mock_llm = MagicMock()
        mock_openai_llm_class.return_value = mock_llm

        result = LLMFactory.create({
            "type": "remote",
            "base_url": "https://api.example.com/v1",
            "api_key": "test-key",
            "model": "gpt-4"
        })

        assert result == mock_llm
        mock_openai_llm_class.assert_called_once()

    def test_create_invalid_config(self):
        """create with invalid config raises ValueError."""
        from web.core.factories.llm_factory import LLMFactory

        with pytest.raises(ValueError, match="Invalid LLM configuration"):
            LLMFactory.create({"type": "unknown"})
