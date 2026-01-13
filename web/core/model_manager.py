from typing import Dict, Optional, Any
from langrag.llm.base import BaseLLM, ModelManager
from langrag.llm.stages import LLMStage
from loguru import logger


class WebModelManager(ModelManager):
    """
    Web应用的ModelManager实现。

    实现了LangRAG的ModelManager接口，支持：
    1. 多LLM实例管理
    2. 阶段化配置（不同阶段使用不同模型）
    3. 向后兼容的API

    Web Application's ModelManager implementation.

    Implements LangRAG's ModelManager interface with support for:
    1. Multiple LLM instance management
    2. Stage-based configuration (different models for different stages)
    3. Backward-compatible APIs
    """

    def __init__(self):
        # 模型存储
        self._models: Dict[str, BaseLLM] = {}
        self._default_model_name: Optional[str] = None

        # 阶段配置：stage -> model_name
        self._stage_config: Dict[str, Optional[str]] = {
            stage: None for stage in LLMStage.ALL_STAGES
        }

    # ========== ModelManager接口实现 ==========

    def get_model(self, name: str = None) -> Optional[BaseLLM]:
        """
        按名称获取模型实例。

        Get an LLM instance by name.

        Args:
            name: Model name to retrieve. If None, returns default model.

        Returns:
            BaseLLM instance or None if not found.
        """
        target_name = name or self._default_model_name
        if not target_name:
            return None

        return self._models.get(target_name)

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self._models.keys())

    def get_stage_model(self, stage: str) -> Optional[BaseLLM]:
        """
        获取指定阶段配置的模型。

        Get the LLM configured for a specific stage.

        Args:
            stage: Stage name (e.g., "chat", "router", "rewriter")

        Returns:
            BaseLLM instance configured for the stage, or None if not configured.
        """
        if not LLMStage.is_valid_stage(stage):
            logger.warning(f"Invalid stage: {stage}")
            return None

        model_name = self._stage_config.get(stage)
        if model_name:
            return self.get_model(model_name)

        return None

    def set_stage_model(self, stage: str, model_name: str) -> None:
        """
        为指定阶段配置模型。

        Configure an LLM for a specific stage.

        Args:
            stage: Stage name (e.g., "chat", "router", "rewriter")
            model_name: Name of the model to assign to this stage

        Raises:
            ValueError: If stage or model_name is invalid
        """
        if not LLMStage.is_valid_stage(stage):
            raise ValueError(f"Invalid stage: {stage}. Valid stages: {LLMStage.ALL_STAGES}")

        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not registered. Available models: {self.list_models()}")

        self._stage_config[stage] = model_name
        logger.info(f"Configured stage '{stage}' to use model '{model_name}'")

    def get_stage_model_name(self, stage: str) -> Optional[str]:
        """
        获取指定阶段配置的模型名称。

        Get the model name configured for a specific stage.

        Args:
            stage: Stage name

        Returns:
            Model name configured for the stage, or None if not configured.
        """
        if not LLMStage.is_valid_stage(stage):
            return None

        return self._stage_config.get(stage)

    def list_stages(self) -> list[str]:
        """
        列出所有可用阶段。

        List all available stages.

        Returns:
            List of stage names.
        """
        return LLMStage.ALL_STAGES.copy()

    # ========== 扩展功能 ==========

    def register_model(self, name: str, llm_instance: BaseLLM, set_as_default: bool = False) -> None:
        """
        注册新的LLM实例。

        Register a new LLM instance.

        Args:
            name: Unique identifier for the model (e.g., "gpt-4", "qwen-local").
            llm_instance: The initialized LLM instance (BaseLLM subclass).
            set_as_default: Whether to set this model as the default.
        """
        if name in self._models:
            logger.warning(f"Overwriting existing model registration: {name}")

        self._models[name] = llm_instance
        logger.info(f"Registered LLM: {name} ({type(llm_instance).__name__})")

        if set_as_default or self._default_model_name is None:
            self._default_model_name = name
            logger.info(f"Set default LLM to: {name}")

    def remove_model(self, name: str) -> None:
        """
        注销模型。

        Unregister a model.
        """
        if name in self._models:
            del self._models[name]

            # 清理阶段配置中对这个模型的引用
            for stage in self._stage_config:
                if self._stage_config[stage] == name:
                    self._stage_config[stage] = None
                    logger.info(f"Removed model '{name}' from stage '{stage}'")

            # Reset default if we deleted it
            if self._default_model_name == name:
                self._default_model_name = next(iter(self._models)) if self._models else None
                if self._default_model_name:
                    logger.info(f"Set new default LLM to: {self._default_model_name}")

    def get_stage_config(self) -> Dict[str, Optional[str]]:
        """
        获取所有阶段的配置。

        Get configuration for all stages.

        Returns:
            Dict mapping stage names to model names (or None if not configured).
        """
        return self._stage_config.copy()

    def set_default_model(self, name: str) -> None:
        """
        设置默认模型。

        Set the default model.

        Args:
            name: Model name to set as default

        Raises:
            ValueError: If model name is not registered
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered. Available models: {self.list_models()}")

        self._default_model_name = name
        logger.info(f"Set default LLM to: {name}")

    def get_default_model_name(self) -> Optional[str]:
        """获取默认模型名称。"""
        return self._default_model_name

    def get_model_info(self, name: str = None) -> Optional[Dict[str, Any]]:
        """
        获取模型信息。

        Get model information.

        Args:
            name: Model name. If None, returns default model info.

        Returns:
            Dict with model info, or None if not found.
        """
        model = self.get_model(name)
        if not model:
            return None

        target_name = name or self._default_model_name
        return {
            "name": target_name,
            "type": type(model).__name__,
            "stages": [stage for stage, model_name in self._stage_config.items() if model_name == target_name]
        }

    def list_models_detailed(self) -> list[Dict[str, Any]]:
        """
        获取所有模型的详细信息。

        Get detailed information for all models.

        Returns:
            List of dicts with model information.
        """
        result = []
        for name in self.list_models():
            info = self.get_model_info(name)
            if info:
                result.append(info)
        return result
