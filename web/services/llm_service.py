"""LLM Configuration Service"""


from sqlmodel import Session, select

from web.core.rag_kernel import RAGKernel
from web.models.database import LLMConfig


class LLMService:
    """LLM Configuration Service"""

    @staticmethod
    @staticmethod
    def save_config(
        session: Session,
        rag_kernel: RAGKernel,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model_path: str | None = None
    ) -> LLMConfig:
        """Save LLM configuration"""
        # Check if exists
        statement = select(LLMConfig).where(LLMConfig.name == name)
        config = session.exec(statement).first()

        if config:
            # Update existing
            config.base_url = base_url
            config.api_key = api_key
            config.model = model
            config.temperature = temperature
            config.max_tokens = max_tokens
            config.model_path = model_path
        else:
            # Create new
            config = LLMConfig(
                name=name,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                model_path=model_path
            )

        session.add(config)
        session.commit()
        session.refresh(config)

        # No longer auto-activate, all models are available by default

        return config

    @staticmethod
    def get_active_config(session: Session) -> LLMConfig | None:
        """Get the currently active configuration"""
        statement = select(LLMConfig).where(LLMConfig.is_active == True)
        return session.exec(statement).first()

    @staticmethod
    def activate_config(
        session: Session,
        rag_kernel: RAGKernel,
        name: str
    ) -> LLMConfig | None:
        """Set as default configuration (for backward compatibility)"""
        # Find target config
        statement = select(LLMConfig).where(LLMConfig.name == name)
        config = session.exec(statement).first()
        if config:
            # No longer enforce single active, just return the config
            # This method is kept for backward compatibility
            return config
        return None

    @staticmethod
    def list_all(session: Session) -> list[LLMConfig]:
        """List all LLM configurations"""
        statement = select(LLMConfig)
        return list(session.exec(statement).all())
