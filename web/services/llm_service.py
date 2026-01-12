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

        # If it was active or it's the only one, activate it
        statement = select(LLMConfig)
        all_configs = session.exec(statement).all()
        if len(all_configs) == 1 or config.is_active:
            LLMService.activate_config(session, rag_kernel, config.name)

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
        """Activate the specified configuration"""
        # Deactivate all
        statement = select(LLMConfig)
        configs = session.exec(statement).all()
        for cfg in configs:
            cfg.is_active = False
            session.add(cfg)

        # Activate target
        statement = select(LLMConfig).where(LLMConfig.name == name)
        config = session.exec(statement).first()
        if config:
            config.is_active = True
            session.add(config)
            session.commit()
            session.refresh(config)

            # Inject into RAG kernel using generic add_llm logic
            llm_conf = {
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            if config.model_path:
                llm_conf["type"] = "local"
                llm_conf["model_path"] = config.model_path
            else:
                llm_conf["type"] = "remote"
                llm_conf["base_url"] = config.base_url
                llm_conf["api_key"] = config.api_key
                llm_conf["model"] = config.model
            
            rag_kernel.add_llm(name=config.name, config=llm_conf, set_as_default=True)

        return config

    @staticmethod
    def list_all(session: Session) -> list[LLMConfig]:
        """List all LLM configurations"""
        statement = select(LLMConfig)
        return list(session.exec(statement).all())
