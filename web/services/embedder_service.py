"""Embedder Configuration Service"""


from sqlmodel import Session, select

from web.core.rag_kernel import RAGKernel
from web.models.database import EmbedderConfig


class EmbedderService:
    """Embedder Configuration Service"""

    @staticmethod
    def save_config(
        session: Session,
        rag_kernel: RAGKernel,
        name: str,
        embedder_type: str,
        model: str,
        base_url: str = "",
        api_key: str = ""
    ) -> EmbedderConfig:
        """Save Embedder configuration"""
        # Check if exists
        statement = select(EmbedderConfig).where(EmbedderConfig.name == name)
        config = session.exec(statement).first()

        if config:
            # Update existing
            config.embedder_type = embedder_type
            config.base_url = base_url
            config.api_key = api_key
            config.model = model
        else:
            # Create new
            config = EmbedderConfig(
                name=name,
                embedder_type=embedder_type,
                base_url=base_url,
                api_key=api_key,
                model=model
            )

        session.add(config)
        session.commit()
        session.refresh(config)

        # Inject into RAG kernel
        rag_kernel.set_embedder(embedder_type, model, base_url, api_key)

        return config

    @staticmethod
    def get_active_config(session: Session) -> EmbedderConfig | None:
        """Get the currently active configuration"""
        statement = select(EmbedderConfig).where(EmbedderConfig.is_active == True)
        return session.exec(statement).first()

    @staticmethod
    def activate_config(
        session: Session,
        rag_kernel: RAGKernel,
        name: str
    ) -> EmbedderConfig | None:
        """Activate the specified configuration"""
        # Deactivate all
        statement = select(EmbedderConfig)
        configs = session.exec(statement).all()
        for cfg in configs:
            cfg.is_active = False
            session.add(cfg)

        # Activate target
        statement = select(EmbedderConfig).where(EmbedderConfig.name == name)
        config = session.exec(statement).first()
        if config:
            config.is_active = True
            session.add(config)
            session.commit()
            session.refresh(config)

            # Inject into RAG kernel
            rag_kernel.set_embedder(config.embedder_type, config.model, config.base_url, config.api_key)

        return config

    @staticmethod
    def list_all(session: Session) -> list[EmbedderConfig]:
        """List all Embedder configurations"""
        statement = select(EmbedderConfig)
        return list(session.exec(statement).all())
