"""LLM 配置服务"""

from sqlmodel import Session, select
from typing import Optional

from web.models.database import LLMConfig
from web.core.rag_kernel import RAGKernel


class LLMService:
    """LLM 配置服务"""
    
    @staticmethod
    def save_config(
        session: Session,
        rag_kernel: RAGKernel,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> LLMConfig:
        """保存 LLM 配置"""
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
        else:
            # Create new
            config = LLMConfig(
                name=name,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
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
    def get_active_config(session: Session) -> Optional[LLMConfig]:
        """获取当前激活的配置"""
        statement = select(LLMConfig).where(LLMConfig.is_active == True)
        return session.exec(statement).first()
    
    @staticmethod
    def activate_config(
        session: Session,
        rag_kernel: RAGKernel,
        name: str
    ) -> Optional[LLMConfig]:
        """激活指定配置"""
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
            
            # Inject into RAG kernel
            rag_kernel.set_llm(
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
        return config
    
    @staticmethod
    def list_all(session: Session) -> list[LLMConfig]:
        """列出所有 LLM 配置"""
        statement = select(LLMConfig)
        return list(session.exec(statement).all())
