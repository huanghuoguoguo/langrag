"""Database connection and session management."""

from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = Path(__file__).parents[1] / "data" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create engine
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """初始化数据库，创建所有表"""
    SQLModel.metadata.create_all(engine)
    logger.info(f"Database initialized at {DB_PATH}")


def get_session():
    """获取数据库会话"""
    with Session(engine) as session:
        yield session
