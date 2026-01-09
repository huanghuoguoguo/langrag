"""Database connection and session management."""

import logging

from sqlmodel import Session, SQLModel, create_engine

from web.config import DATABASE_URL, DB_DIR

logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """Initialize database and create all tables"""
    SQLModel.metadata.create_all(engine)
    logger.info(f"Database initialized at {DB_DIR / 'app.db'}")


def get_session():
    """Get database session"""
    with Session(engine) as session:
        yield session
