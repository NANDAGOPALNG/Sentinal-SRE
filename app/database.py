import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentinal:sentinal@postgres:5432/sentinal_orders",
)

engine = create_engine(
    DATABASE_URL,
    pool_size=5,  # Adjusted from 3 to 5
    max_overflow=10,  # Adjusted from 2 to 10
    pool_timeout=30,  # Added a connection timeout setting
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a request-scoped DB session."""
    db = SessionLocal()
    yield db
    db.close()