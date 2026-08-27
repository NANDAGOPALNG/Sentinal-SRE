"""
app/database.py
================
Database engine and session configuration.

Responsibility: own the SQLAlchemy engine, session factory, declarative
Base, and the `get_db` FastAPI dependency. No business logic and no
HTTP concerns live here.

Connection info comes from the DATABASE_URL environment variable so
credentials are never hardcoded in source.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentinal:sentinal@postgres:5432/sentinal_orders",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
