import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentinal:sentinal@postgres:5432/sentinal_orders",
)

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=4,
    pool_timeout=5,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()