"""
app/models.py
=============
SQLAlchemy ORM models.

Responsibility: define the database table shape. No queries, no
business rules -- just the schema.
"""

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from database import Base


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String, nullable=False)
    product_name = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
