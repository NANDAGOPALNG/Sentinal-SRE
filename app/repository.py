"""
app/repository.py
==================
Data access layer.

Responsibility: talk to the database via SQLAlchemy sessions. No
validation, no HTTP concerns -- just persistence operations on the
Order model.
"""

from typing import Optional

from sqlalchemy.orm import Session

from .models import Order


class OrderRepository:
    """Encapsulates all direct database operations for orders."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, customer_name: str, product_name: str, quantity: int) -> Order:
        order = Order(
            customer_name=customer_name,
            product_name=product_name,
            quantity=quantity,
        )
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order

    def get_by_id(self, order_id: int) -> Optional[Order]:
        return self.db.query(Order).filter(Order.id == order_id).first()
