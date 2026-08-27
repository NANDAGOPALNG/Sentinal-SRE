"""
app/service.py
===============
Business logic layer.

Responsibility: validate input and orchestrate repository calls. No
SQL, no HTTP status codes -- those belong to repository.py and
main.py respectively. Domain errors are raised as plain exceptions
so main.py can translate them into the appropriate HTTP responses.
"""

from sqlalchemy.orm import Session

from repository import OrderRepository


class OrderNotFoundError(Exception):
    """Raised when a requested order does not exist."""


class InvalidOrderError(Exception):
    """Raised when order input fails a business rule."""


class OrderService:
    """Business logic for creating and retrieving orders."""

    def __init__(self, db: Session):
        self.repository = OrderRepository(db)

    def create_order(self, customer_name: str, product_name: str, quantity: int):
        customer_name = (customer_name or "").strip()
        product_name = (product_name or "").strip()

        if not customer_name:
            raise InvalidOrderError("customer_name must not be empty.")
        if not product_name:
            raise InvalidOrderError("product_name must not be empty.")
        if quantity <= 0:
            raise InvalidOrderError("quantity must be greater than zero.")

        return self.repository.create(customer_name, product_name, quantity)

    def get_order(self, order_id: int):
        order = self.repository.get_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"Order {order_id} not found.")
        return order
