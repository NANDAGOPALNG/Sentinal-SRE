class OrderService:
    """Business logic for creating and retrieving orders."""

    def __init__(self, db: Session):
        self.repository = OrderRepository(db)

    def create_order(self, product_name: str, customer_name: str, quantity: int):
        # Corrected parameter order to match the expected order in the repository's create method
        customer_name = (customer_name or "").strip()
        product_name = (product_name or "").strip()

        if not customer_name:
            raise InvalidOrderError("customer_name must not be empty.")
        if not product_name:
            raise InvalidOrderError("product_name must not be empty.")
        if quantity <= 0:
            raise InvalidOrderError("quantity must be greater than zero.")

        return self.repository.create(product_name, customer_name, quantity)

    def get_order(self, order_id: int):
        order = self.repository.get_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"Order {order_id} not found.")
        return order