"""
tests/test_orders.py
=====================
Integration tests for the Order Management API.

These tests exercise the real request path (main -> service ->
repository -> database) but swap the production Postgres engine for
an in-memory SQLite engine via FastAPI's dependency override, so the
suite runs fast and without requiring the full Docker Compose stack.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app import main as app_main

TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app_main.app.dependency_overrides[get_db] = override_get_db

client = TestClient(app_main.app)


@pytest.fixture(autouse=True)
def reset_db():
    """Give every test a clean set of tables."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


def test_create_order_persists_and_returns_it():
    response = client.post(
        "/orders",
        json={"customer_name": "Alice", "product_name": "Widget", "quantity": 3},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["customer_name"] == "Alice"
    assert data["product_name"] == "Widget"
    assert data["quantity"] == 3
    assert isinstance(data["id"], int)
    assert "created_at" in data


def test_get_existing_order_returns_the_order_that_was_created():
    create_response = client.post(
        "/orders",
        json={"customer_name": "Bob", "product_name": "Gadget", "quantity": 2},
    )
    order_id = create_response.json()["id"]

    get_response = client.get(f"/orders/{order_id}")

    assert get_response.status_code == 200
    data = get_response.json()
    assert data["id"] == order_id
    assert data["customer_name"] == "Bob"
    assert data["product_name"] == "Gadget"
    assert data["quantity"] == 2


def test_get_nonexistent_order_returns_404():
    response = client.get("/orders/999999")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_create_order_rejects_zero_quantity():
    response = client.post(
        "/orders",
        json={"customer_name": "Carol", "product_name": "Sprocket", "quantity": 0},
    )

    # Pydantic's gt=0 constraint rejects this before it reaches the service layer.
    assert response.status_code == 422
