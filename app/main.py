"""
HTTP layer.

Responsibility: define routes, parse/validate request bodies via
Pydantic, translate service-layer exceptions into HTTP responses, and
log request outcomes. No business logic and no direct DB access live
here -- those belong to service.py and repository.py.
"""

import logging
import os
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from . import models  # noqa: F401 - registers Order with SQLAlchemy metadata
from .database import Base, engine, get_db
from .service import InvalidOrderError, OrderNotFoundError, OrderService


LOG_FILE = "/var/log/sentinal/app.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("ProdApp")

app = FastAPI(title="SentinalSRE-Sandbox")


# Demo app: create tables on startup instead of running migrations.
#Base.metadata.create_all(bind=engine)


class OrderCreateRequest(BaseModel):
    customer_name: str = Field(..., min_length=1)
    product_name: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)


class OrderResponse(BaseModel):
    id: int
    customer_name: str
    product_name: str
    quantity: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


@app.get("/")
def read_root():
    logger.info("Health check performed.")
    return {"status": "online"}


@app.post("/orders", response_model=OrderResponse, status_code=201)
def create_order(
    payload: OrderCreateRequest,
    db: Session = Depends(get_db),
):
    service = OrderService(db)

    try:
        order = service.create_order(
            payload.customer_name,
            payload.product_name,
            payload.quantity,
        )
    except InvalidOrderError as e:
        logger.warning(f"Rejected invalid order payload: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    logger.info(f"Order {order.id} created for {order.customer_name}.")
    return order


@app.get("/orders/{order_id}", response_model=OrderResponse)
def get_order(
    order_id: int,
    db: Session = Depends(get_db),
):
    service = OrderService(db)

    try:
        order = service.get_order(order_id)
    except OrderNotFoundError as e:
        logger.info(f"Order lookup failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    return order