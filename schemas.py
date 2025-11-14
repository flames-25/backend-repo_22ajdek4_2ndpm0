"""
Database Schemas for the Co-Working/Service Center Management System

Each Pydantic model corresponds to a MongoDB collection using the lowercase of
its class name for the collection name.

Use these schemas as the single source of truth for documents stored in MongoDB.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal

# -----------------------------
# Core Entities
# -----------------------------

class Customer(BaseModel):
    first_name: str = Field(..., description="Customer first name")
    last_name: str = Field(..., description="Customer last name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")
    tags: List[str] = Field(default_factory=list, description="Customer tags/segments")
    notes: Optional[str] = Field(None, description="Internal notes")
    is_active: bool = Field(True, description="Whether the customer is active")

class Product(BaseModel):
    sku: Optional[str] = Field(None, description="Unique SKU")
    name: str = Field(..., description="Product or Service name")
    description: Optional[str] = Field(None, description="Description")
    category: str = Field(..., description="Category name")
    price: float = Field(..., ge=0, description="Unit price in default currency")
    is_service: bool = Field(False, description="True if time-based service")
    unit: Optional[Literal["item","hour","minute"]] = Field("item", description="Pricing unit")
    stock: Optional[int] = Field(None, ge=0, description="Current stock (for physical items)")
    low_stock_threshold: Optional[int] = Field(None, ge=0, description="Low stock alert threshold")
    tax_rate: float = Field(0.0, ge=0, description="Tax rate as a fraction (e.g., 0.07 for 7%)")
    is_active: bool = Field(True, description="Whether product is available for sale")

class SessionItem(BaseModel):
    product_id: str = Field(..., description="Referenced Product _id as string")
    name: str = Field(..., description="Snapshot of product name")
    quantity: int = Field(1, ge=1, description="Quantity consumed")
    unit_price: float = Field(..., ge=0, description="Price per unit at time of adding")
    line_total: float = Field(..., ge=0, description="Computed quantity * unit_price")

class PauseWindow(BaseModel):
    start: float = Field(..., description="UTC timestamp seconds when pause started")
    end: Optional[float] = Field(None, description="UTC timestamp seconds when pause ended")

class Discount(BaseModel):
    type: Optional[Literal["percent","amount"]] = Field(None, description="Discount type")
    value: Optional[float] = Field(None, ge=0, description="Percent (0-100) or currency amount")
    reason: Optional[str] = Field(None, description="Discount reason or code")

class Session(BaseModel):
    customer_id: str = Field(..., description="Referenced Customer _id as string")
    status: Literal["active","paused","completed"] = Field("active", description="Session status")
    started_at: float = Field(..., description="UTC timestamp seconds when session started")
    ended_at: Optional[float] = Field(None, description="UTC timestamp seconds when session ended")
    pauses: List[PauseWindow] = Field(default_factory=list, description="Pause windows")
    items: List[SessionItem] = Field(default_factory=list, description="Consumed products/services")
    discount: Optional[Discount] = Field(None, description="Any applied discount")
    elapsed_active_seconds: float = Field(0, ge=0, description="Total active seconds so far")
    subtotal: float = Field(0, ge=0, description="Subtotal before discounts and tax")
    tax: float = Field(0, ge=0, description="Total tax amount")
    total: float = Field(0, ge=0, description="Final total after discounts and tax")
    payment_status: Literal["unpaid","paid"] = Field("unpaid", description="Payment status")

class Payment(BaseModel):
    session_id: str = Field(..., description="Referenced Session _id as string")
    amount: float = Field(..., ge=0, description="Amount paid")
    method: Literal["cash","card","external"] = Field(..., description="Payment method")
    status: Literal["processed","failed"] = Field("processed", description="Payment status")
    reference: Optional[str] = Field(None, description="External reference or transaction id")

class Activity(BaseModel):
    type: Literal["checkin","pause","resume","checkout","inventory_alert","system"] = Field(...)
    message: str = Field(...)
    customer_id: Optional[str] = None
    session_id: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    created_at: float = Field(..., description="UTC timestamp seconds")
