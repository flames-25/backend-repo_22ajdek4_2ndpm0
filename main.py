import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import Customer, Product, Session, SessionItem, PauseWindow, Discount, Payment, Activity

app = FastAPI(title="Operations & BI Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
# -----------------------------

def now_ts() -> float:
    return time.time()


def oid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")


def calculate_session_totals(sess: dict) -> dict:
    """Recalculate elapsed_active_seconds, subtotal, tax, total based on items and time."""
    # Active seconds: (now or ended) - started - sum(paused)
    end_ts = sess.get("ended_at") or now_ts()
    paused_seconds = 0.0
    for p in sess.get("pauses", []):
        s = p.get("start")
        e = p.get("end") or (end_ts if sess.get("status") != "active" else now_ts())
        if s and e:
            paused_seconds += max(0.0, e - s)
    elapsed = max(0.0, end_ts - sess.get("started_at", end_ts) - paused_seconds)

    # Subtotal from items
    subtotal = 0.0
    tax_total = 0.0
    for it in sess.get("items", []):
        line = float(it.get("quantity", 1)) * float(it.get("unit_price", 0))
        it["line_total"] = round(line, 2)
        subtotal += line
    # Apply discount
    discount = sess.get("discount")
    if discount and discount.get("type") and discount.get("value"):
        if discount["type"] == "percent":
            subtotal = subtotal * (1 - float(discount["value"]) / 100.0)
        elif discount["type"] == "amount":
            subtotal = max(0.0, subtotal - float(discount["value"]))

    # Tax: naive sum of item tax_rate snapshot if provided in payload
    # If items include tax_rate in payload, use it; else 0
    for it in sess.get("items", []):
        rate = float(it.get("tax_rate", 0))
        tax_total += (it.get("line_total", 0.0)) * rate

    total = max(0.0, subtotal + tax_total)

    sess["elapsed_active_seconds"] = round(elapsed, 2)
    sess["subtotal"] = round(subtotal, 2)
    sess["tax"] = round(tax_total, 2)
    sess["total"] = round(total, 2)
    return sess


def log_activity(type_: str, message: str, customer_id: Optional[str] = None, session_id: Optional[str] = None, payload: dict = None):
    create_document("activity", Activity(
        type=type_,
        message=message,
        customer_id=customer_id,
        session_id=session_id,
        payload=payload or {},
        created_at=now_ts(),
    ))


# -----------------------------
# Health & Schema
# -----------------------------

@app.get("/")
def read_root():
    return {"message": "Operations & BI Platform API Running"}


@app.get("/schema")
def get_schema():
    return {
        "customer": Customer.model_json_schema(),
        "product": Product.model_json_schema(),
        "session": Session.model_json_schema(),
        "payment": Payment.model_json_schema(),
        "activity": Activity.model_json_schema(),
    }


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# -----------------------------
# Customers
# -----------------------------

class CustomerCreate(Customer):
    pass


@app.post("/customers")
def create_customer(payload: CustomerCreate):
    _id = create_document("customer", payload)
    log_activity("system", f"New customer created: {payload.first_name} {payload.last_name}")
    return {"_id": _id}


@app.get("/customers")
def list_customers(q: Optional[str] = None, limit: int = 50):
    filt = {}
    if q:
        filt = {"$or": [
            {"first_name": {"$regex": q, "$options": "i"}},
            {"last_name": {"$regex": q, "$options": "i"}},
            {"email": {"$regex": q, "$options": "i"}},
            {"phone": {"$regex": q, "$options": "i"}},
        ]}
    docs = list(db["customer"].find(filt).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    doc = db["customer"].find_one({"_id": oid(customer_id)})
    if not doc:
        raise HTTPException(404, "Customer not found")
    doc["_id"] = str(doc["_id"])
    return doc


# -----------------------------
# Products
# -----------------------------

class ProductCreate(Product):
    pass


@app.post("/products")
def create_product(payload: ProductCreate):
    _id = create_document("product", payload)
    return {"_id": _id}


@app.get("/products")
def list_products(active_only: bool = True, limit: int = 200):
    filt = {"is_active": True} if active_only else {}
    docs = list(db["product"].find(filt).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


# -----------------------------
# Sessions
# -----------------------------

class SessionStart(BaseModel):
    customer_id: str
    start_time: Optional[float] = None


@app.post("/sessions/start")
def start_session(payload: SessionStart):
    sess = Session(
        customer_id=payload.customer_id,
        status="active",
        started_at=payload.start_time or now_ts(),
        pauses=[],
        items=[],
        discount=None,
        elapsed_active_seconds=0,
        subtotal=0,
        tax=0,
        total=0,
        payment_status="unpaid"
    ).model_dump()

    calculate_session_totals(sess)
    inserted_id = db["session"].insert_one(sess).inserted_id
    sid = str(inserted_id)
    cust = db["customer"].find_one({"_id": oid(payload.customer_id)})
    name = f"{cust.get('first_name','')} {cust.get('last_name','')}" if cust else payload.customer_id
    log_activity("checkin", f"Customer checked in: {name}", customer_id=payload.customer_id, session_id=sid)
    return {"_id": sid, **sess}


@app.post("/sessions/{session_id}/pause")
def pause_session(session_id: str):
    sess = db["session"].find_one({"_id": oid(session_id)})
    if not sess:
        raise HTTPException(404, "Session not found")
    if sess.get("status") != "active":
        raise HTTPException(400, "Only active sessions can be paused")

    sess.setdefault("pauses", []).append({"start": now_ts(), "end": None})
    sess["status"] = "paused"
    calculate_session_totals(sess)
    db["session"].update_one({"_id": sess["_id"]}, {"$set": sess})
    log_activity("pause", f"Session paused", session_id=session_id, customer_id=sess.get("customer_id"))
    return {"ok": True}


@app.post("/sessions/{session_id}/resume")
def resume_session(session_id: str):
    sess = db["session"].find_one({"_id": oid(session_id)})
    if not sess:
        raise HTTPException(404, "Session not found")
    if sess.get("status") != "paused":
        raise HTTPException(400, "Only paused sessions can be resumed")

    # close last pause window
    for p in reversed(sess.get("pauses", [])):
        if p.get("end") is None:
            p["end"] = now_ts()
            break
    sess["status"] = "active"
    calculate_session_totals(sess)
    db["session"].update_one({"_id": sess["_id"]}, {"$set": sess})
    log_activity("resume", f"Session resumed", session_id=session_id, customer_id=sess.get("customer_id"))
    return {"ok": True}


class AddItemPayload(BaseModel):
    product_id: str
    quantity: int = 1


@app.post("/sessions/{session_id}/items")
def add_item(session_id: str, body: AddItemPayload):
    sess = db["session"].find_one({"_id": oid(session_id)})
    if not sess:
        raise HTTPException(404, "Session not found")

    prod = db["product"].find_one({"_id": oid(body.product_id)})
    if not prod:
        raise HTTPException(404, "Product not found")
    # Build snapshot line
    item = {
        "product_id": str(prod["_id"]),
        "name": prod["name"],
        "quantity": int(body.quantity),
        "unit_price": float(prod["price"]),
        "tax_rate": float(prod.get("tax_rate", 0.0)),
        "line_total": 0.0,
    }
    sess.setdefault("items", []).append(item)

    # If product is physical, decrement stock and alert if low
    if not prod.get("is_service") and prod.get("stock") is not None:
        new_stock = int(prod["stock"]) - int(body.quantity)
        db["product"].update_one({"_id": prod["_id"]}, {"$set": {"stock": new_stock}})
        if prod.get("low_stock_threshold") is not None and new_stock <= int(prod["low_stock_threshold"]):
            log_activity("inventory_alert", f"Low stock: {prod['name']} ({new_stock} left)")

    calculate_session_totals(sess)
    db["session"].update_one({"_id": sess["_id"]}, {"$set": sess})
    return {"ok": True}


class ApplyDiscountPayload(Discount):
    pass


@app.post("/sessions/{session_id}/discount")
def apply_discount(session_id: str, body: ApplyDiscountPayload):
    sess = db["session"].find_one({"_id": oid(session_id)})
    if not sess:
        raise HTTPException(404, "Session not found")
    sess["discount"] = body.model_dump()
    calculate_session_totals(sess)
    db["session"].update_one({"_id": sess["_id"]}, {"$set": sess})
    return {"ok": True}


class CheckoutPayload(BaseModel):
    method: Literal["cash","card","external"]
    amount: Optional[float] = None
    reference: Optional[str] = None


@app.post("/sessions/{session_id}/checkout")
def checkout_session(session_id: str, body: CheckoutPayload):
    sess = db["session"].find_one({"_id": oid(session_id)})
    if not sess:
        raise HTTPException(404, "Session not found")

    # end session
    sess["ended_at"] = now_ts()
    sess["status"] = "completed"
    calculate_session_totals(sess)

    amount_to_pay = sess.get("total", 0.0) if body.amount is None else float(body.amount)

    # Simulate external gateway processing success
    payment = Payment(
        session_id=session_id,
        amount=amount_to_pay,
        method=body.method,
        status="processed",
        reference=body.reference,
    )
    pay_id = create_document("payment", payment)

    sess["payment_status"] = "paid"
    db["session"].update_one({"_id": sess["_id"]}, {"$set": sess})

    log_activity("checkout", f"Session checked out. Paid {amount_to_pay:.2f}", session_id=session_id, customer_id=sess.get("customer_id"))

    return {
        "ok": True,
        "payment_id": pay_id,
        "receipt": {
            "session_id": session_id,
            "items": sess.get("items", []),
            "subtotal": sess.get("subtotal", 0.0),
            "tax": sess.get("tax", 0.0),
            "total": sess.get("total", 0.0),
            "discount": sess.get("discount"),
            "paid": amount_to_pay,
            "method": body.method,
            "ended_at": sess.get("ended_at"),
        }
    }


@app.get("/sessions/active")
def get_active_sessions():
    docs = list(db["session"].find({"status": {"$in": ["active", "paused"]}}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.get("/activity")
def activity_feed(limit: int = 50):
    docs = list(db["activity"].find({}).sort("created_at", -1).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


# -----------------------------
# Metrics & BI (basic real-time aggregates)
# -----------------------------

@app.get("/metrics/summary")
def metrics_summary():
    now = datetime.now(timezone.utc)
    start_of_today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    start_ts = start_of_today.timestamp()

    active_count = db["session"].count_documents({"status": {"$in": ["active", "paused"]}})

    today_payments = list(db["payment"].find({"created_at": {"$gte": start_ts}}))
    today_revenue = sum(float(p.get("amount", 0)) for p in today_payments)

    today_sessions = list(db["session"].find({"started_at": {"$gte": start_ts}}))
    total_booked_duration = sum(float(s.get("elapsed_active_seconds", 0)) for s in today_sessions)

    # Yesterday comparison
    start_of_yesterday = start_of_today.timestamp() - 86400
    yesterday_payments = list(db["payment"].find({"created_at": {"$gte": start_of_yesterday, "$lt": start_ts}}))
    yesterday_revenue = sum(float(p.get("amount", 0)) for p in yesterday_payments)

    def growth(cur: float, prev: float) -> float:
        if prev == 0:
            return 100.0 if cur > 0 else 0.0
        return round(((cur - prev) / prev) * 100.0, 2)

    return {
        "active_sessions": int(active_count),
        "today_revenue": round(today_revenue, 2),
        "today_vs_yesterday": growth(today_revenue, yesterday_revenue),
        "product_sales_volume": db["payment"].count_documents({"created_at": {"$gte": start_ts}}),
        "total_booked_duration_seconds": round(total_booked_duration, 2),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
