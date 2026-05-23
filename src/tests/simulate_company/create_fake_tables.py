"""
seed_realistic.py
------------------
Idempotent database seeder for Kestral ticket-triage system.
- Drops all tables on every run
- Loads data from deterministic JSON files
- Validates all foreign keys before insertion
- Prints final schema, sample rows, foreign keys, and indexes

Data files expected in same directory:
    users.json, products.json, orders.json, billing.json,
    tickets.json (optional), seed_tickets_dspy.json

Usage:
    kubectl port-forward -n default svc/postgres-pooler 5432:5432 &
    python seed_realistic.py
"""

import asyncio
import base64
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, Numeric,
    String, Text, Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
NAMESPACE = "default"
SECRET_NAME = "postgres-cluster-app"
POOLER_HOST = "localhost"
POOLER_PORT = 5432

JSON_FILES = {
    "users":      SCRIPT_DIR / "users.json",
    "products":   SCRIPT_DIR / "products.json",
    "orders":     SCRIPT_DIR / "orders.json",
    "billing":    SCRIPT_DIR / "billing.json",
    "tickets":    SCRIPT_DIR / "tickets.json",
    "dspy_seed":  SCRIPT_DIR / "seed_tickets_dspy.json",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def kubectl(args: List[str]) -> str:
    try:
        r = subprocess.run(["kubectl"] + args, capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ kubectl failed: {' '.join(['kubectl'] + args)}\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("❌ kubectl not found", file=sys.stderr)
        sys.exit(1)

def fetch_credentials() -> Dict[str, str]:
    print("🔐 Fetching credentials from Kubernetes secret...")
    def _b64(key: str) -> str:
        raw = kubectl(["get", "secret", SECRET_NAME, "-n", NAMESPACE,
                        "-o", f"jsonpath={{.data.{key}}}"])
        return base64.b64decode(raw).decode()
    return {"username": _b64("username"), "password": _b64("password"), "dbname": _b64("dbname")}

def db_url(creds: Dict[str, str]) -> str:
    return (f"postgresql+asyncpg://{creds['username']}:{creds['password']}"
            f"@{POOLER_HOST}:{POOLER_PORT}/{creds['dbname']}")

def load_json(path: Path) -> list:
    if not path.exists():
        print(f"⚠️  {path.name} not found – skipping.")
        return []
    with open(path) as f:
        return json.load(f)

def safe_uuid(raw: Any) -> uuid.UUID:
    """Parse UUID from string, bytes, or already a UUID."""
    if isinstance(raw, uuid.UUID):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode()
    raw = str(raw).strip()
    return uuid.UUID(raw)

def parse_dt(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def parse_dec(val: Any) -> Decimal:
    return Decimal(str(val))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20), nullable=False)
    language_pref = Column(String(5), nullable=False, default="en")
    segment = Column(String(20), nullable=False, default="new")
    created_at = Column(DateTime(timezone=True), nullable=False)
    orders = relationship("Order", back_populates="user")
    tickets = relationship("Ticket", back_populates="user")
    __table_args__ = (Index("idx_users_segment", "segment"),)

class Product(Base):
    __tablename__ = "products"
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False)
    subcategory = Column(String(100))
    price = Column(Numeric(10,2), nullable=False)
    return_window_days = Column(Integer, nullable=False, default=10)
    warranty_months = Column(Integer, nullable=False, default=12)
    is_returnable = Column(Boolean, nullable=False, default=True)
    is_express_eligible = Column(Boolean, nullable=False, default=False)
    stock_quantity = Column(Integer, nullable=False, default=100)
    orders = relationship("Order", back_populates="product")
    __table_args__ = (Index("idx_products_category", "category"),)

class Order(Base):
    __tablename__ = "orders"
    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    status = Column(String(20), nullable=False, default="placed")
    quantity = Column(Integer, nullable=False, default=1)
    amount = Column(Numeric(10,2), nullable=False)
    discount_amount = Column(Numeric(10,2), default=0)
    shipping_amount = Column(Numeric(10,2), default=0)
    cod_fee = Column(Numeric(10,2), default=0)
    payment_method = Column(String(20), nullable=False)
    shipping_address = Column(JSONB, nullable=False)
    pincode = Column(String(10), nullable=False, index=True)
    city = Column(String(100), nullable=False)
    order_date = Column(DateTime(timezone=True), nullable=False)
    delivery_date = Column(DateTime(timezone=True))
    promised_delivery_date = Column(DateTime(timezone=True))
    is_delayed = Column(Boolean, default=False)
    delivery_attempts = Column(Integer, default=0)
    tracking_number = Column(String(50))
    notes = Column(Text)
    user = relationship("User", back_populates="orders")
    product = relationship("Product", back_populates="orders")
    billing = relationship("Billing", back_populates="order")
    __table_args__ = (
        Index("idx_orders_status", "status"),
        Index("idx_orders_pincode", "pincode"),
        Index("idx_orders_order_date", "order_date"),
    )

class Billing(Base):
    __tablename__ = "billing"
    id = Column(UUID(as_uuid=True), primary_key=True)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    transaction_type = Column(String(20), nullable=False)
    amount = Column(Numeric(10,2), nullable=False)
    status = Column(String(20), default="pending")
    refund_eligible = Column(Boolean, default=False)
    refund_reason = Column(String(50))
    payment_gateway = Column(String(50))
    gateway_transaction_id = Column(String(100))
    transaction_date = Column(DateTime(timezone=True), nullable=False)
    completed_date = Column(DateTime(timezone=True))
    order = relationship("Order", back_populates="billing")
    __table_args__ = (
        Index("idx_billing_status", "status"),
        Index("idx_billing_transaction_type", "transaction_type"),
    )

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    order_id = Column(UUID(as_uuid=True), ForeignKey("orders.id"), nullable=True, index=True)
    query_text = Column(Text, nullable=False)
    classification = Column(JSONB)
    resolution_type = Column(String(20))
    status = Column(String(20), nullable=False, default="open")
    priority = Column(String(10), default="medium")
    assigned_agent = Column(String(100))
    resolution_summary = Column(Text)
    source = Column(String(20), default="chat")
    language = Column(String(5), default="en")
    created_at = Column(DateTime(timezone=True), nullable=False)
    resolved_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), nullable=False)
    user = relationship("User", back_populates="tickets")
    overrides = relationship("HumanOverride", back_populates="ticket")
    __table_args__ = (
        Index("idx_tickets_status", "status"),
        Index("idx_tickets_priority", "priority"),
        Index("idx_tickets_created_at", "created_at"),
    )

class HumanOverride(Base):
    __tablename__ = "human_overrides"
    id = Column(UUID(as_uuid=True), primary_key=True)
    ticket_id = Column(UUID(as_uuid=True), ForeignKey("tickets.id"), nullable=False, index=True)
    original_classification = Column(JSONB, nullable=False)
    corrected_classification = Column(JSONB, nullable=False)
    reason = Column(Text)
    overridden_by = Column(String(100))
    created_at = Column(DateTime(timezone=True), nullable=False)
    ticket = relationship("Ticket", back_populates="overrides")

# ---------------------------------------------------------------------------
# Seed logic
# ---------------------------------------------------------------------------

async def seed(engine, session_factory):
    # ── drop & recreate ──────────────────────────────────────────────
    print("🗑️  Dropping all tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("🏗️  Creating fresh tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Schema ready.\n")

    # ── load JSON ────────────────────────────────────────────────────
    print("📂 Loading JSON files...")
    users_j    = load_json(JSON_FILES["users"])
    products_j = load_json(JSON_FILES["products"])
    orders_j   = load_json(JSON_FILES["orders"])
    billing_j  = load_json(JSON_FILES["billing"])
    tickets_j  = load_json(JSON_FILES["tickets"])
    dspy_j     = load_json(JSON_FILES["dspy_seed"])
    print(f"   users={len(users_j)}  products={len(products_j)}  orders={len(orders_j)}")
    print(f"   billing={len(billing_j)}  tickets={len(tickets_j)}  dspy_examples={len(dspy_j)}\n")

    async with session_factory() as session:
        # ── Users ────────────────────────────────────────────────────
        print("👤 Inserting users...")
        user_ids = set()
        for u in users_j:
            uid = safe_uuid(u["id"])
            session.add(User(
                id=uid, full_name=u["full_name"], email=u["email"],
                phone=u["phone"], language_pref=u.get("language_pref","en"),
                segment=u.get("segment","new"), created_at=parse_dt(u["created_at"]),
            ))
            user_ids.add(uid)
        await session.flush()
        print(f"   ✅ {len(user_ids)} users")

        # ── Products ─────────────────────────────────────────────────
        print("📦 Inserting products...")
        prod_ids = set()
        for p in products_j:
            pid = safe_uuid(p["id"])
            session.add(Product(
                id=pid, name=p["name"], category=p["category"],
                subcategory=p.get("subcategory"), price=parse_dec(p["price"]),
                return_window_days=p.get("return_window_days",10),
                warranty_months=p.get("warranty_months",12),
                is_returnable=p.get("is_returnable",True),
                is_express_eligible=p.get("is_express_eligible",False),
                stock_quantity=p.get("stock_quantity",100),
            ))
            prod_ids.add(pid)
        await session.flush()
        print(f"   ✅ {len(prod_ids)} products")

        # ── Orders ───────────────────────────────────────────────────
        print("🛒 Inserting orders...")
        order_ids = set()
        for o in orders_j:
            oid, uid, pid = safe_uuid(o["id"]), safe_uuid(o["user_id"]), safe_uuid(o["product_id"])
            if uid not in user_ids or pid not in prod_ids:
                print(f"   ⚠️  skipping order {oid} (bad FK)")
                continue
            session.add(Order(
                id=oid, user_id=uid, product_id=pid,
                status=o.get("status","placed"), quantity=o.get("quantity",1),
                amount=parse_dec(o["amount"]),
                discount_amount=parse_dec(o.get("discount_amount",0)),
                shipping_amount=parse_dec(o.get("shipping_amount",0)),
                cod_fee=parse_dec(o.get("cod_fee",0)),
                payment_method=o["payment_method"],
                shipping_address=o["shipping_address"],
                pincode=o.get("pincode","000000"), city=o.get("city","Unknown"),
                order_date=parse_dt(o["order_date"]),
                delivery_date=parse_dt(o.get("delivery_date")),
                promised_delivery_date=parse_dt(o.get("promised_delivery_date")),
                is_delayed=o.get("is_delayed",False),
                delivery_attempts=o.get("delivery_attempts",0),
                tracking_number=o.get("tracking_number"),
                notes=o.get("notes"),
            ))
            order_ids.add(oid)
        await session.flush()
        print(f"   ✅ {len(order_ids)} orders")

        # ── Billing ──────────────────────────────────────────────────
        print("💳 Inserting billing...")
        bill_count = 0
        for b in billing_j:
            oid, uid = safe_uuid(b["order_id"]), safe_uuid(b["user_id"])
            if oid not in order_ids or uid not in user_ids:
                print(f"   ⚠️  skipping billing {b['id']} (bad FK)")
                continue
            session.add(Billing(
                id=safe_uuid(b["id"]), order_id=oid, user_id=uid,
                transaction_type=b["transaction_type"],
                amount=parse_dec(b["amount"]),
                status=b.get("status","pending"),
                refund_eligible=b.get("refund_eligible",False),
                refund_reason=b.get("refund_reason"),
                payment_gateway=b.get("payment_gateway"),
                gateway_transaction_id=b.get("gateway_transaction_id"),
                transaction_date=parse_dt(b["transaction_date"]),
                completed_date=parse_dt(b.get("completed_date")),
            ))
            bill_count += 1
        await session.flush()
        print(f"   ✅ {bill_count} billing rows")

        # ── Tickets (pre-existing) ───────────────────────────────────
        ticket_ids = set()
        if tickets_j:
            print("🎫 Inserting tickets...")
            for t in tickets_j:
                tid = safe_uuid(t["id"])
                uid = safe_uuid(t["user_id"])
                oid = safe_uuid(t["order_id"]) if t.get("order_id") else None
                if uid not in user_ids or (oid and oid not in order_ids):
                    print(f"   ⚠️  skipping ticket {tid} (bad FK)")
                    continue
                session.add(Ticket(
                    id=tid, user_id=uid, order_id=oid,
                    query_text=t["query_text"],
                    classification=t.get("classification"),
                    resolution_type=t.get("resolution_type"),
                    status=t.get("status","open"),
                    priority=t.get("priority","medium"),
                    assigned_agent=t.get("assigned_agent"),
                    resolution_summary=t.get("resolution_summary"),
                    source=t.get("source","chat"),
                    language=t.get("language","en"),
                    created_at=parse_dt(t["created_at"]),
                    resolved_at=parse_dt(t.get("resolved_at")),
                    updated_at=parse_dt(t.get("updated_at")),
                ))
                ticket_ids.add(tid)
            await session.flush()
            print(f"   ✅ {len(ticket_ids)} tickets")
        else:
            print("ℹ️  No tickets.json – skipping")

        # ── DSPy seed (validated, not inserted) ──────────────────────
        valid_intents = {
            "return_request","refund_status","delayed_delivery",
            "wrong_item_delivered","damaged_product","cancellation_request",
            "warranty_claim","defective_product","escalation_request",
            "delivery_issue","order_status","payment_issue",
        }
        dspy_ok = 0
        for d in dspy_j:
            if all(k in d for k in ("query","intent","urgency","sentiment","auto_resolvable")):
                if d["intent"] in valid_intents:
                    dspy_ok += 1
        print(f"🧠 {dspy_ok}/{len(dspy_j)} DSPy examples valid (not inserted)\n")

        # ── Commit ───────────────────────────────────────────────────
        print("💾 Committing...")
        await session.commit()

    # ── Print final schema & samples ─────────────────────────────────
    await print_schema_and_samples(engine)

    print("\n" + "=" * 60)
    print("🎉 Seed complete!")
    print(f"   users={len(user_ids)}  products={len(prod_ids)}  orders={len(order_ids)}")
    print(f"   billing={bill_count}  tickets={len(ticket_ids)}  dspy_examples={dspy_ok}")
    print("=" * 60)

# ---------------------------------------------------------------------------
# Schema printer (runs after seed, uses raw asyncpg for simplicity)
# ---------------------------------------------------------------------------

async def print_schema_and_samples(engine):
    """Print schema, first row, foreign keys, and indexes for every table."""
    # Get a raw asyncpg connection from the SQLAlchemy engine
    async with engine.connect() as conn:
        raw_conn = await conn.get_raw_connection()
        pg_conn: Any = raw_conn.driver_connection  # the underlying asyncpg.Connection

        tables = ["users", "products", "orders", "billing", "tickets", "human_overrides"]

        for tbl in tables:
            # Columns
            cols = await pg_conn.fetch(
                """SELECT column_name, data_type, is_nullable
                   FROM information_schema.columns
                   WHERE table_name = $1
                   ORDER BY ordinal_position""", tbl)
            print(f"\n=== {tbl.upper()} SCHEMA ===")
            for c in cols:
                print(f"  {c['column_name']:25s} {c['data_type']:20s} nullable={c['is_nullable']}")

            # First row
            try:
                row = await pg_conn.fetchrow(f"SELECT * FROM {tbl} LIMIT 1")
                print(f"\n  First row:")
                if row:
                    print(f"  {dict(row)}")
                else:
                    print("  (empty)")
            except Exception:
                print("  (table empty or inaccessible)")

        # Foreign keys
        fks = await pg_conn.fetch("""
            SELECT tc.table_name, kcu.column_name,
                   ccu.table_name AS foreign_table, ccu.column_name AS foreign_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            ORDER BY tc.table_name
        """)
        print(f"\n=== FOREIGN KEYS ===")
        for fk in fks:
            print(f"  {fk['table_name']}.{fk['column_name']} -> {fk['foreign_table']}.{fk['foreign_column']}")

        # Indexes
        idxs = await pg_conn.fetch("""
            SELECT tablename, indexname FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname
        """)
        print(f"\n=== INDEXES ===")
        for ix in idxs:
            print(f"  {ix['tablename']:20s} {ix['indexname']}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("KESTRAL E-COMMERCE — REALISTIC JSON SEEDER")
    print("=" * 60)

    creds = fetch_credentials()
    print(f"🔗 Database: {creds['dbname']} (user: {creds['username']})")
    print(f"🔗 Pooler:   {POOLER_HOST}:{POOLER_PORT}\n")

    engine = create_async_engine(db_url(creds), echo=False,
                                  pool_size=5, max_overflow=2, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        print("✅ Database connection successful.\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    await seed(engine, session_factory)
    await engine.dispose()
    print("\n✨ Kestral database ready.\n")

if __name__ == "__main__":
    asyncio.run(main())