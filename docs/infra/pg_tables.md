
## What the Seed Script Creates vs What Services Manage

```
Seed script to simulate a fictional e-commerce company (one-time):
  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌─────────┐  ┌─────────┐
  │  users   │  │ products │  │ orders │  │ billing │  │ tickets │ (initial rows)
  └──────────┘  └──────────┘  └────────┘  └─────────┘  └─────────┘

Runtime:
  ┌─────────┐  ← mcp-server writes (create_ticket, escalate, route_to_team)
  │ tickets │  ← agent-service reads (admin dashboard)
  └─────────┘
  ┌─────────┐  ← mcp-server writes wallet credits, refunds; updates orders status
  │ billing │
  │ orders  │
  └─────────┘

  ┌──────────────────┐  ← agent-service writes (admin override endpoint)
  │ human_overrides  │
  └──────────────────┘

  ┌────────────┐  ┌─────────────┐  ← auth service writes (OIDC login upserts)
  │ auth_users │  │ auth_admins │
  └────────────┘  └─────────────┘

  ┌──────────────────────────────┐
  │ langgraph_checkpoints        │  ← LangGraph auto-manages
  │ langgraph_checkpoint_writes  │
  └──────────────────────────────┘
```


## Complete Table Inventory (Updated)

| # | Table | Schema Location | Created By | Managed By | Purpose |
|---|-------|----------------|------------|------------|---------|
| 1 | `users` | `src/tests/simulate_company/seed.py` | Seed script (one‑time) | Seed script | Customer profiles (business data) |
| 2 | `products` | Same seed script | Seed script | Seed script | Product catalog with return/warranty rules |
| 3 | `orders` | Same seed script | Seed script | Seed script | Customer orders; status transitions by `mcp-server` tools |
| 4 | `billing` | Same seed script | Seed script | Seed script; `mcp-server` writes wallet credits & refunds | Payment, refund, and wallet credit transactions |
| 5 | `tickets` | `mcp-server/src/db.py` + seed script pre‑populates | Seed script creates table; `mcp-server` writes rows | `mcp-server` (create, escalate, route); `agent-service` reads | Support tickets with classification and routing info |
| 6 | `human_overrides` | `agent-service/src/db.py` | Alembic migration `001_initial.py` | `agent-service` (admin override endpoint) | Feedback records for DSPy retraining |
| 7 | `auth_users` | `auth/src/db.py` | Alembic migration | `auth` service (OIDC login upserts) | Public chat users authenticated via OIDC |
| 8 | `auth_admins` | `auth/src/db.py` | Alembic migration | `auth` service (OIDC login upserts) | Internal admins with restricted domain access |
| 9 | `langgraph_checkpoints` | Auto | `PostgresSaver.setup()` | LangGraph internally | Graph state snapshots for resumability |
| 10 | `langgraph_checkpoint_writes` | Auto | `PostgresSaver.setup()` | LangGraph internally | Pending writes for fault tolerance |

---

## You Do NOT Manage Tables 9 and 10

LangGraph creates these automatically when you call:

```python
checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url)
await checkpointer.setup()  # ← creates both tables if they don't exist
```

You never write SQL for them. You never query them directly. LangGraph handles all reads/writes internally.

---

## Table Creation Summary

| Creator | Tables | When |
|---------|--------|------|
| **Seed script** (`simulate_company/seed.py`) | `users`, `products`, `orders`, `billing`, `tickets` (schema + initial rows) | One‑time, before system starts |
| **mcp-server** | Reads/writes `tickets`, writes `billing` rows (wallet credits, refunds), updates `orders` statuses | Runtime (create_ticket, escalate, route_to_team, issue_wallet_credit, schedule_return_pickup) |
| **agent-service** (Alembic) | `human_overrides` | Migration during deployment |
| **auth** (Alembic) | `auth_users`, `auth_admins` | Migration during deployment |
| **LangGraph** (`PostgresSaver`) | `langgraph_checkpoints`, `langgraph_checkpoint_writes` | Auto‑created on first `checkpointer.setup()` |

---

## Exact Schema for Each Table (with Indexes & Optimization)

### `users`
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20) NOT NULL,
    language_pref VARCHAR(5) DEFAULT 'en',
    segment VARCHAR(50) DEFAULT 'new',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_users_email ON users (email);          -- lookup_customer by email
CREATE INDEX idx_users_phone ON users (phone);          -- lookup_customer by phone
CREATE INDEX idx_users_segment ON users (segment);      -- analytics/dashboard filters
```

### `products`
```sql
CREATE TABLE products (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    price NUMERIC(10,2) NOT NULL,
    return_window_days INTEGER DEFAULT 10,
    warranty_months INTEGER DEFAULT 12,
    is_returnable BOOLEAN DEFAULT TRUE,
    is_express_eligible BOOLEAN DEFAULT FALSE,
    stock_quantity INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX idx_products_category ON products (category);  -- product filters
```

### `orders`
```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    product_id UUID REFERENCES products(id),
    status VARCHAR(50) DEFAULT 'placed'
        CHECK (status IN ('placed','shipped','delivered','cancelled','returned','return_initiated')),
    quantity INTEGER DEFAULT 1,
    amount NUMERIC(10,2) NOT NULL,
    discount_amount NUMERIC(10,2) DEFAULT 0,
    shipping_amount NUMERIC(10,2) DEFAULT 0,
    cod_fee NUMERIC(10,2) DEFAULT 0,
    payment_method VARCHAR(50) NOT NULL,
    shipping_address JSONB,
    pincode VARCHAR(10),
    city VARCHAR(100),
    order_date TIMESTAMPTZ DEFAULT NOW(),
    delivery_date TIMESTAMPTZ,
    promised_delivery_date TIMESTAMPTZ,
    is_delayed BOOLEAN DEFAULT FALSE,
    delivery_attempts INTEGER DEFAULT 0,
    tracking_number VARCHAR(100),
    notes TEXT
);

-- Indexes
CREATE INDEX idx_orders_user_id ON orders (user_id);                 -- get_recent_orders
CREATE INDEX idx_orders_status ON orders (status);                   -- admin dashboard filters
CREATE INDEX idx_orders_order_date ON orders (order_date DESC);      -- sort by recency
CREATE INDEX idx_orders_pincode ON orders (pincode);                 -- delivery serviceability
```

### `billing`
```sql
CREATE TABLE billing (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    user_id UUID REFERENCES users(id),
    transaction_type VARCHAR(50) NOT NULL
        CHECK (transaction_type IN ('payment','refund','wallet_credit')),
    amount NUMERIC(10,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending'
        CHECK (status IN ('pending','completed','failed')),
    refund_eligible BOOLEAN DEFAULT FALSE,
    refund_reason TEXT,
    payment_gateway VARCHAR(100),
    gateway_transaction_id VARCHAR(255),
    transaction_date TIMESTAMPTZ DEFAULT NOW(),
    completed_date TIMESTAMPTZ
);

-- Indexes
CREATE INDEX idx_billing_order_id ON billing (order_id);                     -- check_refund_eligibility
CREATE INDEX idx_billing_user_id ON billing (user_id);                       -- user billing history
CREATE INDEX idx_billing_transaction_type ON billing (transaction_type);     -- filter by type
CREATE INDEX idx_billing_status ON billing (status);                         -- pending refunds
```

### `tickets`
```sql
CREATE TABLE tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    order_id UUID REFERENCES orders(id),
    query_text TEXT NOT NULL,
    classification JSONB,
    resolution_type VARCHAR(50),
    status VARCHAR(50) DEFAULT 'open'
        CHECK (status IN ('open','pending_human','resolved','closed')),
    priority VARCHAR(50)
        CHECK (priority IN ('critical','high','medium','low')),
    assigned_team VARCHAR(100)
        CHECK (assigned_team IN ('order_fulfillment','payments','logistics','service_center','senior_support','general_support')),
    assigned_agent VARCHAR(255),
    resolution_summary TEXT,
    source VARCHAR(50) DEFAULT 'chat',
    language VARCHAR(5) DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_tickets_user_id ON tickets (user_id);
CREATE INDEX idx_tickets_order_id ON tickets (order_id);
CREATE INDEX idx_tickets_status ON tickets (status);
CREATE INDEX idx_tickets_priority ON tickets (priority);
CREATE INDEX idx_tickets_assigned_team ON tickets (assigned_team);
CREATE INDEX idx_tickets_created_at ON tickets (created_at DESC);

-- Partial index for unresolved tickets (most queries)
CREATE INDEX idx_tickets_open ON tickets (created_at DESC)
    WHERE status IN ('open','pending_human');

-- GIN index for querying inside classification JSON
CREATE INDEX idx_tickets_classification ON tickets USING GIN (classification);
```

### `human_overrides`
```sql
CREATE TABLE human_overrides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticket_id UUID REFERENCES tickets(id),
    original_classification JSONB NOT NULL,
    corrected_classification JSONB NOT NULL,
    reason TEXT,
    overridden_by VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_human_overrides_ticket_id ON human_overrides (ticket_id);
CREATE INDEX idx_human_overrides_created_at ON human_overrides (created_at DESC);
```

### `auth_users`
```sql
CREATE TABLE auth_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    oidc_sub VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ DEFAULT NOW()
);

-- UNIQUE constraints automatically create indexes on email and oidc_sub.
```

### `auth_admins`
```sql
CREATE TABLE auth_admins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    oidc_sub VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    domain VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'admin',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ DEFAULT NOW()
);

-- UNIQUE constraints automatically create indexes on email and oidc_sub.
```

---

## Optimization & Production Notes

1. **CHECK constraints** on `orders.status`, `billing.transaction_type`, `billing.status`, `tickets.status`, `tickets.priority`, `tickets.assigned_team` prevent invalid data at the database level.
2. **Partial index** `idx_tickets_open` covers the 90% of dashboard queries that only care about unresolved tickets — smaller and faster than a full index.
3. **GIN index** on `tickets.classification` enables queries like `WHERE classification @> '{"intent":"refund_status"}'`.
4. **LangGraph checkpoint cleanup** – the `langgraph_checkpoints` table grows unboundedly. In production, schedule a periodic cleanup with `pg_cron`:
   ```sql
   SELECT cron.schedule('cleanup-checkpoints', '0 3 * * *',
     $$DELETE FROM langgraph_checkpoints WHERE created_at < NOW() - INTERVAL '30 days'$$
   );
   ```
5. **`pg_stat_statements`** – enable this extension to track slow queries in production.

---


Each table has exactly one writer. Clean boundaries, minimal overlap. The `billing` and `orders` tables are now also written by `mcp-server` for the new agent actions (wallet credits, return scheduling). The seed script still sets up the initial schema and demo data.