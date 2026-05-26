## Complete Table Inventory

| # | Table | Schema Location | Created By | Managed By |
|---|-------|----------------|------------|------------|
| 1 | `users` | `src/tests/simulate_company/seed.py` | Seed script (one-time) | Seed script |
| 2 | `products` | Same seed script | Seed script | Seed script |
| 3 | `orders` | Same seed script | Seed script | Seed script |
| 4 | `billing` | Same seed script | Seed script | Seed script |
| 5 | `tickets` | `mcp-server/src/db.py` + seed script pre-populates | Seed script creates table, `mcp-server` writes rows | `mcp-server` (create_ticket, escalate, route_to_team) |
| 6 | `human_overrides` | `agent-service/src/db.py` | Alembic migration `001_initial.py` | `agent-service` (admin override endpoint) |
| 7 | `auth_users` | `auth/src/db.py` | Alembic migration | `auth` service (OIDC login upserts) |
| 8 | `auth_admins` | `auth/src/db.py` | Alembic migration | `auth` service (OIDC login upserts) |
| 9 | `langgraph_checkpoints` | Auto | `PostgresSaver.setup()` | LangGraph internally |
| 10 | `langgraph_checkpoint_writes` | Auto | `PostgresSaver.setup()` | LangGraph internally |

---

## You Do NOT Manage Tables 9 and 10

LangGraph creates these automatically when you call:

```python
checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url)
await checkpointer.setup()  # ← creates both tables if they don't exist
```

You never write SQL for these. You never query them directly. LangGraph handles all reads/writes internally. They store graph state snapshots at every node boundary for resumability.

---

## Table Creation Summary

| Creator | Tables | When |
|---------|--------|------|
| **Seed script** (`simulate_company/seed.py`) | `users`, `products`, `orders`, `billing`, `tickets` (schema + initial rows) | One-time, before system starts |
| **mcp-server** | Reads/writes `tickets` rows | Runtime (create_ticket, escalate, route_to_team) |
| **agent-service** (Alembic) | `human_overrides` | Migration during deployment |
| **auth** (Alembic) | `auth_users`, `auth_admins` | Migration during deployment |
| **LangGraph** (`PostgresSaver`) | `langgraph_checkpoints`, `langgraph_checkpoint_writes` | Auto-created on first `checkpointer.setup()` |

---

## Exact Schema for Each Table

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
```

### `orders`
```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    product_id UUID REFERENCES products(id),
    status VARCHAR(50) DEFAULT 'placed',
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
```

### `billing`
```sql
CREATE TABLE billing (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    user_id UUID REFERENCES users(id),
    transaction_type VARCHAR(50) NOT NULL,  -- 'payment' or 'refund'
    amount NUMERIC(10,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    refund_eligible BOOLEAN DEFAULT FALSE,
    refund_reason TEXT,
    payment_gateway VARCHAR(100),
    gateway_transaction_id VARCHAR(255),
    transaction_date TIMESTAMPTZ DEFAULT NOW(),
    completed_date TIMESTAMPTZ
);
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
    status VARCHAR(50) DEFAULT 'open',
    priority VARCHAR(50),
    assigned_team VARCHAR(100),
    assigned_agent VARCHAR(255),
    resolution_summary TEXT,
    source VARCHAR(50) DEFAULT 'chat',
    language VARCHAR(5) DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
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
```

### `langgraph_checkpoints` + `langgraph_checkpoint_writes`
Created automatically by `PostgresSaver.setup()`. You never touch these.

---

## What the Seed Script Creates vs What Services Manage

```
Seed script (one-time):
  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌─────────┐  ┌─────────┐
  │  users   │  │ products │  │ orders │  │ billing │  │ tickets │ (initial rows)
  └──────────┘  └──────────┘  └────────┘  └─────────┘  └─────────┘

Runtime:
  ┌─────────┐  ← mcp-server writes (create_ticket, escalate, route_to_team)
  │ tickets │  ← agent-service reads (admin dashboard)
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

No overlap. Each table has exactly one writer. Clear ownership boundaries.