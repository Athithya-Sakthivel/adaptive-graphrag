# MCP Server — Kestral Ticket Triage System

The **mcp‑server** exposes 10 tools over MCP using FastMCP 3.3.1 and HTTP transport. The agent‑service calls these tools to look up customers, retrieve orders, search policies, issue compensation, schedule returns, create tickets, escalate, and route to human teams.

---

## Tools

### 1. `lookup_customer`

Find a customer by their email address or phone number.

**Input**
```json
{
  "email": "priya.sharma@email.com"
}
```

**Output**
```json
{
  "result": {
    "id": "a1b2c3d4-e5f6-4a7b-8c9d-000000000001",
    "full_name": "Priya Sharma",
    "email": "priya.sharma@email.com",
    "phone": "+919876543210",
    "language_pref": "en",
    "segment": "premium",
    "created_at": "2025-01-15T10:30:00Z"
  }
}
```

Returns `null` / empty if no customer matches the given email or phone.

---

### 2. `get_recent_orders`

Return the 5 most recent orders for a customer, newest first.

**Input**
```json
{
  "user_id": "a1b2c3d4-e5f6-4a7b-8c9d-000000000001"
}
```

**Output** (truncated – up to 5 orders)
```json
{
  "result": [
    {
      "id": "c3d4e5f6-a7b8-4c9d-0e1f-000000000021",
      "user_id": "a1b2c3d4-e5f6-4a7b-8c9d-000000000001",
      "product_id": "b2c3d4e5-f6a7-4b8c-9d0e-000000000008",
      "status": "cancelled",
      "quantity": 1,
      "amount": "2499.00",
      "discount_amount": "0.00",
      "shipping_amount": "0.00",
      "cod_fee": "0.00",
      "payment_method": "upi",
      "shipping_address": "{\"full_name\":\"Priya Sharma\",\"address_line1\":\"42, MG Road\",\"city\":\"Bengaluru\",\"state\":\"Karnataka\",\"pincode\":\"560001\",\"phone\":\"+919876543210\"}",
      "pincode": "560001",
      "city": "Bengaluru",
      "order_date": "2026-05-20T09:00:00Z",
      "delivery_date": null,
      "promised_delivery_date": "2026-05-24T18:00:00Z",
      "is_delayed": false,
      "delivery_attempts": 0,
      "tracking_number": null,
      "notes": "Premium customer cancelled before shipment. REFUND STATUS inquiry expected."
    },
    {
      "id": "c3d4e5f6-a7b8-4c9d-0e1f-000000000001",
      "status": "delivered",
      "amount": "124999.00",
      "payment_method": "upi",
      "delivery_date": "2026-05-14T14:00:00Z",
      "tracking_number": "KST-BLR-001"
    }
  ]
}
```

Returns an empty list `[]` if the customer has no orders.

---

### 3. `get_order_details`

Full order information including the product it contains (joined from the products table).

**Input**
```json
{
  "order_id": "c3d4e5f6-a7b8-4c9d-0e1f-000000000001"
}
```

**Output**
```json
{
  "result": {
    "id": "c3d4e5f6-a7b8-4c9d-0e1f-000000000001",
    "user_id": "a1b2c3d4-e5f6-4a7b-8c9d-000000000001",
    "product_id": "b2c3d4e5-f6a7-4b8c-9d0e-000000000001",
    "status": "delivered",
    "quantity": 1,
    "amount": "124999.00",
    "discount_amount": "0.00",
    "shipping_amount": "0.00",
    "cod_fee": "0.00",
    "payment_method": "upi",
    "shipping_address": "{\"full_name\":\"Priya Sharma\",\"address_line1\":\"42, MG Road\",\"city\":\"Bengaluru\",\"state\":\"Karnataka\",\"pincode\":\"560001\",\"phone\":\"+919876543210\"}",
    "pincode": "560001",
    "city": "Bengaluru",
    "order_date": "2026-05-10T10:00:00Z",
    "delivery_date": "2026-05-14T14:00:00Z",
    "promised_delivery_date": "2026-05-14T18:00:00Z",
    "is_delayed": false,
    "delivery_attempts": 1,
    "tracking_number": "KST-BLR-001",
    "notes": "Premium customer. Phone delivered on time. No issues.",
    "product_name": "Samsung Galaxy S25 Ultra 5G",
    "category": "electronics",
    "subcategory": "mobile_phones",
    "price": "124999.00",
    "return_window_days": 10,
    "warranty_months": 12,
    "is_returnable": true,
    "is_express_eligible": true
  }
}
```

Returns `null` / empty if the order is not found.

---

### 4. `check_refund_eligibility`

Determine whether an order can be refunded, and if so, for how much and via which method.  
Examines the return window, product category, and existing billing rows (refunds already processed).

**Input**
```json
{
  "order_id": "c3d4e5f6-a7b8-4c9d-0e1f-000000000004"
}
```

**Output (ineligible)**
```json
{
  "eligible": false,
  "reason": "return_window_expired (10 days)"
}
```

**Output (eligible)**
```json
{
  "eligible": true,
  "reason": "within_return_window",
  "amount": "2499.00",
  "method": "upi"
}
```

Possible `reason` values: `order_not_found`, `already_refunded`, `return_window_expired (N days)`, `no_payment_found`, `category_not_returnable`, `within_return_window`.

---

### 5. `search_policies`

Semantic search over the full set of company policy documents (returns, refunds, delivery, warranty, payments, grievance).  
The query is embedded by the dense service, then searched against Qdrant.

**Input**
```json
{
  "query": "return policy for damaged phone",
  "top_k": 3
}
```

**Output**
```json
{
  "result": [
    {
      "text": "- Product must be returned in original, sealed packaging.\n- If the seal is broken, return is not eligible unless the product is Dead on Arrival (DOA).\n- For DOA products, customer must record an unboxing video showing the defect.\n- Serial number must match the invoice.\n- All accessories (charger, cable, manuals, SIM ejector tool) must be included.",
      "score": 0.76642865,
      "metadata": {
        "policy_name": "Returns & Refunds Policy",
        "section_title": "2.1 Electronics & Mobile Phones",
        "heading_path": "Returns & Refunds Policy > 2. Return Conditions by Category > 2.1 Electronics & Mobile Phones",
        "tags": ["damaged_items", "refunds"]
      }
    },
    {
      "text": "- Available for mobile phones, tablets, and laptops only.\n- Must be purchased within 7 days of delivery.\n- Covers:\n  - Screen damage (cracked, shattered, display issues).\n  - Liquid damage (spills, submersion).\n  - Drops and impact damage.\n- Limit: 2 claims per 12-month period.\n- Service fee per claim: Rs. 499 for phones, Rs. 999 for laptops.",
      "score": 0.7389772,
      "metadata": {
        "policy_name": "Warranty & Service Policy",
        "section_title": "4. Accidental Damage Protection (ADP)",
        "tags": ["damaged_items"]
      }
    }
  ]
}
```

Returns an empty list `[]` if no policies match the query.

---

### 6. `issue_wallet_credit`

Add store credit to a customer’s wallet (e.g. delivery delay compensation or goodwill gesture).  
**Hard limit:** amounts greater than Rs.500 are rejected.

**Input**
```json
{
  "user_id": "a1b2c3d4-e5f6-4a7b-8c9d-000000000001",
  "amount": 100.0,
  "reason": "Battle test compensation"
}
```

**Output (accepted)**
```json
{
  "status": "issued",
  "transaction_id": "WC-87e27c9c",
  "amount": 100.0
}
```

**Output (rejected)**
```json
{
  "status": "rejected",
  "reason": "Amount exceeds maximum of Rs.500"
}
```

---

### 7. `schedule_return_pickup`

Schedule a return pickup for an order.  
**Guardrail:** internally calls `check_refund_eligibility` first — the pickup is only scheduled if the order is eligible.

**Input**
```json
{
  "order_id": "c3d4e5f6-a7b8-4c9d-0e1f-000000000001",
  "pickup_date": "2026-06-01"
}
```

**Output (scheduled)**
```json
{
  "status": "scheduled",
  "order_id": "c3d4e5f6-a7b8-4c9d-0e1f-000000000001",
  "pickup_date": "2026-06-01"
}
```

**Output (rejected — return window expired)**
```json
{
  "status": "failed",
  "reason": "return_window_expired (10 days)"
}
```

Side effect: updates the order status to `return_initiated` and appends the pickup date to the order notes.

---

### 8. `create_ticket`

Create a new support ticket in the database. The ticket is created with status `pending_human`, resolution type `escalated`, and can optionally be assigned to a team.

**Input**
```json
{
  "user_id": "a1b2c3d4-e5f6-4a7b-8c9d-000000000001",
  "query_text": "Test ticket from battle test",
  "classification": "{\"intent\":\"test\",\"urgency\":5,\"sentiment\":\"neutral\",\"auto_resolvable\":true}",
  "priority": "medium",
  "assigned_team": "general_support"
}
```

**Output**
```json
{
  "result": "c933e52a-5415-4320-9bbe-6edbf677ae4a"
}
```

Returns the UUID of the newly created ticket.

---

### 9. `escalate_to_human`

Flag an existing ticket for immediate human attention.

**Input**
```json
{
  "ticket_id": "e5f6a7b8-c9d0-4e1f-2a3b-000000000001"
}
```

**Output**
```json
{
  "status": "escalated",
  "ticket_id": "e5f6a7b8-c9d0-4e1f-2a3b-000000000001"
}
```

Side effects:
- Sets ticket `status` to `pending_human`
- Sets ticket `priority` to `critical`

---

### 10. `route_to_team`

Assign a ticket to a specific team queue. Supported teams: `order_fulfillment`, `payments`, `logistics`, `service_center`, `senior_support`, `general_support`.

**Input**
```json
{
  "ticket_id": "e5f6a7b8-c9d0-4e1f-2a3b-000000000001",
  "team": "payments"
}
```

**Output**
```json
{
  "status": "routed",
  "ticket_id": "e5f6a7b8-c9d0-4e1f-2a3b-000000000001",
  "team": "payments"
}
```

Side effect: updates `assigned_team` on the ticket row.
