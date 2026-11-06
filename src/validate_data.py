
import sqlite3
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. DATABASE CONNECTION
# --------------------------------------------------

DB_PATH = Path("data/instamart.db")

connection = sqlite3.connect(DB_PATH)

# --------------------------------------------------
# 2. VALIDATION CHECKPOINTS
# --------------------------------------------------

checks = []

def check(description, condition):
    if not condition:
        raise AssertionError(f"FAILED: {description}")

    checks.append(description)
    print(f"PASSED: {description}")


# --------------------------------------------------
# 3. CHECK DATABASE TABLES
# --------------------------------------------------

tables = pd.read_sql_query(
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    """,
    connection
)["name"].tolist()

check(
    "Customers table exists",
    "customers" in tables
)

check(
    "Products table exists",
    "products" in tables
)

check(
    "Orders table exists",
    "orders" in tables
)


# --------------------------------------------------
# 4. CHECK ROW COUNTS
# --------------------------------------------------

customer_count = pd.read_sql_query(
    "SELECT COUNT(*) AS total FROM customers",
    connection
)["total"].iloc[0]

product_count = pd.read_sql_query(
    "SELECT COUNT(*) AS total FROM products",
    connection
)["total"].iloc[0]

order_count = pd.read_sql_query(
    "SELECT COUNT(*) AS total FROM orders",
    connection
)["total"].iloc[0]

check("Customer count is 1200", customer_count == 1200)
check("Product count is 40", product_count == 40)
check("Order count is 12309", order_count == 12309)


# --------------------------------------------------
# 5. CHECK ORDER STATUS COUNTS
# --------------------------------------------------

status_counts = pd.read_sql_query(
    """
    SELECT status, COUNT(*) AS total
    FROM orders
    GROUP BY status
    """,
    connection
).set_index("status")["total"].to_dict()

check(
    "Completed orders count is 11110",
    status_counts.get("completed", 0) == 11110
)

check(
    "Cancelled orders count is 1199",
    status_counts.get("cancelled", 0) == 1199
)


# --------------------------------------------------
# 6. CHECK ORDER AMOUNTS
# --------------------------------------------------

missing_amounts = pd.read_sql_query(
    """
    SELECT COUNT(*) AS total
    FROM orders
    WHERE order_amount IS NULL
    """,
    connection
)["total"].iloc[0]

check(
    "No missing order amounts",
    missing_amounts == 0
)


# --------------------------------------------------
# 7. CHECK DELIVERY TIMES
# --------------------------------------------------

missing_completed_delivery = pd.read_sql_query(
    """
    SELECT COUNT(*) AS total
    FROM orders
    WHERE status = 'completed'
    AND delivery_minutes IS NULL
    """,
    connection
)["total"].iloc[0]

check(
    "Completed orders have no missing delivery times",
    missing_completed_delivery == 0
)

missing_cancelled_delivery = pd.read_sql_query(
    """
    SELECT COUNT(*) AS total
    FROM orders
    WHERE status = 'cancelled'
    AND delivery_minutes IS NULL
    """,
    connection
)["total"].iloc[0]

check(
    "Cancelled orders have missing delivery times",
    missing_cancelled_delivery == 1199
)


# --------------------------------------------------
# 8. CHECK COMPLETED REVENUE
# --------------------------------------------------

completed_revenue = pd.read_sql_query(
    """
    SELECT ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    """,
    connection
)["revenue"].iloc[0]

check(
    "Completed revenue is 2998668.80",
    abs(completed_revenue - 2998668.80) < 0.01
)


# --------------------------------------------------
# 9. CHECK ORDER TIMESTAMP TYPE
# --------------------------------------------------

timestamp_type = pd.read_sql_query(
    "PRAGMA table_info(orders)",
    connection
)

order_ts_type = timestamp_type.loc[
    timestamp_type["name"] == "order_ts",
    "type"
].iloc[0]

check(
    "Order timestamp stored as TEXT",
    order_ts_type.upper() == "TEXT"
)


# --------------------------------------------------
# 10. FINAL RESULT
# --------------------------------------------------

connection.close()

print("\nTotal checkpoints passed:", len(checks))
print("ALL CHECKPOINTS PASSED")
