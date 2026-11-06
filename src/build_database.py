
import pandas as pd
import sqlite3
from pathlib import Path

# --------------------------------------------------
# 1. DEFINE FILE PATHS
# --------------------------------------------------

CLEAN_DIR = Path("data/clean")
DB_PATH = Path("data/instamart.db")

# Create database folder if needed
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD CLEANED CSV FILES
# --------------------------------------------------

customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Ensure order timestamp is stored as formatted text
orders["order_ts"] = (
    pd.to_datetime(orders["order_ts"])
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)

# --------------------------------------------------
# 3. CREATE SQLITE DATABASE AND TABLES
# --------------------------------------------------

with sqlite3.connect(DB_PATH) as connection:

    customers.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    products.to_sql(
        "products",
        connection,
        if_exists="replace",
        index=False
    )

    orders.to_sql(
        "orders",
        connection,
        if_exists="replace",
        index=False
    )

    # --------------------------------------------------
    # 4. VERIFY DATABASE TABLES
    # --------------------------------------------------

    cursor = connection.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """)

    tables = cursor.fetchall()

    print("DATABASE TABLES:")
    for table in tables:
        print("-", table[0])

    # Check row counts
    print("\nDATABASE ROW COUNTS:")

    for table_name in ["customers", "products", "orders"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{table_name}: {count}")

    # Check order_ts storage type
    cursor.execute("PRAGMA table_info(orders)")
    columns = cursor.fetchall()

    print("\nORDERS TABLE STRUCTURE:")
    for column in columns:
        print(column)

print("\nDATABASE CREATED AND VERIFIED")
print("Database location:", DB_PATH)
