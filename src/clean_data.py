
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load raw datasets
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# --------------------------------------------------
# 2. CLEAN CUSTOMERS
# --------------------------------------------------

customers["city"] = (
    customers["city"]
    .str.strip()
    .str.title()
)

customers["gender"] = (
    customers["gender"]
    .fillna("")
    .str.strip()
    .str.lower()
)

gender_mapping = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "": "unknown"
}

customers["gender"] = (
    customers["gender"]
    .map(gender_mapping)
    .fillna("unknown")
)

customers["age"] = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

invalid_age = (
    (customers["age"] < 10)
    | (customers["age"] > 100)
)

customers.loc[invalid_age, "age"] = pd.NA

customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

customers.to_csv(
    CLEAN_DIR / "customers.csv",
    index=False,
    date_format="%Y-%m-%d"
)

# --------------------------------------------------
# 3. CLEAN ORDERS - STEP 1
# Convert order_ts to datetime
# --------------------------------------------------

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

# --------------------------------------------------
# 4. CLEAN ORDERS - STEP 2
# Remove exact duplicate rows
# --------------------------------------------------

before_duplicates = len(orders)

orders = orders.drop_duplicates().copy()

exact_duplicates_removed = before_duplicates - len(orders)

# --------------------------------------------------
# 5. CLEAN ORDERS - STEP 3
# Keep latest order_ts for each order_id
# --------------------------------------------------

before_order_id_dedup = len(orders)

orders = (
    orders.sort_values("order_ts", kind="mergesort")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

duplicate_order_ids_removed = (
    before_order_id_dedup - len(orders)
)

# --------------------------------------------------
# 6. CLEAN ORDERS - STEP 4
# Fix order_amount
# --------------------------------------------------

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# Get product prices using product_id
price_map = products.set_index("product_id")["price_inr"]

orders["price_inr"] = orders["product_id"].map(price_map)

# Calculate missing or invalid order amounts
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

missing_amount = orders["order_amount"].isna()

orders.loc[missing_amount, "order_amount"] = (
    calculated_amount.loc[missing_amount]
)

# --------------------------------------------------
# 7. CLEAN ORDERS - STEP 5
# Fix negative delivery minutes
# --------------------------------------------------

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

# --------------------------------------------------
# 8. CLEAN ORDERS - STEP 6
# Add product category
# --------------------------------------------------

category_map = products.set_index("product_id")["category"]

orders["category"] = orders["product_id"].map(category_map)

# --------------------------------------------------
# 9. CLEAN ORDERS - STEP 7
# Fill missing delivery times for completed orders
# using category-wise median
# --------------------------------------------------

completed_mask = orders["status"].eq("completed")

completed_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .median()
)

missing_completed = (
    completed_mask
    & orders["delivery_minutes"].isna()
)

orders.loc[missing_completed, "delivery_minutes"] = (
    orders.loc[missing_completed, "category"]
    .map(completed_medians)
)

# Remove helper column
orders = orders.drop(columns=["price_inr", "category"])

# --------------------------------------------------
# 10. SAVE CLEAN DATASETS
# --------------------------------------------------

products.to_csv(
    CLEAN_DIR / "products.csv",
    index=False
)

orders.to_csv(
    CLEAN_DIR / "orders.csv",
    index=False,
    date_format="%Y-%m-%d %H:%M:%S"
)

# --------------------------------------------------
# 11. VALIDATE CLEANED DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("CUSTOMER CLEANING RESULTS")
print("=" * 60)

print("Customer shape:", customers.shape)
print("\nCity counts:")
print(customers["city"].value_counts().sort_index())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("Missing ages:", customers["age"].isna().sum())
print("Age range:", customers["age"].min(), "-", customers["age"].max())

print("\n" + "=" * 60)
print("ORDER CLEANING RESULTS")
print("=" * 60)

print("Exact duplicate rows removed:", exact_duplicates_removed)
print("Duplicate order IDs removed:", duplicate_order_ids_removed)

print("Final order shape:", orders.shape)

print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nMissing order amounts:", orders["order_amount"].isna().sum())

print(
    "Missing delivery time in completed orders:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)

print(
    "Missing delivery time in cancelled orders:",
    orders.loc[
        orders["status"].eq("cancelled"),
        "delivery_minutes"
    ].isna().sum()
)

completed_revenue = orders.loc[
    orders["status"].eq("completed"),
    "order_amount"
].sum()

print("\nCompleted revenue: INR", round(completed_revenue, 2))

print("\nClean files saved in:", CLEAN_DIR)
print("DATA CLEANING COMPLETED")
