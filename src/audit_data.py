
import pandas as pd
from pathlib import Path

# Folder containing raw CSV files
RAW_DIR = Path("data/raw")

# Load datasets
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# Keep empty ticket text as an empty string
tickets = pd.read_csv(
    RAW_DIR / "tickets.csv",
    keep_default_na=False
)

# --------------------------------------------------
# 1. AUDIT CUSTOMERS, PRODUCTS AND ORDERS
# --------------------------------------------------

datasets = {
    "customers": customers,
    "products": products,
    "orders": orders
}

for name, df in datasets.items():

    print("\n" + "=" * 65)
    print(f"RAW DATA AUDIT: {name.upper()}")
    print("=" * 65)

    print("Shape:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")

    missing_count = df.isna().sum()
    missing_percentage = (missing_count / len(df)) * 100

    missing_report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage.round(2)
    })

    print(missing_report)


# --------------------------------------------------
# 2. ORDERS AUDIT FIGURES
# --------------------------------------------------

amount_numeric = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

delivery_numeric = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# Count both NaN values and empty strings
blank_delivery = (
    orders["delivery_minutes"].isna()
    | orders["delivery_minutes"]
        .astype("string")
        .str.strip()
        .eq("")
)

orders_audit = {
    "Raw order rows": len(orders),
    "Exact duplicate rows": orders.duplicated().sum(),
    "Pending orders": orders["status"].eq("pending").sum(),
    "Distinct order IDs": orders["order_id"].nunique(),
    "Non-numeric order amounts": amount_numeric.isna().sum(),
    "Negative delivery minutes": delivery_numeric.lt(0).sum(),
    "Blank delivery minutes": blank_delivery.sum()
}

print("\n" + "=" * 65)
print("ORDERS AUDIT FIGURES")
print("=" * 65)

for key, value in orders_audit.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# 3. CUSTOMERS AUDIT FIGURES
# --------------------------------------------------

blank_age = customers["age"].isna().sum()

age_numeric = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

invalid_age = (
    (age_numeric < 10) | (age_numeric > 100)
).sum()

# Count both NaN values and empty strings
blank_gender = (
    customers["gender"].isna()
    | customers["gender"]
        .astype("string")
        .str.strip()
        .eq("")
).sum()

non_blank_gender = customers.loc[
    customers["gender"].astype("string").str.strip().ne("")
    & customers["gender"].notna(),
    "gender"
]

customers_audit = {
    "Blank ages": blank_age,
    "Ages outside 10-100": invalid_age,
    "Distinct raw city spellings": customers["city"].nunique(),
    "Distinct non-blank raw gender spellings": non_blank_gender.nunique(),
    "Blank genders": blank_gender
}

print("\n" + "=" * 65)
print("CUSTOMERS AUDIT FIGURES")
print("=" * 65)

for key, value in customers_audit.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# 4. TICKETS DATA VERIFICATION
# --------------------------------------------------

print("\n" + "=" * 65)
print("TICKETS DATA VERIFICATION")
print("=" * 65)

print("Shape:", tickets.shape)
print("Text column dtype:", tickets["text"].dtype)
print("Empty ticket texts:", tickets["text"].eq("").sum())


# --------------------------------------------------
# 5. AUDIT COMPLETION
# --------------------------------------------------

print("\nRAW DATA AUDIT COMPLETED")
