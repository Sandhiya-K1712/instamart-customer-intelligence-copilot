
from pathlib import Path
import pandas as pd

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"

# Load cleaned datasets
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Join orders with customer city and product category
df = orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

df = df.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Analyse completed orders only
completed = df[df["status"] == "completed"].copy()

completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="coerce"
)

print("\n" + "=" * 60)
print("PANDAS EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Pivot table: Revenue by city and category
print("\n1. REVENUE BY CITY AND CATEGORY")

pivot = pd.pivot_table(
    completed,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True
)

print(pivot.round(2))

# 2. Named aggregation: City-wise revenue
print("\n2. CITY-WISE REVENUE SUMMARY")

city_summary = completed.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    order_count=("order_id", "count")
).reset_index()

print(city_summary.round(2).to_string(index=False))

# 3. Order amount statistics
print("\n3. ORDER AMOUNT STATISTICS")

amount = completed["order_amount"].dropna()

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers_above = (amount > upper_fence).sum()
outliers_below = (amount < lower_fence).sum()

print(f"Mean: {amount.mean():.2f}")
print(f"Median: {amount.median():.2f}")
print(f"Skewness: {amount.skew():.2f}")
print(f"Q1: {q1:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Lower fence: {lower_fence:.2f}")
print(f"Upper fence: {upper_fence:.2f}")
print(f"Outliers above upper fence: {outliers_above}")
print(f"Outliers below lower fence: {outliers_below}")

# 4. Correlation analysis
print("\n4. CORRELATION ANALYSIS")

correlation = completed[
    ["order_amount", "discount_pct", "quantity"]
].corr()

print(correlation.round(3))

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)
