
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.base import clone 
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Snapshot moment
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

# Load clean datasets
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")

# Convert columns to correct data types
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Use completed orders only
completed = orders[
    orders["status"] == "completed"
].copy()

# Calculate RFM
rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency in whole days
rfm["recency"] = (
    SNAPSHOT - rfm["last_order_ts"]
).dt.days

# Keep required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Display RFM summary
print("\n" + "=" * 55)
print("TASK 1: RFM ANALYSIS")
print("=" * 55)

print("Number of customers:", len(rfm))
print("\nRFM statistics:")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nAdditional checks:")
print("Median recency:", rfm["recency"].median())
print("Minimum recency:", rfm["recency"].min())
print("Maximum recency:", rfm["recency"].max())
print("Maximum frequency:", rfm["frequency"].max())
print("Maximum monetary:", round(rfm["monetary"].max(), 2))

# ============================================================
# TASK 2: K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 55)
print("TASK 2: K-MEANS CLUSTERING")
print("=" * 55)

# Select RFM features
rfm_features = rfm[
    ["recency", "frequency", "monetary"]
]

# Standardise RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# Test K values from 2 to 8
clustering_results = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(rfm_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(rfm_scaled, labels)

    clustering_results.append({
        "K": k,
        "Inertia": round(inertia, 2),
        "Silhouette": round(silhouette, 3)
    })

# Display results
clustering_table = pd.DataFrame(clustering_results)

print("\nK-means evaluation results:")
print(clustering_table.to_string(index=False))

# ============================================================
# TASK 3: SEGMENT PROFILE AND PCA
# ============================================================

print("\n" + "=" * 55)
print("TASK 3: CUSTOMER SEGMENTATION")
print("=" * 55)

# Fit final K-means model with K=4
final_kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_kmeans.fit_predict(rfm_scaled)

# Profile each segment
segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).reset_index()

print("\nSegment profiles:")
print(segment_profile.round(2).to_string(index=False))

# PCA on the scaled RFM matrix
pca = PCA(n_components=2, random_state=42)
rfm_pca = pca.fit_transform(rfm_scaled)

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_.round(3))

print(
    "Total variance explained:",
    round(pca.explained_variance_ratio_.sum(), 3)
)

# Assign business names to the clusters
segment_names = {
    0: "Occasional Shoppers",
    1: "Premium Customers",
    2: "Frequent Loyalists",
    3: "Dormant Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Premium flag: highest mean monetary segment
rfm["is_premium"] = (
    rfm["segment_id"] == 1
)

# Save required customer segments file
customer_segments = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
].copy()

customer_segments.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print("\nCustomer segments saved successfully.")
print("File: outputs/customer_segments.csv")
print("\nSegment name counts:")
print(customer_segments["segment_name"].value_counts())

print(
    "\nPremium customers:",
    customer_segments["is_premium"].sum()
)

print(
    "\nSaved columns:",
    customer_segments.columns.tolist()
)

# ============================================================
# TASK 4: LAPSE COHORT AND FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 55)
print("TASK 4: LAPSE PREDICTION DATASET")
print("=" * 55)

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

# Separate historical orders from the label period
historical_orders = orders[
    orders["order_ts"] < CUTOFF
].copy()

september_orders = orders[
    (orders["order_ts"] >= CUTOFF) &
    (orders["order_ts"] < LABEL_END)
].copy()

# Only completed historical orders are used for behaviour features
historical_completed = historical_orders[
    historical_orders["status"] == "completed"
].copy()

# Cohort: customers with at least one historical completed order
cohort_ids = historical_completed["customer_id"].unique()

# Feature 1: recency, frequency and monetary
lapse_features = historical_completed.groupby(
    "customer_id"
).agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

lapse_features["recency"] = (
    CUTOFF - lapse_features["last_order_ts"]
).dt.days

# Feature 2: cancellation rate using final-status orders only
final_status_orders = historical_orders[
    historical_orders["status"].isin(["completed", "cancelled"])
].copy()

cancellation_summary = final_status_orders.groupby(
    "customer_id"
).agg(
    total_final_orders=("order_id", "nunique"),
    cancelled_orders=(
        "status",
        lambda x: (x == "cancelled").sum()
    )
).reset_index()

cancellation_summary["cancellation_rate"] = (
    cancellation_summary["cancelled_orders"] /
    cancellation_summary["total_final_orders"]
)

# Feature 3: most frequent product category
historical_with_category = historical_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_category = historical_with_category.groupby(
    "customer_id"
)["category"].agg(
    lambda x: x.mode().iloc[0]
).reset_index(name="favourite_category")

# Feature 4: customer information
customer_features = customers[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

# Combine historical features
lapse_features = lapse_features.merge(
    cancellation_summary[
        ["customer_id", "cancellation_rate"]
    ],
    on="customer_id",
    how="left"
)

lapse_features = lapse_features.merge(
    favourite_category,
    on="customer_id",
    how="left"
)

lapse_features = lapse_features.merge(
    customer_features,
    on="customer_id",
    how="left"
)

lapse_features["tenure_days"] = (
    CUTOFF - lapse_features["signup_date"]
).dt.days

# Create target label using September completed orders
september_completed_ids = set(
    september_orders.loc[
        september_orders["status"] == "completed",
        "customer_id"
    ]
)

lapse_features["lapsed"] = (
    ~lapse_features["customer_id"].isin(
        september_completed_ids
    )
).astype(int)

# Remove date helper columns
lapse_features = lapse_features.drop(
    columns=["last_order_ts", "signup_date"]
)

# Display cohort and label information
print("Cohort customers:", len(lapse_features))
print("\nLapse label counts:")
print(lapse_features["lapsed"].value_counts().sort_index())

print(
    "\nLapsed percentage:",
    round(lapse_features["lapsed"].mean() * 100, 2),
    "%"
)

print("\nFeature columns:")
print(lapse_features.columns.tolist())

print("\nMissing values:")
print(lapse_features.isnull().sum())
# ============================================================
# TASK 5: TRAIN-TEST SPLIT AND PREPROCESSING
# ============================================================

print("\n" + "=" * 55)
print("TASK 5: TRAIN-TEST SPLIT AND PREPROCESSING")
print("=" * 55)

# Separate features, target and customer IDs
X = lapse_features.drop(
    columns=["customer_id", "lapsed"]
)

y = lapse_features["lapsed"]
customer_ids = lapse_features["customer_id"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X,
    y,
    customer_ids,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining label counts:")
print(y_train.value_counts().sort_index())

print("\nTesting label counts:")
print(y_test.value_counts().sort_index())

# Baseline: predict every customer as not lapsed
baseline_predictions = [0] * len(y_test)

print("\nBaseline results:")
print(
    "Accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)
print(
    "Lapsed F1:",
    round(
        f1_score(
            y_test,
            baseline_predictions,
            zero_division=0
        ),
        4
    )
)

# Identify numeric and categorical features
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# Fit only on training data
X_train_transformed = preprocessor.fit_transform(X_train)

# Transform test data without fitting again
X_test_transformed = preprocessor.transform(X_test)

print("\nPreprocessing completed.")
print("Training transformed shape:", X_train_transformed.shape)
print("Testing transformed shape:", X_test_transformed.shape)

# Compare training scaler mean with full-cohort mean
numeric_scaler = preprocessor.named_transformers_["numeric"]

frequency_position = numeric_features.index("frequency")

print("\nFrequency mean comparison:")
print(
    "Training scaler frequency mean:",
    round(numeric_scaler.mean_[frequency_position], 4)
)
print(
    "Full cohort frequency mean:",
    round(lapse_features["frequency"].mean(), 4)
)
# ============================================================
# TASK 6: MODEL TRAINING AND HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 55)
print("TASK 6: MODEL TRAINING AND TUNING")
print("=" * 55)

# Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_params = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Logistic Regression...")
logistic_search.fit(X_train, y_train)

print("Best Logistic Regression parameters:")
print(logistic_search.best_params_)
print("Best CV F1:", round(logistic_search.best_score_, 4))


# Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

rf_params = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf_search.fit(X_train, y_train)

print("Best Random Forest parameters:")
print(rf_search.best_params_)
print("Best CV F1:", round(rf_search.best_score_, 4))


# Evaluate both models on the untouched test set
def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    print("\n" + "-" * 45)
    print(name)
    print("-" * 45)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print(
        "Precision:",
        round(
            precision_score(y_test, predictions, zero_division=0),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(y_test, predictions, zero_division=0),
            4
        )
    )

    print(
        "F1 score:",
        round(
            f1_score(y_test, predictions, zero_division=0),
            4
        )
    )


evaluate_model(
    "LOGISTIC REGRESSION",
    logistic_search.best_estimator_,
    X_test,
    y_test
)

evaluate_model(
    "RANDOM FOREST",
    rf_search.best_estimator_,
    X_test,
    y_test
)
# ============================================================
# TASK 7: OVERSAMPLING THE TRAINING DATA
# ============================================================

print("\n" + "=" * 55)
print("TASK 7: OVERSAMPLING")
print("=" * 55)

# Combine training features and labels
train_data = X_train.copy()
train_data["lapsed"] = y_train

# Separate the two classes
not_lapsed = train_data[train_data["lapsed"] == 0]
lapsed = train_data[train_data["lapsed"] == 1]

# Duplicate lapsed rows with replacement
lapsed_oversampled = lapsed.sample(
    n=len(not_lapsed) - len(lapsed),
    replace=True,
    random_state=42
)

# Create balanced training data
balanced_train = pd.concat(
    [train_data, lapsed_oversampled],
    ignore_index=True
)

# Shuffle the balanced data
balanced_train = balanced_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train_balanced = balanced_train.drop(columns=["lapsed"])
y_train_balanced = balanced_train["lapsed"]

print("Balanced training class counts:")
print(y_train_balanced.value_counts().sort_index())

# Refit the best logistic regression configuration
balanced_logistic_model = clone(
    logistic_search.best_estimator_
)

print("\nTraining Logistic Regression on balanced data...")

balanced_logistic_model.fit(
    X_train_balanced,
    y_train_balanced
)

# Evaluate on the original, untouched test set
balanced_predictions = balanced_logistic_model.predict(X_test)

print("\nBalanced Logistic Regression results:")
print("Confusion matrix:")
print(confusion_matrix(y_test, balanced_predictions))

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            balanced_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            balanced_predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "F1 score:",
    round(
        f1_score(
            y_test,
            balanced_predictions,
            zero_division=0
        ),
        4
    )
)
# ============================================================
# TASK 8: FEATURE IMPORTANCE, MODEL SAVING AND RETENTION LIST
# ============================================================

print("\n" + "=" * 55)
print("TASK 8: FEATURE IMPORTANCE AND RETENTION LIST")
print("=" * 55)

# ------------------------------------------------------------
# 1. Random Forest: Top 5 feature importances
# ------------------------------------------------------------

rf_model = rf_search.best_estimator_

rf_feature_names = (
    rf_model.named_steps["preprocessor"].get_feature_names_out()
)

rf_importances = pd.Series(
    rf_model.named_steps["model"].feature_importances_,
    index=rf_feature_names
).sort_values(ascending=False)

print("\nTop 5 Random Forest features:")
print(rf_importances.head(5).round(4))


# ------------------------------------------------------------
# 2. Logistic Regression: Top 3 coefficient signs
# ------------------------------------------------------------

lr_feature_names = (
    balanced_logistic_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

lr_coefficients = pd.Series(
    balanced_logistic_model
    .named_steps["model"]
    .coef_[0],
    index=lr_feature_names
)

top_coefficients = lr_coefficients.loc[
    lr_coefficients.abs().sort_values(
        ascending=False
    ).head(3).index
]

print("\nTop 3 Logistic Regression coefficients:")
print(top_coefficients.round(4))

print("\nCoefficient interpretation:")
for feature, coefficient in top_coefficients.items():
    direction = (
        "increases"
        if coefficient > 0
        else "decreases"
    )

    print(
        f"{feature}: {direction} the predicted likelihood "
        f"of customer lapse."
    )


# ------------------------------------------------------------
# 3. Save the selected oversampled Logistic Regression model
# ------------------------------------------------------------

model_path = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(
    balanced_logistic_model,
    model_path
)

print("\nSelected model saved:", model_path)


# ------------------------------------------------------------
# 4. Reload model and verify predictions
# ------------------------------------------------------------

loaded_model = joblib.load(model_path)

original_predictions = balanced_logistic_model.predict(X_test)
loaded_predictions = loaded_model.predict(X_test)

print(
    "Reloaded model predictions match:",
    (original_predictions == loaded_predictions).all()
)


# ------------------------------------------------------------
# 5. Create Premium customer retention list
# ------------------------------------------------------------

# Test customers and their predicted lapse probabilities
test_results = pd.DataFrame({
    "customer_id": ids_test.values,
    "lapse_probability": loaded_model.predict_proba(
        X_test
    )[:, 1],
    "actually_lapsed": y_test.values
})

# Load Premium segment information
segments = pd.read_csv(
    OUTPUT_DIR / "customer_segments.csv"
)

# Join segment information only for retention targeting
test_results = test_results.merge(
    segments[["customer_id", "is_premium"]],
    on="customer_id",
    how="left"
)

# Keep Premium customers and rank by lapse probability
retention_list = (
    test_results[
        test_results["is_premium"] == True
    ]
    .sort_values(
        "lapse_probability",
        ascending=False
    )
    .head(10)
    [["customer_id", "lapse_probability", "actually_lapsed"]]
    .reset_index(drop=True)
)

# Save retention list
retention_path = OUTPUT_DIR / "retention_list.csv"

retention_list.to_csv(
    retention_path,
    index=False
)

print("\nTop Premium customers for retention:")
print(retention_list.to_string(index=False))

print("\nRetention list saved:", retention_path)

print(
    "\nNumber of Premium customers selected:",
    len(retention_list)
)

print(
    "Number actually lapsed:",
    int(retention_list["actually_lapsed"].sum())
)

print("\nPART B TASKS COMPLETED.")
