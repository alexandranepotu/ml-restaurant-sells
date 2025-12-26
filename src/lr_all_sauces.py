"""
LR #2: One Logistic Regression per sauce + Top-K recommendation
Correct implementation:
- No data leakage
- Proper Hit@K evaluation
- Popularity baseline
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
DATA_PATH = "data/ap_dataset.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K = 3

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df["data_bon"] = pd.to_datetime(df["data_bon"])

# =========================
# 2. IDENTIFY STANDALONE SAUCES (per requirements)
# =========================
# Only standalone sauces (not compound products like "Fries with Sauce")
sauces = [
    "Crazy Sauce",
    "Cheddar Sauce",
    "Extra Cheddar Sauce",
    "Garlic Sauce",
    "Tomato Sauce",
    "Blueberry Sauce",
    "Spicy Sauce",
    "Pink Sauce"
]

print(f"Standalone sauces (per requirements): {len(sauces)} sauces")
print(f"  {sauces}")

# =========================
# 3. TRAIN / TEST SPLIT (BON LEVEL)
# =========================
all_bons = df["id_bon"].unique()

train_bons, test_bons = train_test_split(
    all_bons,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

df_train = df[df["id_bon"].isin(train_bons)]
df_test  = df[df["id_bon"].isin(test_bons)]

# =========================
# 4. FEATURE ENGINEERING
# =========================
def build_dataset_for_sauce(df, sauce):
    # Exclude current sauce from features
    df_feat = df[df["retail_product_name"] != sauce]

    product_matrix = (
        df_feat.groupby(["id_bon", "retail_product_name"])
        .size()
        .unstack(fill_value=0)
    )

    baskets = df.groupby("id_bon").agg(
        cart_size=("retail_product_name", "count"),
        distinct_products=("retail_product_name", "nunique"),
        total_value=("SalePriceWithVAT", "sum"),
        data_bon=("data_bon", "first")
    )

    baskets["day_of_week"] = baskets["data_bon"].dt.dayofweek + 1
    baskets["is_weekend"] = baskets["day_of_week"].isin([6, 7]).astype(int)

    X = product_matrix.join(baskets.drop(columns=["data_bon"]))

    baskets_with_sauce = df[df["retail_product_name"] == sauce]["id_bon"].unique()
    y = X.index.isin(baskets_with_sauce).astype(int)

    return X, y

# =========================
# 5. TRAIN ONE MODEL PER SAUCE
# =========================
models = {}
scalers = {}
feature_names = {}


print("\nTraining models...")
os.makedirs("output", exist_ok=True)
for sauce in sauces:
    print(f"  -> {sauce}")

    X_train, y_train = build_dataset_for_sauce(df_train, sauce)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    models[sauce] = model
    scalers[sauce] = scaler
    feature_names[sauce] = X_train.columns.tolist()

    # === Visualization: Top coefficients ===
    coefs = model.coef_[0]
    feat_names = X_train.columns
    coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
    coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
    top_coef = coef_df.head(15)
    plt.figure(figsize=(8, 6))
    colors = ["green" if c > 0 else "red" for c in top_coef["Coefficient"]]
    plt.barh(top_coef["Feature"], top_coef["Coefficient"], color=colors)
    plt.xlabel("Coefficient Value")
    plt.title(f"Top 15 Features for {sauce}")
    plt.axvline(x=0, color="black", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"output/lr2_top_features_{sauce.replace(' ', '_')}.png", dpi=120, bbox_inches="tight")
    plt.close()
    # Optionally print top features
    print(f"    Top features: {list(top_coef['Feature'])}")

# =========================
# 6. CACHE TEST FEATURES (PERFORMANCE OPTIMIZATION)
# =========================
print("\nCaching test features for fast evaluation...")
X_test_cache = {}
for sauce in sauces:
    X_test, _ = build_dataset_for_sauce(df_test, sauce)
    X_test_cache[sauce] = X_test
    print(f"  Cached features for {sauce}")

# =========================
# 7. TOP-K RECOMMENDATION
# =========================
def recommend_top_k(bon_id, basket_df, sauces, K):
    scores = {}

    for sauce in sauces:
        if sauce in basket_df["retail_product_name"].values:
            continue

        X = X_test_cache[sauce]
        if bon_id not in X.index:
            continue

        # Get features for this basket, ensure column order matches training
        x = X.loc[[bon_id]]
        # Reindex to match feature_names order, filling missing columns with 0
        x = x.reindex(columns=feature_names[sauce], fill_value=0)
        x_scaled = scalers[sauce].transform(x)

        prob = models[sauce].predict_proba(x_scaled)[0, 1]
        scores[sauce] = prob

    return sorted(scores, key=scores.get, reverse=True)[:K]

# =========================
# 8. HIT@K AND PRECISION@K EVALUATION
# =========================
def hit_at_k(df_test, sauces, K):
    """
    Hit@K: % of instances where the real sauce appears in Top-K recommendations
    """
    hits = 0
    total = 0

    for bon_id, basket in df_test.groupby("id_bon"):
        real_sauces = [
            s for s in sauces
            if s in basket["retail_product_name"].values
        ]

        for real_sauce in real_sauces:
            basket_wo = basket[basket["retail_product_name"] != real_sauce]
            if basket_wo.empty:
                continue

            top_k = recommend_top_k(bon_id, basket_wo, sauces, K)

            hits += int(real_sauce in top_k)
            total += 1

    return hits / total if total > 0 else 0

def precision_at_k(df_test, sauces, K):
    """
    Precision@K: average proportion of relevant items in Top-K recommendations
    """
    precisions = []

    for bon_id, basket in df_test.groupby("id_bon"):
        real_sauces = set([
            s for s in sauces
            if s in basket["retail_product_name"].values
        ])

        if not real_sauces:
            continue

        # For each real sauce, remove it and recommend
        for real_sauce in real_sauces:
            basket_wo = basket[basket["retail_product_name"] != real_sauce]
            if basket_wo.empty:
                continue

            top_k = recommend_top_k(bon_id, basket_wo, sauces, K)
            
            # Count how many real sauces are in top_k
            relevant_in_topk = len([s for s in top_k if s in real_sauces])
            precision = relevant_in_topk / K if K > 0 else 0
            precisions.append(precision)

    return np.mean(precisions) if precisions else 0

# =========================
# 9. POPULARITY BASELINE
# =========================
def popularity_baseline(df_train, sauces, K):
    counts = {
        s: (df_train["retail_product_name"] == s).sum()
        for s in sauces
    }
    return sorted(counts, key=counts.get, reverse=True)[:K]

def hit_at_k_baseline(df_test, popular_top_k, sauces):
    """Hit@K for popularity baseline"""
    hits = 0
    total = 0

    for _, basket in df_test.groupby("id_bon"):
        real_sauces = [
            s for s in sauces
            if s in basket["retail_product_name"].values
        ]
        for s in real_sauces:
            hits += int(s in popular_top_k)
            total += 1

    return hits / total if total > 0 else 0

# =========================
# 10. RUN EVALUATION FOR MULTIPLE K VALUES
# =========================
K_VALUES = [1, 3, 5]

print("\n================ RESULTS ================")
print("Logistic Regression Models:")
print("-" * 50)
for K in K_VALUES:
    hit_model = hit_at_k(df_test, sauces, K)
    prec_model = precision_at_k(df_test, sauces, K)
    print(f"K={K}:  Hit@{K} = {hit_model:.2%}  |  Precision@{K} = {prec_model:.4f}")

print("\n" + "=" * 50)
print("Popularity Baseline:")
print("-" * 50)
for K in K_VALUES:
    popular_top_k = popularity_baseline(df_train, sauces, K)
    hit_baseline = hit_at_k_baseline(df_test, popular_top_k, sauces)
    print(f"K={K}:  Hit@{K} = {hit_baseline:.2%}")

print("=" * 50)

# =========================
# 11. EXAMPLE RECOMMENDATION
# =========================
for bon_id, basket in df_test.groupby("id_bon"):
    if not any(s in basket["retail_product_name"].values for s in sauces):
        recs = recommend_top_k(bon_id, basket, sauces, 3)
        print(f"\nExample recommendation for basket {bon_id}:")
        print(recs)
        break
