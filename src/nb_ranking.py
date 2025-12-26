"""
Naive Bayes Ranking for Upsell Recommendation
- custom Bernoulli Naive Bayes
- Score(p | cart) = P(p | cart) * price(p)
- Hit@K evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATA_PATH = "data/ap_dataset.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
K_VALUES = [1, 3, 5]  # Multiple K values for evaluation

# PRODUSE CANDIDATE (UPSHELL) – DEFINITE EXPLICIT
CANDIDATES = [
    "Crazy Sauce",
    "Blueberry Sauce",
    "Cheddar Sauce",
    "Tomato Sauce",
    "Garlic Sauce",
    "Pink Sauce",
    "Spicy Sauce",
    "Extra Cheddar Sauce"
]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df["data_bon"] = pd.to_datetime(df["data_bon"])

# Pret mediu per produs
product_prices = (
    df.groupby("retail_product_name")["SalePriceWithVAT"]
    .mean()
    .to_dict()
)

# =========================
# TRAIN / TEST SPLIT (BON LEVEL)
# =========================
all_bons = df["id_bon"].unique()

train_bons, test_bons = train_test_split(
    all_bons,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

df_train = df[df["id_bon"].isin(train_bons)]
df_test = df[df["id_bon"].isin(test_bons)]

# =========================
# FEATURE ENGINEERING
# =========================
def build_dataset_for_product(df, product):
    # Exclude current product from features (no leakage)
    df_feat = df[df["retail_product_name"] != product]

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

    baskets_with_product = df[df["retail_product_name"] == product]["id_bon"].unique()
    y = X.index.isin(baskets_with_product).astype(int)

    # Bernoulli NB → binarizare
    X_bin = (X > 0).astype(int)

    return X_bin, y

# =========================
# CUSTOM BERNOULLI NAIVE BAYES
# =========================
class BernoulliNaiveBayes:
    def fit(self, X, y):
        self.prior_1 = y.mean()
        self.prior_0 = 1 - self.prior_1

        # Laplace smoothing
        self.p1 = (X[y == 1].sum(axis=0) + 1) / (len(X[y == 1]) + 2)
        self.p0 = (X[y == 0].sum(axis=0) + 1) / (len(X[y == 0]) + 2)

    def predict_proba(self, X):
        probs = []

        for _, x in X.iterrows():
            log_p1 = np.log(self.prior_1) + np.sum(
                x * np.log(self.p1) + (1 - x) * np.log(1 - self.p1)
            )
            log_p0 = np.log(self.prior_0) + np.sum(
                x * np.log(self.p0) + (1 - x) * np.log(1 - self.p0)
            )

            # Softmax
            max_log = max(log_p1, log_p0)
            p1 = np.exp(log_p1 - max_log)
            p0 = np.exp(log_p0 - max_log)

            probs.append(p1 / (p1 + p0))

        return np.array(probs)

# =========================
# TRAIN NB MODELS (1 / PRODUCT)
# =========================
models = {}
feature_names = {}

print("Training Naive Bayes models...")
for product in CANDIDATES:
    print(f"  -> {product}")

    X_train, y_train = build_dataset_for_product(df_train, product)

    nb = BernoulliNaiveBayes()
    nb.fit(X_train, y_train)

    models[product] = nb
    feature_names[product] = X_train.columns.tolist()

# =========================
# CACHE TEST FEATURES
# =========================
X_test_cache = {}
for product in CANDIDATES:
    X_test_cache[product], _ = build_dataset_for_product(df_test, product)

# =========================
# RANKING FUNCTION
# =========================
def rank_products(bon_id, basket_df, K):
    scores = {}

    for product in CANDIDATES:
        if product in basket_df["retail_product_name"].values:
            continue

        X = X_test_cache[product]
        if bon_id not in X.index:
            continue

        x = X.loc[[bon_id], feature_names[product]]
        prob = models[product].predict_proba(x)[0]
        price = product_prices.get(product, 1.0)

        scores[product] = prob * price

    return sorted(scores, key=scores.get, reverse=True)[:K]

# =========================
# HIT@K EVALUATION
# =========================
def hit_at_k(df_test, K):
    hits = 0
    total = 0

    for bon_id, basket in df_test.groupby("id_bon"):
        real_products = [
            p for p in CANDIDATES
            if p in basket["retail_product_name"].values
        ]

        for real_p in real_products:
            basket_wo = basket[basket["retail_product_name"] != real_p]
            if basket_wo.empty:
                continue

            top_k = rank_products(bon_id, basket_wo, K)
            hits += int(real_p in top_k)
            total += 1

    return hits / total if total > 0 else 0

# =========================
# BASELINE: POPULARITY
# =========================
def popularity_baseline(df_train, K):
    counts = {
        p: (df_train["retail_product_name"] == p).sum()
        for p in CANDIDATES
    }
    return sorted(counts, key=counts.get, reverse=True)[:K]

def hit_at_k_baseline(df_test, popular_top_k):
    hits = 0
    total = 0

    for _, basket in df_test.groupby("id_bon"):
        real_products = [
            p for p in CANDIDATES
            if p in basket["retail_product_name"].values
        ]
        for p in real_products:
            hits += int(p in popular_top_k)
            total += 1

    return hits / total if total > 0 else 0

# =========================
# RUN EVALUATION
# =========================
print("\n================ RESULTS ================")
print("Naive Bayes Ranking:")
print("-" * 50)
for K in K_VALUES:
    hit_nb = hit_at_k(df_test, K)
    print(f"K={K}:  Hit@{K} = {hit_nb:.2%}")

print("\n" + "=" * 50)
print("Popularity Baseline:")
print("-" * 50)
for K in K_VALUES:
    popular_top_k = popularity_baseline(df_train, K)
    hit_pop = hit_at_k_baseline(df_test, popular_top_k)
    print(f"K={K}:  Hit@{K} = {hit_pop:.2%}")

print("=" * 50)

# =========================
# EXAMPLE RANKING
# =========================
for bon_id, basket in df_test.groupby("id_bon"):
    if not any(p in basket["retail_product_name"].values for p in CANDIDATES):
        ranking = rank_products(bon_id, basket, 3)
        print(f"\nExample ranking for basket {bon_id}:")
        print(ranking)
        break
