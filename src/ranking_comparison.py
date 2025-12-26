"""
Ranking Algorithm Comparison for Upselling
Compares 4 algorithms: Naive Bayes (custom), k-NN, Decision Tree (ID3), AdaBoost
All use Score(p | coș) = P(p | coș) × price(p)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from tqdm import tqdm
import os

# CONFIG
DATA_PATH = "data/ap_dataset.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
K_VALUES = [1, 3, 5]

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

# LOAD DATA
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["data_bon"] = pd.to_datetime(df["data_bon"])

product_prices = df.groupby("retail_product_name")["SalePriceWithVAT"].mean().to_dict()
print(f"Total: {len(df)} transactions, {df['id_bon'].nunique()} baskets, {len(CANDIDATES)} candidate sauces")

# TRAIN/TEST SPLIT
all_bons = df["id_bon"].unique()
train_bons, test_bons = train_test_split(all_bons, test_size=TEST_SIZE, random_state=RANDOM_STATE)
df_train = df[df["id_bon"].isin(train_bons)]
df_test = df[df["id_bon"].isin(test_bons)]
print(f"Split: {len(train_bons)} train / {len(test_bons)} test baskets")

# FEATURE ENGINEERING
def build_dataset_for_product(df, product):
    """Build features excluding the target product"""
    df_feat = df[df["retail_product_name"] != product]
    product_matrix = df_feat.groupby(["id_bon", "retail_product_name"]).size().unstack(fill_value=0)
    baskets = df.groupby("id_bon").agg(
        cart_size=("retail_product_name", "count"),
        distinct_products=("retail_product_name", "nunique"),
        total_value=("SalePriceWithVAT", "sum"),
        data_bon=("data_bon", "first")
    )
    baskets["day_of_week"] = baskets["data_bon"].dt.dayofweek + 1
    baskets["hour"] = baskets["data_bon"].dt.hour
    baskets["is_weekend"] = baskets["day_of_week"].isin([6, 7]).astype(int)
    X = product_matrix.join(baskets.drop(columns=["data_bon"]))
    baskets_with_product = df[df["retail_product_name"] == product]["id_bon"].unique()
    y = X.index.isin(baskets_with_product).astype(int)
    return X, y

# CUSTOM NAIVE BAYES
class BernoulliNaiveBayes:
    def fit(self, X, y):
        X_bin = (X > 0).astype(int)
        self.prior_1 = y.mean()
        self.prior_0 = 1 - self.prior_1
        self.p1 = (X_bin[y == 1].sum(axis=0) + 1) / (len(X_bin[y == 1]) + 2)
        self.p0 = (X_bin[y == 0].sum(axis=0) + 1) / (len(X_bin[y == 0]) + 2)
        return self

    def predict_proba(self, X):
        X_bin = (X > 0).astype(int)
        probs = []
        for _, x in X_bin.iterrows():
            log_p1 = np.log(self.prior_1) + np.sum(x * np.log(self.p1) + (1 - x) * np.log(1 - self.p1))
            log_p0 = np.log(self.prior_0) + np.sum(x * np.log(self.p0) + (1 - x) * np.log(1 - self.p0))
            max_log = max(log_p1, log_p0)
            p1 = np.exp(log_p1 - max_log)
            p0 = np.exp(log_p0 - max_log)
            probs.append(p1 / (p1 + p0))
        return np.array(probs)

# TRAIN ALL MODELS
print("\nTraining models for 8 sauces...")
models = {'NaiveBayes': {}, 'kNN': {}, 'DecisionTree': {}, 'AdaBoost': {}}
feature_names = {}
X_test_cache = {}

for product in tqdm(CANDIDATES, desc="Training"):
    X_train, y_train = build_dataset_for_product(df_train, product)
    X_test, _ = build_dataset_for_product(df_test, product)
    X_test_cache[product] = X_test
    feature_names[product] = X_train.columns.tolist()
    
    # 1. Naive Bayes (Custom)
    models['NaiveBayes'][product] = BernoulliNaiveBayes().fit(X_train, y_train)
    
    # 2. k-NN
    models['kNN'][product] = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1).fit(X_train, y_train)
    
    # 3. Decision Tree (ID3-like)
    models['DecisionTree'][product] = DecisionTreeClassifier(
        criterion='entropy', max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=RANDOM_STATE
    ).fit(X_train, y_train)
    
    # 4. AdaBoost
    models['AdaBoost'][product] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        n_estimators=50, learning_rate=1.0, random_state=RANDOM_STATE
    ).fit(X_train, y_train)

# =========================
# RANKING FUNCTIONS
# =========================
def rank_products_cached(algorithm_name, bon_id, basket_df, K):
    """
    Ranking using pre-cached probabilities: Score(p | coș) = P(p | coș) × price(p)
    """
    scores = {}
    
    for product in CANDIDATES:
        # Skip if product already in basket
        if product in basket_df["retail_product_name"].values:
            continue

        # Get cached probability
        if bon_id not in prob_cache[algorithm_name][product]:
            continue
        
        prob = prob_cache[algorithm_name][product][bon_id]
        
        # Calculate score: P(p | coș) × price(p)
        price = product_prices.get(product, 1.0)
        scores[product] = prob * price

    return sorted(scores, key=scores.get, reverse=True)[:K]

def rank_products(algorithm_name, bon_id, basket_df, K):
    """
    Generic ranking function using Score(p | coș) = P(p | coș) × price(p)
    [Used for demonstration only - evaluation uses cached version]
    """
    scores = {}
    
    for product in CANDIDATES:
        # Skip if product already in basket
        if product in basket_df["retail_product_name"].values:
            continue

        X = X_test_cache[product]
        if bon_id not in X.index:
            continue

        x = X.loc[[bon_id], feature_names[product]]
        
        # Get probability P(p | coș)
        if algorithm_name == 'NaiveBayes':
            prob = models[algorithm_name][product].predict_proba(x)[0]
        else:  # sklearn models
            prob = models[algorithm_name][product].predict_proba(x)[0, 1]
        
        # Calculate score: P(p | coș) × price(p)
        price = product_prices.get(product, 1.0)
        scores[product] = prob * price

    return sorted(scores, key=scores.get, reverse=True)[:K]

# PRE-COMPUTE PROBABILITIES FOR FAST EVALUATION
print("Pre-computing probabilities...")
prob_cache = {alg: {} for alg in ['NaiveBayes', 'kNN', 'DecisionTree', 'AdaBoost']}

for alg in tqdm(['NaiveBayes', 'kNN', 'DecisionTree', 'AdaBoost'], desc="Caching"):
    for product in CANDIDATES:
        X_test = X_test_cache[product]
        if alg == 'NaiveBayes':
            probs = models[alg][product].predict_proba(X_test)
        else:
            probs = models[alg][product].predict_proba(X_test)[:, 1]
        prob_cache[alg][product] = dict(zip(X_test.index, probs))

# =========================
# HIT@K EVALUATION (using cached probabilities)
# =========================
def hit_at_k(algorithm_name, df_test, K):
    """Evaluate Hit@K for given algorithm using cached probabilities"""
    hits = 0
    total = 0

    for bon_id, basket in df_test.groupby("id_bon"):
        real_products = [
            p for p in CANDIDATES
            if p in basket["retail_product_name"].values
        ]

        for real_p in real_products:
            # Remove one real product and try to recover it
            basket_wo = basket[basket["retail_product_name"] != real_p]
            if basket_wo.empty:
                continue

            top_k = rank_products_cached(algorithm_name, bon_id, basket_wo, K)
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

# RUN EVALUATION
print("\nEvaluating algorithms...")
results = {'NaiveBayes': [], 'kNN': [], 'DecisionTree': [], 'AdaBoost': [], 'Popularity': []}

for K in K_VALUES:
    for alg in ['NaiveBayes', 'kNN', 'DecisionTree', 'AdaBoost']:
        results[alg].append(hit_at_k(alg, df_test, K))
    popular_top_k = popularity_baseline(df_train, K)
    results['Popularity'].append(hit_at_k_baseline(df_test, popular_top_k))

# DISPLAY RESULTS
print("\n" + "="*70)
print("RESULTS")
print("="*70)
results_df = pd.DataFrame(results, index=[f"Hit@{k}" for k in K_VALUES])
print("\n" + results_df.to_string())
avg_results = {alg: np.mean(results[alg]) for alg in results}
print("\nAverage Hit@K:")
for alg, avg in sorted(avg_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {alg:15s}: {avg:.2%}")

# VISUALIZATION
print("\nGenerating visualizations...")
os.makedirs("output", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Line plot
for alg in ['NaiveBayes', 'kNN', 'DecisionTree', 'AdaBoost', 'Popularity']:
    ax1.plot(K_VALUES, [r * 100 for r in results[alg]], marker='o', label=alg, linewidth=2, markersize=8)
ax1.set_xlabel('K (Top-K)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Hit@K (%)', fontsize=12, fontweight='bold')
ax1.set_title('Ranking Algorithms Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(K_VALUES)
ax1.set_ylim(0, 105)

# Bar chart for K=3
k3_results = {alg: results[alg][1] * 100 for alg in results}
colors = ['#2ecc71' if alg == 'NaiveBayes' else '#3498db' if alg in ['kNN', 'DecisionTree', 'AdaBoost'] else '#95a5a6' 
          for alg in k3_results.keys()]
bars = ax2.bar(k3_results.keys(), k3_results.values(), color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Hit@3 (%)', fontsize=12, fontweight='bold')
ax2.set_title('Hit@3 Performance', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.grid(True, axis='y', alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("output/ranking_comparison_all.png", dpi=150, bbox_inches="tight")
print("✓ Saved: output/ranking_comparison_all.png")

# Heatmap
fig, ax = plt.subplots(figsize=(10, 5))
results_table = pd.DataFrame(results, index=[f"K={k}" for k in K_VALUES])
sns.heatmap(results_table * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
            vmin=0, vmax=100, cbar_kws={'label': 'Hit@K (%)'}, ax=ax, linewidths=1, linecolor='black')
ax.set_title('Hit@K Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("output/ranking_heatmap.png", dpi=150, bbox_inches="tight")
print("✓ Saved: output/ranking_heatmap.png")

print("\n" + "="*70)
print(f"DONE! Best: {max(avg_results, key=avg_results.get)} ({avg_results[max(avg_results, key=avg_results.get)]:.2%})")
print("="*70)
