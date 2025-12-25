"""
LR #1: Crazy Sauce conditioned on Crazy Schnitzel
Binary classification with custom Logistic Regression implementation (Gradient Descent).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression as SklearnLR

# =============================================================================
# Configuration
# =============================================================================
DATA_PATH = "data/ap_dataset.csv"
TARGET_PRODUCT = "Crazy Schnitzel"  # Filter baskets containing this
TARGET_SAUCE = "Crazy Sauce"        # Predict presence of this
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# Custom Logistic Regression Implementation (Gradient Descent)
# =============================================================================
class LogisticRegressionCustom:
    """
    Logistic Regression implemented from scratch using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates.
    n_iterations : int, default=1000
        Number of gradient descent iterations.
    regularization : float, default=0.0
        L2 regularization strength (lambda).
    verbose : bool, default=False
        If True, print cost every 100 iterations.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization=0.0, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y, m):
        """
        Compute binary cross-entropy cost with L2 regularization.
        
        Cost = -1/m * Σ[y*log(h) + (1-y)*log(1-h)] + λ/(2m) * Σw²
        """
        h = self._sigmoid(np.dot(X, self.weights) + self.bias)
        # Clip to prevent log(0)
        h = np.clip(h, 1e-10, 1 - 1e-10)
        
        cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        
        # L2 regularization term
        if self.regularization > 0:
            cost += (self.regularization / (2 * m)) * np.sum(self.weights ** 2)
        
        return cost
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels (0 or 1).
        
        Returns
        -------
        self : object
            Fitted model.
        """
        m, n = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(z)
            
            # Compute gradients
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            
            # Add regularization gradient
            if self.regularization > 0:
                dw += (self.regularization / m) * self.weights
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Record cost
            cost = self._compute_cost(X, y, m)
            self.cost_history.append(cost)
            
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probability of class 1.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
        
        Returns
        -------
        proba : np.ndarray of shape (n_samples,)
            Probability of class 1.
        """
        return self._sigmoid(np.dot(X, self.weights) + self.bias)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
        threshold : float, default=0.5
            Classification threshold.
        
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted labels (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def get_coefficients(self):
        """Return weights and bias."""
        return self.weights, self.bias


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
def load_and_preprocess_data():
    """Load data and build feature matrix for baskets with Crazy Schnitzel."""
    
    print("=" * 60)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 60)
    
    # Load raw data
    df = pd.read_csv(DATA_PATH)
    df["data_bon"] = pd.to_datetime(df["data_bon"])
    
    print(f"Total rows: {len(df)}")
    print(f"Total baskets: {df['id_bon'].nunique()}")
    
    # Find baskets containing Crazy Schnitzel
    baskets_with_schnitzel = df[df["retail_product_name"] == TARGET_PRODUCT]["id_bon"].unique()
    print(f"\nBaskets containing '{TARGET_PRODUCT}': {len(baskets_with_schnitzel)}")
    
    # Filter data to only these baskets
    df_filtered = df[df["id_bon"].isin(baskets_with_schnitzel)]
    print(f"Rows in filtered data: {len(df_filtered)}")
    
    # Create target variable: does basket contain Crazy Sauce?
    baskets_with_sauce = df_filtered[df_filtered["retail_product_name"] == TARGET_SAUCE]["id_bon"].unique()
    
    # Build product count matrix (exclude target sauce to avoid data leakage)
    products_to_exclude = [TARGET_SAUCE]
    df_features = df_filtered[~df_filtered["retail_product_name"].isin(products_to_exclude)]
    
    product_matrix = (
        df_features.groupby(["id_bon", "retail_product_name"])
        .size()
        .unstack(fill_value=0)
    )
    
    # Basket-level aggregations
    baskets = df_filtered.groupby("id_bon").agg(
        cart_size=("retail_product_name", "count"),
        distinct_products=("retail_product_name", "nunique"),
        total_value=("SalePriceWithVAT", "sum"),
        data_bon=("data_bon", "first")
    )
    
    # Temporal features
    baskets["day_of_week"] = baskets["data_bon"].dt.dayofweek + 1
    baskets["hour"] = baskets["data_bon"].dt.hour
    baskets["is_weekend"] = baskets["day_of_week"].isin([6, 7]).astype(int)
    
    # Combine features
    dataset = product_matrix.join(baskets.drop(columns=["data_bon"]))
    
    # Create target variable
    dataset["has_crazy_sauce"] = dataset.index.isin(baskets_with_sauce).astype(int)
    
    print(f"\nFinal dataset shape: {dataset.shape}")
    print(f"Target distribution:")
    print(dataset["has_crazy_sauce"].value_counts())
    print(f"Sauce rate: {dataset['has_crazy_sauce'].mean():.2%}")
    
    return dataset


# =============================================================================
# Train-Test Split
# =============================================================================
def split_data(dataset):
    """Split data into train and test sets at basket level."""
    
    print("\n" + "=" * 60)
    print("STEP 2: Train-Test Split (80/20)")
    print("=" * 60)
    
    # Separate features and target
    X = dataset.drop(columns=["has_crazy_sauce"])
    y = dataset["has_crazy_sauce"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {len(X_train)} baskets")
    print(f"Test set: {len(X_test)} baskets")
    print(f"Train sauce rate: {y_train.mean():.2%}")
    print(f"Test sauce rate: {y_test.mean():.2%}")
    
    # Feature scaling (important for gradient descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = X.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names, scaler


# =============================================================================
# Model Training
# =============================================================================
def train_models(X_train, y_train):
    """Train custom LR, sklearn LR, and compute baseline."""
    
    print("\n" + "=" * 60)
    print("STEP 3: Training Models")
    print("=" * 60)
    
    # 1. Custom Logistic Regression (Gradient Descent)
    print("\n[1] Training Custom Logistic Regression (Gradient Descent)...")
    custom_lr = LogisticRegressionCustom(
        learning_rate=0.1,
        n_iterations=1000,
        regularization=0.1,
        verbose=True
    )
    custom_lr.fit(X_train, y_train)
    print(f"Final cost: {custom_lr.cost_history[-1]:.6f}")
    
    # 2. Sklearn Logistic Regression (for comparison)
    print("\n[2] Training Sklearn Logistic Regression...")
    sklearn_lr = SklearnLR(max_iter=1000, random_state=RANDOM_STATE)
    sklearn_lr.fit(X_train, y_train)
    print("Sklearn model trained.")
    
    # 3. Majority baseline
    majority_class = int(y_train.mean() >= 0.5)
    print(f"\n[3] Majority Baseline: always predict {majority_class}")
    
    return custom_lr, sklearn_lr, majority_class


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_models(custom_lr, sklearn_lr, majority_class, X_test, y_test):
    """Evaluate all models and compare."""
    
    print("\n" + "=" * 60)
    print("STEP 4: Model Evaluation")
    print("=" * 60)
    
    # Predictions
    y_pred_custom = custom_lr.predict(X_test)
    y_proba_custom = custom_lr.predict_proba(X_test)
    
    y_pred_sklearn = sklearn_lr.predict(X_test)
    y_proba_sklearn = sklearn_lr.predict_proba(X_test)[:, 1]
    
    y_pred_baseline = np.full(len(y_test), majority_class)
    
    # Metrics
    results = {}
    
    for name, y_pred, y_proba in [
        ("Custom LR", y_pred_custom, y_proba_custom),
        ("Sklearn LR", y_pred_sklearn, y_proba_sklearn),
        ("Baseline", y_pred_baseline, None)
    ]:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = 0.5  # Baseline has no discriminative power
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": auc
        }
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
    
    return results, y_pred_custom, y_proba_custom, y_pred_sklearn, y_proba_sklearn


def plot_results(custom_lr, y_test, y_proba_custom, y_proba_sklearn, 
                 y_pred_custom, feature_names):
    """Generate visualization plots."""
    
    print("\n" + "=" * 60)
    print("STEP 5: Generating Visualizations")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training cost curve
    ax1 = axes[0, 0]
    ax1.plot(custom_lr.cost_history)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cost (Binary Cross-Entropy)")
    ax1.set_title("Custom LR: Training Cost Curve")
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC Curves
    ax2 = axes[0, 1]
    fpr_custom, tpr_custom, _ = roc_curve(y_test, y_proba_custom)
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, y_proba_sklearn)
    
    ax2.plot(fpr_custom, tpr_custom, label=f"Custom LR (AUC={roc_auc_score(y_test, y_proba_custom):.3f})")
    ax2.plot(fpr_sklearn, tpr_sklearn, label=f"Sklearn LR (AUC={roc_auc_score(y_test, y_proba_sklearn):.3f})", linestyle="--")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix (Custom LR)
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred_custom)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
                xticklabels=["No Sauce", "Sauce"],
                yticklabels=["No Sauce", "Sauce"])
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Custom LR: Confusion Matrix")
    
    # 4. Top feature coefficients
    ax4 = axes[1, 1]
    weights, bias = custom_lr.get_coefficients()
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": weights
    }).sort_values("Coefficient", key=abs, ascending=False).head(15)
    
    colors = ["green" if c > 0 else "red" for c in coef_df["Coefficient"]]
    ax4.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax4.set_xlabel("Coefficient Value")
    ax4.set_title("Top 15 Features by Coefficient Magnitude")
    ax4.axvline(x=0, color="black", linewidth=0.5)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("output/lr_crazy_sauce_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plots saved to output/lr_crazy_sauce_results.png")


def interpret_coefficients(custom_lr, feature_names):
    """Interpret which products increase/decrease sauce probability."""
    
    print("\n" + "=" * 60)
    print("STEP 6: Coefficient Interpretation")
    print("=" * 60)
    
    weights, bias = custom_lr.get_coefficients()
    
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": weights
    }).sort_values("Coefficient", ascending=False)
    
    print("\n[+] TOP 10 Features INCREASING Crazy Sauce probability:")
    print("-" * 50)
    for _, row in coef_df.head(10).iterrows():
        print(f"  {row['Feature']:35s} {row['Coefficient']:+.4f}")
    
    print("\n[-] TOP 10 Features DECREASING Crazy Sauce probability:")
    print("-" * 50)
    for _, row in coef_df.tail(10).iterrows():
        print(f"  {row['Feature']:35s} {row['Coefficient']:+.4f}")
    
    print(f"\nBias term: {bias:.4f}")
    
    return coef_df


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Run the complete LR #1 pipeline."""
    
    print("\n" + "=" * 60)
    print("  LR #1: Crazy Sauce conditioned on Crazy Schnitzel")
    print("=" * 60)
    
    # Step 1: Load and preprocess
    dataset = load_and_preprocess_data()
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test, feature_names, scaler = split_data(dataset)
    
    # Step 3: Train models
    custom_lr, sklearn_lr, majority_class = train_models(X_train, y_train)
    
    # Step 4: Evaluate
    results, y_pred_custom, y_proba_custom, y_pred_sklearn, y_proba_sklearn = \
        evaluate_models(custom_lr, sklearn_lr, majority_class, X_test, y_test)
    
    # Step 5: Visualizations
    import os
    os.makedirs("output", exist_ok=True)
    plot_results(custom_lr, y_test, y_proba_custom, y_proba_sklearn, 
                 y_pred_custom, feature_names)
    
    # Step 6: Interpret coefficients
    coef_df = interpret_coefficients(custom_lr, feature_names)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Model Comparison")
    print("=" * 60)
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    
    print("\n[OK] LR #1 Complete!")
    
    return custom_lr, sklearn_lr, results, coef_df


if __name__ == "__main__":
    main()
