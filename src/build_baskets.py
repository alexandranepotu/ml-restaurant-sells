import pandas as pd

DATA_PATH = "data/ap_dataset.csv"

# Load data
df = pd.read_csv(DATA_PATH)
df["data_bon"] = pd.to_datetime(df["data_bon"])

# Group by receipt (id_bon)
baskets = df.groupby("id_bon").agg(
    cart_size=("retail_product_name", "count"),
    distinct_products=("retail_product_name", "nunique"),
    total_value=("SalePriceWithVAT", "sum"),
    data_bon=("data_bon", "first")
).reset_index()

# Time features
baskets["day_of_week"] = baskets["data_bon"].dt.dayofweek + 1
baskets["hour"] = baskets["data_bon"].dt.hour
baskets["is_weekend"] = baskets["day_of_week"].isin([6, 7]).astype(int)

print("Baskets shape:", baskets.shape)
print(baskets.head())
