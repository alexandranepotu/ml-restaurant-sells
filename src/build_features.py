import pandas as pd

DATA_PATH = "data/ap_dataset.csv"

df = pd.read_csv(DATA_PATH)
df["data_bon"] = pd.to_datetime(df["data_bon"])

# Product counts per basket
product_matrix = (
    df.groupby(["id_bon", "retail_product_name"])
      .size()
      .unstack(fill_value=0)
)

# Basket-level aggregations
baskets = df.groupby("id_bon").agg(
    cart_size=("retail_product_name", "count"),
    distinct_products=("retail_product_name", "nunique"),
    total_value=("SalePriceWithVAT", "sum"),
    data_bon=("data_bon", "first")
)

baskets["day_of_week"] = baskets["data_bon"].dt.dayofweek + 1
baskets["hour"] = baskets["data_bon"].dt.hour
baskets["is_weekend"] = baskets["day_of_week"].isin([6, 7]).astype(int)

# Final dataset
dataset = product_matrix.join(baskets)

print("Final dataset shape:", dataset.shape)
print(dataset.head())
