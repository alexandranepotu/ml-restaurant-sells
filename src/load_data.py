import pandas as pd

# Load dataset 
DATA_PATH = "data/ap_dataset.csv"  

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst rows:")
print(df.head())

# Convert data_bon to datetime 
df["data_bon"] = pd.to_datetime(df["data_bon"])

# Basic stats 
print("\nNumber of receipts (id_bon):", df["id_bon"].nunique())
print("Number of rows:", len(df))
print("Unique products:", df["retail_product_name"].nunique())
print("Date interval:")
print(df["data_bon"].min(), "->", df["data_bon"].max())
