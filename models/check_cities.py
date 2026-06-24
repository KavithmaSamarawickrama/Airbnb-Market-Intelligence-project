import pandas as pd

df = pd.read_csv("models/dataset.csv")
print("dataset.csv columns:", list(df.columns))
print("\nFirst 10 rows:")
print(df[['target_price', 'avg_price', 'accommodates', 'bedrooms', 'beds']].head(10))
print("\nDataset length:", len(df))
print("Null count for target_price:", df['target_price'].isnull().sum())
print("Null count for avg_price:", df['avg_price'].isnull().sum())
print("\nSummary stats:")
print(df[['target_price', 'avg_price', 'accommodates', 'bedrooms', 'beds', 'host_tenure_days']].describe())
