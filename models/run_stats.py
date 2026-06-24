import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("models/dataset.csv")

# Clean target_price (exclude outliers > 1000 for cleaner comparisons)
df_clean = df.dropna(subset=["target_price"]).copy()
df_no_outliers = df_clean[df_clean["target_price"] < 1000].copy()

# How do we separate Bergamo and Denver?
# We can use coordinates: Bergamo is Lat > 45, Denver is Lat < 41
df_no_outliers["city"] = df_no_outliers["latitude"].apply(lambda x: "Bergamo" if x > 44 else "Denver")
df_clean["city"] = df_clean["latitude"].apply(lambda x: "Bergamo" if x > 44 else "Denver")

print("=== CROSS-CITY MARKET COMPARISON ===")
for city in ["Bergamo", "Denver"]:
    df_city = df_clean[df_clean["city"] == city]
    df_city_no_out = df_no_outliers[df_no_outliers["city"] == city]
    
    print(f"\nMarket: {city}")
    print(f"  Total listings: {len(df_city)}")
    print(f"  Mean Price (with outliers): ${df_city['target_price'].mean():.2f}")
    print(f"  Median Price (with outliers): ${df_city['target_price'].median():.2f}")
    print(f"  Mean Price (excluding outliers < $1000): ${df_city_no_out['target_price'].mean():.2f}")
    print(f"  Mean Accommodates: {df_city['accommodates'].mean():.2f}")
    print(f"  Mean Bedrooms: {df_city['bedrooms'].mean():.2f}")
    print(f"  Mean Beds: {df_city['beds'].mean():.2f}")
    print(f"  Superhost percentage: {df_city['is_superhost'].mean()*100:.1f}%")
    print(f"  Availability Rate: {df_city['availability_rate'].mean()*100:.1f}%")
    
    # Room Type distribution
    room_types = df_city['room_type'].value_counts(normalize=True)*100
    print("  Room Types Distribution:")
    for rt, pct in room_types.items():
        print(f"    * {rt}: {pct:.1f}%")
