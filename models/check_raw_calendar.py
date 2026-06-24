import gzip
import pandas as pd
from pathlib import Path

bronze_dir = Path("./data/raw_bronze/city=bergamo")
calendar_gz = next(bronze_dir.glob("**/calendar.csv.gz"), None)

if calendar_gz is None:
    print("Raw calendar file not found for Bergamo!")
else:
    print("Reading raw file...")
    chunks = []
    total_rows = 0
    total_non_null_price = 0
    total_non_null_adjusted = 0
    with gzip.open(calendar_gz, 'rt', encoding='utf-8') as f:
        # read in chunks for memory safety
        for chunk in pd.read_csv(f, chunksize=100000):
            total_rows += len(chunk)
            total_non_null_price += chunk['price'].notnull().sum()
            total_non_null_adjusted += chunk['adjusted_price'].notnull().sum()
            
    print(f"Total rows in calendar.csv.gz: {total_rows}")
    print(f"Total non-null price rows: {total_non_null_price}")
    print(f"Total non-null adjusted_price rows: {total_non_null_adjusted}")
