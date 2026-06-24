import os
from pathlib import Path
import pandas as pd
import duckdb
from dotenv import load_dotenv

def prepare_data(db_path: Path, output_csv: Path):
    """
    Connect to local DuckDB and query conformed Gold layer tables.
    Aggregates daily snapshot data to listing-level to construct the tabular dataset.
    """
    print(f"Connecting to DuckDB at {db_path}...")
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found at {db_path}. Please build the gold layer first.")
        
    conn = duckdb.connect(str(db_path))
    
    query = """
    SELECT 
        l.listing_id,
        l.name,
        l.room_type,
        l.property_type,
        l.accommodates,
        l.bedrooms,
        l.beds,
        l.latitude,
        l.longitude,
        h.is_superhost,
        h.host_listings_count,
        h.host_since,
        n.neighbourhood,
        n.neighbourhood_group,
        -- Target variable (median listing price)
        MEDIAN(f.price) as target_price,
        AVG(f.price) as avg_price,
        AVG(CASE WHEN f.is_available THEN 1.0 ELSE 0.0 END) as availability_rate,
        SUM(f.est_revenue) as est_revenue,
        MAX(f.reviews_to_date) as total_reviews
    FROM dim_listings l
    JOIN fact_listings_daily_snapshot f ON l.listing_key = f.listing_key
    JOIN dim_hosts h ON f.host_key = h.host_key
    JOIN dim_neighbourhoods n ON f.neighbourhood_key = n.neighbourhood_key
    GROUP BY 
        l.listing_id, l.name, l.room_type, l.property_type, l.accommodates, l.bedrooms, l.beds, l.latitude, l.longitude,
        h.is_superhost, h.host_listings_count, h.host_since, n.neighbourhood, n.neighbourhood_group
    """
    
    print("Executing query to aggregate conformed listings and facts...")
    df = conn.execute(query).df()
    conn.close()
    
    print(f"Retrieved {len(df)} unique listings.")
    
    # Feature Engineering: Host Tenure
    print("Computing host tenure...")
    df['host_since'] = pd.to_datetime(df['host_since'])
    baseline_date = pd.to_datetime('2026-06-23')  # Using assessment reference date
    df['host_tenure_days'] = (baseline_date - df['host_since']).dt.days
    df['host_tenure_days'] = df['host_tenure_days'].fillna(-1)
    
    # Save output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Dataset prepared successfully and saved to {output_csv}")

if __name__ == "__main__":
    load_dotenv()
    db_file = Path("./data/pipeline.duckdb")
    out_file = Path("./models/dataset.csv")
    prepare_data(db_file, out_file)
