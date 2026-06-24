-- 20_fact.sql
-- Build fact table in Gold layer
-- Grain: one row per listing per calendar date
-- Depends on 10_dims.sql being run first

CREATE TABLE IF NOT EXISTS fact_listings_daily_snapshot (
    fact_id BIGINT PRIMARY KEY,
    listing_key INT NOT NULL REFERENCES dim_listings(listing_key),
    host_key INT NOT NULL REFERENCES dim_hosts(host_key),
    neighbourhood_key INT NOT NULL REFERENCES dim_neighbourhoods(neighbourhood_key),
    date_key INT NOT NULL REFERENCES dim_calendar_dates(date_key),
    price NUMERIC(10, 2),
    is_available BOOLEAN NOT NULL,
    minimum_nights INT,
    est_revenue NUMERIC(10, 2),
    reviews_to_date INT DEFAULT 0,
    captured_at TIMESTAMP DEFAULT NOW()
);

-- Clustering index for query performance
CREATE INDEX IF NOT EXISTS idx_fact_listing_date ON fact_listings_daily_snapshot(listing_key, date_key);
CREATE INDEX IF NOT EXISTS idx_fact_neighbourhood ON fact_listings_daily_snapshot(neighbourhood_key, date_key);
CREATE INDEX IF NOT EXISTS idx_fact_host ON fact_listings_daily_snapshot(host_key, date_key);
CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_listings_daily_snapshot(date_key);

-- Aggregate tables for BI layer (materialized views)
CREATE TABLE IF NOT EXISTS agg_daily_market_summary (
    date_key INT NOT NULL,
    neighbourhood_key INT NOT NULL,
    total_listings INT,
    available_count INT,
    avg_price NUMERIC(10, 2),
    median_price NUMERIC(10, 2),
    total_reviews INT,
    PRIMARY KEY (date_key, neighbourhood_key),
    FOREIGN KEY (date_key) REFERENCES dim_calendar_dates(date_key),
    FOREIGN KEY (neighbourhood_key) REFERENCES dim_neighbourhoods(neighbourhood_key)
);

CREATE INDEX IF NOT EXISTS idx_agg_summary_date ON agg_daily_market_summary(date_key);
