-- 30_analytics.sql
-- Analytics views for BI consumption
-- Run after 10_dims.sql and 20_fact.sql

-- Master analytical view: daily market snapshot
CREATE OR REPLACE VIEW v_daily_market_snapshot AS
SELECT
    dc.full_date,
    dc.year,
    dc.month,
    dc.season,
    dn.neighbourhood,
    dn.neighbourhood_group,
    dl.room_type,
    dl.property_type,
    COUNT(DISTINCT f.listing_key) as active_listings,
    SUM(CASE WHEN f.is_available THEN 1 ELSE 0 END) as available_count,
    ROUND(AVG(f.price)::numeric, 2) as avg_price,
    ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price))::numeric, 2) as median_price,
    MIN(f.price) as min_price,
    MAX(f.price) as max_price,
    SUM(f.est_revenue) as total_estimated_revenue,
    SUM(f.reviews_to_date) as total_reviews
FROM fact_listings_daily_snapshot f
JOIN dim_listings dl ON f.listing_key = dl.listing_key
JOIN dim_hosts dh ON f.host_key = dh.host_key
JOIN dim_neighbourhoods dn ON f.neighbourhood_key = dn.neighbourhood_key
JOIN dim_calendar_dates dc ON f.date_key = dc.date_key
GROUP BY
    dc.full_date, dc.year, dc.month, dc.season,
    dn.neighbourhood, dn.neighbourhood_group,
    dl.room_type, dl.property_type;

-- Host performance view
CREATE OR REPLACE VIEW v_host_performance AS
SELECT
    dh.host_id,
    dh.host_name,
    dh.is_superhost,
    dh.host_since,
    dh.host_listings_count,
    COUNT(DISTINCT f.listing_key) as active_listings,
    ROUND(AVG(f.price)::numeric, 2) as avg_price,
    SUM(CASE WHEN f.is_available THEN 1 ELSE 0 END) as total_available_days,
    SUM(f.est_revenue) as total_estimated_revenue,
    SUM(f.reviews_to_date) as total_reviews
FROM fact_listings_daily_snapshot f
JOIN dim_hosts dh ON f.host_key = dh.host_key
GROUP BY
    dh.host_id, dh.host_name, dh.is_superhost,
    dh.host_since, dh.host_listings_count;

-- Neighbourhood trend analysis
CREATE OR REPLACE VIEW v_neighbourhood_trends AS
SELECT
    dn.neighbourhood,
    dn.neighbourhood_group,
    dc.full_date,
    COUNT(DISTINCT f.listing_key) as listing_count,
    ROUND(AVG(f.price)::numeric, 2) as avg_price,
    SUM(CASE WHEN f.is_available THEN 1 ELSE 0 END) as available_count,
    ROUND(100.0 * SUM(CASE WHEN f.is_available THEN 1 ELSE 0 END) / COUNT(DISTINCT f.listing_key), 2) as availability_rate
FROM fact_listings_daily_snapshot f
JOIN dim_neighbourhoods dn ON f.neighbourhood_key = dn.neighbourhood_key
JOIN dim_calendar_dates dc ON f.date_key = dc.date_key
GROUP BY
    dn.neighbourhood, dn.neighbourhood_group, dc.full_date;
