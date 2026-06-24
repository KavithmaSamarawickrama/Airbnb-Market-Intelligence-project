-- 10_dims.sql
-- Build conformed dimension tables in Gold layer
-- Run after Silver layer is complete

-- Dimension: Dates
CREATE TABLE IF NOT EXISTS dim_calendar_dates (
    date_key INT PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INT NOT NULL,
    month INT NOT NULL,
    day_of_week INT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    season VARCHAR(10) NOT NULL
);

-- Dimension: Listings (Physical properties)
CREATE TABLE IF NOT EXISTS dim_listings (
    listing_key INT PRIMARY KEY,
    listing_id BIGINT NOT NULL UNIQUE,
    name VARCHAR(255),
    room_type VARCHAR(50),
    property_type VARCHAR(100),
    accommodates INT,
    bedrooms FLOAT,
    beds FLOAT,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Dimension: Hosts (Property owners)
CREATE TABLE IF NOT EXISTS dim_hosts (
    host_key INT PRIMARY KEY,
    host_id BIGINT NOT NULL UNIQUE,
    host_name VARCHAR(255),
    is_superhost BOOLEAN,
    host_since DATE,
    host_listings_count INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Dimension: Neighbourhoods (Geographic clustering)
CREATE TABLE IF NOT EXISTS dim_neighbourhoods (
    neighbourhood_key INT PRIMARY KEY,
    neighbourhood VARCHAR(100) NOT NULL,
    neighbourhood_group VARCHAR(100),
    centroid_lat FLOAT NOT NULL,
    centroid_lon FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(neighbourhood, neighbourhood_group)
);

-- Create indexes for join performance
CREATE INDEX IF NOT EXISTS idx_dim_listings_id ON dim_listings(listing_id);
CREATE INDEX IF NOT EXISTS idx_dim_hosts_id ON dim_hosts(host_id);
CREATE INDEX IF NOT EXISTS idx_dim_neighbourhoods_name ON dim_neighbourhoods(neighbourhood);
CREATE INDEX IF NOT EXISTS idx_dim_calendar_date ON dim_calendar_dates(full_date);
