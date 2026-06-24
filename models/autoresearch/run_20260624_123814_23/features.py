def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure expected columns exist
    for col in ['accommodates', 'bedrooms', 'beds', 'latitude', 'longitude',
                'is_superhost', 'host_listings_count', 'availability_rate',
                'est_revenue', 'total_reviews', 'host_tenure_days']:
        if col not in df.columns:
            df[col] = np.nan

    # ---------- Text features from `name` ----------
    name_col = df['name'].fillna('').astype(str)
    name_lower = name_col.str.lower()

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    keyword_cols = []
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)
        keyword_cols.append(col_name)

    # Aggregate text features
    df['name_len_chars'] = name_col.str.len().fillna(0)
    df['name_len_words'] = name_col.str.split().apply(len).astype(float)
    df['name_keyword_count'] = df[keyword_cols].sum(axis=1)

    # ---------- Spatial features ----------
    # Basic safety for lat/lon
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)

    # Global centroid (using available rows only)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    df['lat_center'] = lat_mean
    df['lon_center'] = lon_mean

    # Euclidean distance to centroid (approx, in degrees)
    df['dist_euclid_center'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

    # Haversine distance to centroid (in km)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in km
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df['dist_haversine_center'] = haversine(lat, lon, lat_mean, lon_mean)

    # ---------- Capacity & layout ratios ----------
    accommodates = df['accommodates'].astype(float)
    bedrooms = df['bedrooms'].astype(float)
    beds = df['beds'].astype(float)

    # Existing-style ratios
    df['beds_per_bedroom'] = np.where((bedrooms > 0) & np.isfinite(bedrooms), beds / bedrooms, 0.0)
    df['accommodates_per_bed'] = np.where((beds > 0) & np.isfinite(beds), accommodates / beds, 0.0)

    # New ratios
    df['accommodates_per_bedroom'] = np.where((bedrooms > 0) & np.isfinite(bedrooms), accommodates / bedrooms, 0.0)
    df['beds_per_accommodate'] = np.where((accommodates > 0) & np.isfinite(accommodates), beds / accommodates, 0.0)

    # ---------- Host behavior & experience ----------
    host_tenure_days = df['host_tenure_days'].astype(float).fillna(0.0)
    df['host_tenure_days'] = host_tenure_days
    df['host_tenure_years'] = host_tenure_days / 365.0

    host_listings = df['host_listings_count'].astype(float).fillna(0.0)
    df['host_listings_count'] = host_listings
    df['host_listings_log1p'] = np.log1p(host_listings)

    # ---------- Demand / intensity proxies ----------
    est_revenue = df['est_revenue'].astype(float).fillna(0.0)
    total_reviews = df['total_reviews'].astype(float).fillna(0.0)
    availability_rate = df['availability_rate'].astype(float).fillna(0.0)

    df['est_revenue'] = est_revenue
    df['total_reviews'] = total_reviews
    df['availability_rate'] = availability_rate

    df['revenue_per_review'] = np.where(total_reviews >= 0, est_revenue / (total_reviews + 1.0), 0.0)
    df['revenue_per_available_day'] = est_revenue / (availability_rate * 365.0 + 1.0)

    host_tenure_years = df['host_tenure_years']
    df['reviews_per_year_proxy'] = total_reviews / (host_tenure_years + 0.1)

    # ---------- Categorical frequency encoding ----------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Individual categoricals
    for col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if col in df.columns:
            df[f'{col}_freq'] = freq_encode(df[col].astype(str))
        else:
            df[f'{col}_freq'] = 0.0

    # Interaction frequency encodings
    if ('neighbourhood_group' in df.columns) and ('room_type' in df.columns):
        pair = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(pair)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if ('property_type' in df.columns) and ('room_type' in df.columns):
        pair2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(pair2)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Basic numeric cleanup ----------
    # Ensure boolean is_superhost becomes numeric
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # Drop non-numeric / identifier / raw text / raw date columns
    cols_to_drop = [
        'listing_id', 'name', 'host_since',
        'room_type', 'property_type', 'neighbourhood', 'neighbourhood_group'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Keep target_price if present; it will be handled outside
    target_present = 'target_price' in df.columns
    target_series = df['target_price'] if target_present else None

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Reattach target if it was present
    if target_present:
        numeric_df['target_price'] = target_series

    # Fill remaining NaNs with 0 for model robustness
    numeric_df = numeric_df.fillna(0.0)

    return numeric_df
