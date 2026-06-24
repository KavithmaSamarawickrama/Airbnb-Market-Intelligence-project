def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure expected columns exist
    for col in ['accommodates', 'bedrooms', 'beds', 'latitude', 'longitude', 'is_superhost',
                'host_listings_count', 'availability_rate', 'est_revenue', 'total_reviews',
                'host_tenure_days', 'room_type', 'property_type', 'neighbourhood',
                'neighbourhood_group', 'name']:
        if col not in df.columns:
            df[col] = np.nan

    # -----------------------------
    # Text features from `name`
    # -----------------------------
    name = df['name'].fillna('').astype(str).str.lower()

    # Keyword list
    kw_list = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central', 'beach', 'sea',
        'ocean', 'view', 'modern', 'spacious', 'loft', 'penthouse', 'villa', 'pool',
        'garden', 'city center', 'old town', 'new', 'renovated', 'balcony', 'terrace'
    ]

    # Binary flags for each keyword
    for kw in kw_list:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name.str.contains(kw, regex=False).astype(float)

    # Aggregate text features
    df['name_len_chars'] = name.str.len().astype(float)
    df['name_len_words'] = name.str.split().apply(len).astype(float)

    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0.0

    # -----------------------------
    # Geographic features
    # -----------------------------
    # Global centroid (mean lat/lon) as city center proxy
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)

    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    # Haversine distance to centroid
    def haversine(lat1, lon1, lat2, lon2):
        # convert decimal degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371.0  # Earth radius in kilometers
        return r * c

    df['dist_to_center_haversine_km'] = haversine(lat, lon, lat_mean, lon_mean)

    # Simple Euclidean distance in degrees (proxy)
    df['dist_to_center_euclid'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

    # -----------------------------
    # Host and capacity interaction features
    # -----------------------------
    df['host_tenure_days'] = df['host_tenure_days'].astype(float)
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0

    df['host_listings_count'] = df['host_listings_count'].astype(float)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))

    df['accommodates'] = df['accommodates'].astype(float)
    df['bedrooms'] = df['bedrooms'].astype(float)
    df['beds'] = df['beds'].astype(float)

    # Ratios with safe division
    df['beds_per_bedroom'] = np.where(df['bedrooms'] > 0, df['beds'] / df['bedrooms'], 0.0)
    df['accommodates_per_bed'] = np.where(df['beds'] > 0, df['accommodates'] / df['beds'], 0.0)
    df['accommodates_per_bedroom'] = np.where(df['bedrooms'] > 0, df['accommodates'] / df['bedrooms'], 0.0)
    df['beds_per_accommodate'] = np.where(df['accommodates'] > 0, df['beds'] / df['accommodates'], 0.0)

    # -----------------------------
    # Availability, revenue, and reviews
    # -----------------------------
    df['availability_rate'] = df['availability_rate'].astype(float)
    df['est_revenue'] = df['est_revenue'].astype(float)
    df['total_reviews'] = df['total_reviews'].astype(float)

    # Existing ratio from champion (keep)
    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'] + 1.0)

    # New intensity features
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'] * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'] + 0.1)

    # -----------------------------
    # Categorical frequency encoding
    # -----------------------------
    # Helper for frequency encoding
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Base frequency encodings
    df['room_type_freq'] = freq_encode(df['room_type'].astype(str))
    df['property_type_freq'] = freq_encode(df['property_type'].astype(str))
    df['neighbourhood_freq'] = freq_encode(df['neighbourhood'].astype(str))
    df['neighbourhood_group_freq'] = freq_encode(df['neighbourhood_group'].astype(str))

    # Interaction frequency encodings
    combo_ng_rt = (df['neighbourhood_group'].astype(str) + '|' + df['room_type'].astype(str))
    df['neighbourhood_group_room_type_freq'] = freq_encode(combo_ng_rt)

    combo_prop_rt = (df['property_type'].astype(str) + '|' + df['room_type'].astype(str))
    df['property_room_type_freq'] = freq_encode(combo_prop_rt)

    # -----------------------------
    # Boolean to numeric
    # -----------------------------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # -----------------------------
    # Drop non-numeric / identifier / raw text/date columns
    # -----------------------------
    cols_to_drop = ['listing_id', 'name', 'room_type', 'property_type', 'neighbourhood',
                    'neighbourhood_group', 'host_since']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Keep target_price if present; it will be separated outside this function
    # Ensure all remaining columns except target_price are numeric
    for col in df.columns:
        if col == 'target_price':
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            # Attempt to coerce to numeric; non-convertible values become NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill remaining NaNs with 0
    df = df.fillna(0.0)

    return df
