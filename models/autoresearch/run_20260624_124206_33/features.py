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

    for kw in keywords:
        pattern = kw
        df[f'kw_{kw.replace(" ", "_")}'] = name_lower.str.contains(pattern, regex=False).astype(int)

    # Aggregate text stats
    df['name_len_chars'] = name_col.str.len().fillna(0)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0)
    # total keyword count
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    df['name_kw_count'] = df[kw_cols].sum(axis=1) if kw_cols else 0

    # ---------- Spatial features ----------
    # Fill missing lat/lon with global means to avoid NaNs in distances
    lat_mean = df['latitude'].mean() if df['latitude'].notna().any() else 0.0
    lon_mean = df['longitude'].mean() if df['longitude'].notna().any() else 0.0
    df['latitude'] = df['latitude'].fillna(lat_mean)
    df['longitude'] = df['longitude'].fillna(lon_mean)

    # Global centroid
    centroid_lat = df['latitude'].mean()
    centroid_lon = df['longitude'].mean()

    # Haversine distance to centroid (in km)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df['dist_centroid_haversine'] = haversine(df['latitude'], df['longitude'], centroid_lat, centroid_lon)
    # Simple Euclidean distance in lat/lon space
    df['dist_centroid_euclidean'] = np.sqrt((df['latitude'] - centroid_lat) ** 2 + (df['longitude'] - centroid_lon) ** 2)

    # ---------- Capacity & layout ratios ----------
    # Safe denominators
    bedrooms_safe = df['bedrooms'].replace(0, np.nan)
    beds_safe = df['beds'].replace(0, np.nan)
    accommodates_safe = df['accommodates'].replace(0, np.nan)

    df['beds_per_bedroom'] = (df['beds'] / bedrooms_safe).replace([np.inf, -np.inf], np.nan)
    df['accommodates_per_bed'] = (df['accommodates'] / beds_safe).replace([np.inf, -np.inf], np.nan)
    df['accommodates_per_bedroom'] = (df['accommodates'] / bedrooms_safe).replace([np.inf, -np.inf], np.nan)
    df['beds_per_accommodate'] = (df['beds'] / accommodates_safe).replace([np.inf, -np.inf], np.nan)

    # ---------- Host behavior & experience ----------
    df['host_tenure_days'] = df['host_tenure_days'].fillna(0)
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    df['host_listings_count'] = df['host_listings_count'].fillna(0)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'])

    # ---------- Demand / intensity proxies ----------
    df['total_reviews'] = df['total_reviews'].fillna(0)
    df['est_revenue'] = df['est_revenue'].fillna(0)
    df['availability_rate'] = df['availability_rate'].fillna(0)

    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'] + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'] * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'] + 0.1)

    # ---------- Categorical frequency encoding ----------
    cat_cols = []
    for c in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if c in df.columns:
            cat_cols.append(c)

    for c in cat_cols:
        freq = df[c].value_counts(dropna=False)
        df[f'{c}_freq'] = df[c].map(freq).fillna(0).astype(float)

    # Interaction frequencies
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        pair_freq = pair.value_counts(dropna=False)
        df['neighbourhood_group_room_type_freq'] = pair.map(pair_freq).fillna(0).astype(float)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        pair2_freq = pair2.value_counts(dropna=False)
        df['property_room_type_freq'] = pair2.map(pair2_freq).fillna(0).astype(float)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Binary encoding for boolean ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = ['listing_id', 'name', 'host_since', 'room_type', 'property_type',
                    'neighbourhood', 'neighbourhood_group']
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target if present
    target = None
    if 'target_price' in df.columns:
        target = df['target_price']

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Reattach target if it was present and dropped by numeric selection
    if target is not None:
        numeric_df['target_price'] = target

    # Fill remaining NaNs with 0
    numeric_df = numeric_df.fillna(0)

    return numeric_df
