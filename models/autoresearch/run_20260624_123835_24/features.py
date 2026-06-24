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

    # Simple keyword indicators
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)

    # Aggregate text stats
    df['name_len_chars'] = name_col.str.len().fillna(0)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0)
    # Total keyword count
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    df['name_kw_count'] = df[kw_cols].sum(axis=1) if kw_cols else 0

    # ---------- Spatial features ----------
    # Global centroid (mean lat/lon) computed from available data
    lat_mean = df['latitude'].mean(skipna=True)
    lon_mean = df['longitude'].mean(skipna=True)

    df['lat_center'] = lat_mean
    df['lon_center'] = lon_mean

    # Euclidean distance to centroid (approx, in degrees)
    df['dist_euclid_center'] = np.sqrt(
        (df['latitude'] - lat_mean) ** 2 + (df['longitude'] - lon_mean) ** 2
    )

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

    df['dist_haversine_center'] = haversine(df['latitude'], df['longitude'], lat_mean, lon_mean)

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
    df['est_revenue'] = df['est_revenue'].fillna(0)
    df['total_reviews'] = df['total_reviews'].fillna(0)
    df['availability_rate'] = df['availability_rate'].fillna(0)

    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'] + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'] * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'] + 0.1)

    # ---------- Categorical frequency encoding ----------
    cat_cols = []
    if 'room_type' in df.columns:
        cat_cols.append('room_type')
    if 'property_type' in df.columns:
        cat_cols.append('property_type')
    if 'neighbourhood' in df.columns:
        cat_cols.append('neighbourhood')
    if 'neighbourhood_group' in df.columns:
        cat_cols.append('neighbourhood_group')

    for col in cat_cols:
        freq = df[col].value_counts(dropna=False)
        freq_map = freq / len(df)
        df[f'{col}_freq'] = df[col].map(freq_map).fillna(0).astype(float)

    # Interaction frequency encodings
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        freq_pair = pair.value_counts(dropna=False)
        freq_pair_map = freq_pair / len(df)
        df['neighbourhood_group_room_type_freq'] = pair.map(freq_pair_map).fillna(0).astype(float)

    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        freq_pair2 = pair2.value_counts(dropna=False)
        freq_pair2_map = freq_pair2 / len(df)
        df['property_room_type_freq'] = pair2.map(freq_pair2_map).fillna(0).astype(float)

    # ---------- Boolean to numeric ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = ['listing_id', 'name', 'host_since', 'room_type', 'property_type',
                    'neighbourhood', 'neighbourhood_group']
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target_price if present (it will be handled outside)
    target_col = 'target_price'
    if target_col in df.columns:
        target_series = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target_series = None

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it existed
    if target_series is not None:
        numeric_df[target_col] = target_series

    # Fill remaining NaNs with 0
    numeric_df = numeric_df.fillna(0)

    return numeric_df
