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
    name_col = df['name'].fillna('').astype(str) if 'name' in df.columns else pd.Series([''] * len(df), index=df.index)
    name_lower = name_col.str.lower()

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    # Binary flags for each keyword
    for kw in keywords:
        if ' ' in kw:
            df[f'kw_{kw.replace(" ", "_")}'] = name_lower.str.contains(kw, regex=False).astype(int)
        else:
            df[f'kw_{kw}'] = name_lower.str.contains(kw, regex=False).astype(int)

    # Aggregate text features
    df['name_len_chars'] = name_col.str.len().fillna(0)
    df['name_len_words'] = name_lower.str.split().apply(len).astype(float)

    # Total keyword count
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['kw_total_count'] = df[kw_cols].sum(axis=1)
    else:
        df['kw_total_count'] = 0

    # ---------- Spatial features ----------
    # Global centroid (using available rows only)
    lat_mean = df['latitude'].mean(skipna=True)
    lon_mean = df['longitude'].mean(skipna=True)

    # Euclidean distance to centroid (approximate)
    df['dist_euclid_center'] = np.sqrt(
        (df['latitude'] - lat_mean) ** 2 + (df['longitude'] - lon_mean) ** 2
    )

    # Haversine distance to centroid (in km)
    def haversine(lat1, lon1, lat2, lon2):
        # Convert degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0  # Earth radius in km
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

    # ---------- Host behavior features ----------
    df['host_tenure_years'] = (df['host_tenure_days'] / 365.0).replace([np.inf, -np.inf], np.nan)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))

    # ---------- Demand / intensity proxies ----------
    df['revenue_per_review'] = (df['est_revenue'] / (df['total_reviews'] + 1.0)).replace([np.inf, -np.inf], np.nan)
    df['revenue_per_available_day'] = (
        df['est_revenue'] / (df['availability_rate'].clip(lower=0) * 365.0 + 1.0)
    ).replace([np.inf, -np.inf], np.nan)
    df['reviews_per_year_proxy'] = (
        df['total_reviews'] / (df['host_tenure_years'].fillna(0) + 0.1)
    ).replace([np.inf, -np.inf], np.nan)

    # ---------- Frequency encoding for categoricals ----------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    if 'room_type' in df.columns:
        df['room_type_freq'] = freq_encode(df['room_type'])
    else:
        df['room_type_freq'] = 0.0

    if 'property_type' in df.columns:
        df['property_type_freq'] = freq_encode(df['property_type'])
    else:
        df['property_type_freq'] = 0.0

    if 'neighbourhood' in df.columns:
        df['neighbourhood_freq'] = freq_encode(df['neighbourhood'])
    else:
        df['neighbourhood_freq'] = 0.0

    if 'neighbourhood_group' in df.columns:
        df['neighbourhood_group_freq'] = freq_encode(df['neighbourhood_group'])
    else:
        df['neighbourhood_group_freq'] = 0.0

    # Interaction frequency encodings
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair_ng_rt = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(pair_ng_rt)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair_prop_rt = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(pair_prop_rt)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Drop non-numeric / ID / raw text/date columns ----------
    cols_to_drop = [
        'listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
        'room_type', 'property_type', 'host_since'
    ]
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=c)

    # Keep target_price if present; ensure all other columns are numeric
    target_col = 'target_price'
    cols = list(df.columns)
    if target_col in cols:
        feature_cols = [c for c in cols if c != target_col]
    else:
        feature_cols = cols

    # Convert to numeric where possible
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    # Fill remaining NaNs with 0 (simple, robust baseline strategy)
    df = df.fillna(0)

    return df
