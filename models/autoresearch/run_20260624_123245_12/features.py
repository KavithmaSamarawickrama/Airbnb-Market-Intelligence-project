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

    # Binary keyword indicators
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)

    # Aggregate text features
    df['name_len_chars'] = name_col.str.len().fillna(0)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0)
    # total keyword count
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    df['name_kw_count'] = df[kw_cols].sum(axis=1) if kw_cols else 0

    # ---------- Spatial features ----------
    # Global centroid (non-leaky, uses only feature distribution)
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    # Euclidean distance to centroid (approximate)
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
    def safe_div(num, den):
        return num / den.replace({0: np.nan})

    df['beds_per_bedroom'] = safe_div(df['beds'], df['bedrooms'])
    df['accommodates_per_bed'] = safe_div(df['accommodates'], df['beds'])
    df['accommodates_per_bedroom'] = safe_div(df['accommodates'], df['bedrooms'])
    df['beds_per_accommodate'] = safe_div(df['beds'], df['accommodates'])

    # ---------- Host features ----------
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].fillna(0))

    # ---------- Demand / intensity proxies ----------
    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'].fillna(0) + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'].fillna(0) * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'].fillna(0) + 0.1)

    # ---------- Frequency encoding for categoricals ----------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    cat_cols = []
    for c in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if c in df.columns:
            cat_cols.append(c)
            df[f'{c}_freq'] = freq_encode(df[c].astype(str))

    # Interaction frequency encodings
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(pair)

    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(pair2)

    # ---------- Basic boolean to numeric ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = ['listing_id', 'name', 'host_since', 'room_type', 'property_type',
                    'neighbourhood', 'neighbourhood_group']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop, errors='ignore')

    # Keep target if present
    target = None
    if 'target_price' in df.columns:
        target = df['target_price']
        df = df.drop(columns=['target_price'])

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Reattach target if it existed
    if target is not None:
        df_numeric['target_price'] = target

    # Fill remaining NaNs with 0
    df_numeric = df_numeric.fillna(0)

    return df_numeric
