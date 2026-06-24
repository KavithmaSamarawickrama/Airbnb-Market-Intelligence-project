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
    name_col = 'name'
    if name_col in df.columns:
        name_series = df[name_col].fillna('').astype(str).str.lower()
    else:
        name_series = pd.Series([''] * len(df), index=df.index)

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_series.str.contains(kw, regex=False).astype(int)

    # Aggregate text stats
    df['name_len_chars'] = name_series.str.len()
    df['name_len_words'] = name_series.str.split().apply(len)
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0

    # ---------- Spatial features ----------
    # Global centroid
    lat_mean = df['latitude'].astype(float).mean()
    lon_mean = df['longitude'].astype(float).mean()

    # Euclidean distance to centroid (approximate)
    df['dist_euclid_centroid'] = np.sqrt(
        (df['latitude'] - lat_mean) ** 2 + (df['longitude'] - lon_mean) ** 2
    )

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

    df['dist_haversine_centroid'] = haversine(df['latitude'], df['longitude'], lat_mean, lon_mean)

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

    cat_cols = ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']
    for col in cat_cols:
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

    # ---------- Basic numeric safety ----------
    # Ensure boolean is_superhost is numeric
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target if present
    target_col = 'target_price'
    if target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target = None

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it existed
    if target is not None:
        numeric_df[target_col] = target

    # Fill remaining NaNs with 0
    numeric_df = numeric_df.fillna(0)

    return numeric_df
