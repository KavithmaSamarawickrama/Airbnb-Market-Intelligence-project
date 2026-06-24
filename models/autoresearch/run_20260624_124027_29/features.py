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

    keyword_list = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    # Binary indicators for each keyword
    for kw in keyword_list:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)

    # Aggregate text features
    df['name_len_chars'] = name_col.str.len().fillna(0)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0)
    # total keyword count
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0

    # ---------- Spatial features ----------
    # Global centroid (using available rows only)
    lat_mean = df['latitude'].mean(skipna=True)
    lon_mean = df['longitude'].mean(skipna=True)

    # Euclidean distance to centroid (approx, in degrees)
    df['dist_euclid_centroid'] = np.sqrt(
        (df['latitude'] - lat_mean) ** 2 + (df['longitude'] - lon_mean) ** 2
    )

    # Haversine distance to centroid (in km)
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
        R = 6371.0  # Earth radius in km
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

    # ---------- Categorical frequency encodings ----------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Base categoricals
    for cat_col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if cat_col in df.columns:
            df[f'{cat_col}_freq'] = freq_encode(df[cat_col].astype(str))
        else:
            df[f'{cat_col}_freq'] = 0.0

    # Interaction frequencies
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

    # ---------- Boolean to numeric ----------
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
        df = df.drop(columns=['target_price'])

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it existed
    if target is not None:
        df_numeric['target_price'] = target

    # Fill remaining NaNs with 0
    df_numeric = df_numeric.fillna(0)

    return df_numeric
