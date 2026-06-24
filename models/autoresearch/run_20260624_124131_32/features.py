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
    name = df['name'].fillna('').astype(str)
    name_lower = name.str.lower()

    keyword_list = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    total_kw_count = np.zeros(len(df), dtype=float)
    for kw in keyword_list:
        col_name = f'kw_{kw.replace(" ", "_")}'
        mask = name_lower.str.contains(kw, regex=False)
        df[col_name] = mask.astype(int)
        total_kw_count += df[col_name].values

    df['name_len_chars'] = name.str.len().fillna(0).astype(float)
    df['name_len_words'] = name.str.split().str.len().fillna(0).astype(float)
    df['name_keyword_count'] = total_kw_count

    # ---------- Spatial features ----------
    # Global centroid (using available rows only)
    lat_mean = df['latitude'].mean(skipna=True)
    lon_mean = df['longitude'].mean(skipna=True)

    # Haversine distance to centroid
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # km
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df['dist_centroid_haversine'] = haversine(
        df['latitude'].fillna(lat_mean),
        df['longitude'].fillna(lon_mean),
        lat_mean,
        lon_mean
    )

    # Simple Euclidean distance in lat/lon space
    df['dist_centroid_euclidean'] = np.sqrt(
        (df['latitude'].fillna(lat_mean) - lat_mean) ** 2 +
        (df['longitude'].fillna(lon_mean) - lon_mean) ** 2
    )

    # ---------- Capacity & layout ratios ----------
    eps = 1e-6
    df['beds_per_bedroom'] = df['beds'] / (df['bedrooms'].replace(0, np.nan))
    df['accommodates_per_bed'] = df['accommodates'] / (df['beds'].replace(0, np.nan))
    df['accommodates_per_bedroom'] = df['accommodates'] / (df['bedrooms'].replace(0, np.nan))
    df['beds_per_accommodate'] = df['beds'] / (df['accommodates'].replace(0, np.nan))

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
            df[f'{cat_col}_freq'] = freq_encode(df[cat_col].astype('object'))
        else:
            df[f'{cat_col}_freq'] = 0.0

    # Interaction frequencies
    if ('neighbourhood_group' in df.columns) and ('room_type' in df.columns):
        pair = df['neighbourhood_group'].astype('object').fillna('NA') + '||' + df['room_type'].astype('object').fillna('NA')
        df['neighbourhood_group_room_type_freq'] = freq_encode(pair)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if ('property_type' in df.columns) and ('room_type' in df.columns):
        pair2 = df['property_type'].astype('object').fillna('NA') + '||' + df['room_type'].astype('object').fillna('NA')
        df['property_room_type_freq'] = freq_encode(pair2)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = [
        'listing_id', 'name', 'host_since',
        'room_type', 'property_type', 'neighbourhood', 'neighbourhood_group'
    ]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target_price if present; ensure all others are numeric
    target_col = 'target_price'
    cols = list(df.columns)
    if target_col in cols:
        cols.remove(target_col)
        numeric_part = df[cols].apply(pd.to_numeric, errors='coerce')
        df_out = pd.concat([numeric_part, df[[target_col]]], axis=1)
    else:
        df_out = df.apply(pd.to_numeric, errors='coerce')

    # Fill remaining NaNs with 0
    df_out = df_out.fillna(0)

    return df_out
