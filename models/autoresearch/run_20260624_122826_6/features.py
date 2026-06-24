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

    # ---------- TEXT FEATURES FROM `name` ----------
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

    kw_counts = []
    for kw in keywords:
        if ' ' in kw:
            mask = name_series.str.contains(kw, regex=False)
        else:
            mask = name_series.str.contains(r'\b' + kw + r'\b', regex=True)
        col_name = f'kw_{kw.replace(" ", "_")}'
        df[col_name] = mask.astype(int)
        kw_counts.append(df[col_name])

    # Aggregate text features
    df['name_len_chars'] = name_series.str.len()
    df['name_len_words'] = name_series.str.split().apply(len)
    if kw_counts:
        df['name_kw_count'] = np.vstack(kw_counts).sum(axis=0)
    else:
        df['name_kw_count'] = 0

    # ---------- SPATIAL FEATURES ----------
    # Global centroid
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

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

    df['dist_haversine_centroid'] = haversine(lat, lon, lat_mean, lon_mean)

    # Euclidean distance in lat/lon space (rough proxy)
    df['dist_euclidean_centroid'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

    # ---------- HOST & CAPACITY RATIOS ----------
    # Safe divisions
    def safe_divide(num, den):
        return np.where(den != 0, num / den, 0.0)

    df['beds_per_bedroom'] = safe_divide(df['beds'].astype(float), df['bedrooms'].astype(float))
    df['accommodates_per_bed'] = safe_divide(df['accommodates'].astype(float), df['beds'].astype(float))
    df['accommodates_per_bedroom'] = safe_divide(df['accommodates'].astype(float), df['bedrooms'].astype(float))
    df['beds_per_accommodate'] = safe_divide(df['beds'].astype(float), df['accommodates'].astype(float))

    # Host tenure in years and log host listings
    df['host_tenure_years'] = safe_divide(df['host_tenure_days'].astype(float), 365.0)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].astype(float).fillna(0))

    # ---------- AVAILABILITY / REVENUE / REVIEWS ----------
    df['revenue_per_review'] = safe_divide(df['est_revenue'].astype(float), df['total_reviews'].astype(float) + 1.0)
    df['revenue_per_available_day'] = safe_divide(
        df['est_revenue'].astype(float),
        df['availability_rate'].astype(float) * 365.0 + 1.0
    )
    df['reviews_per_year_proxy'] = safe_divide(
        df['total_reviews'].astype(float),
        df['host_tenure_years'].astype(float) + 0.1
    )

    # ---------- CATEGORICAL FREQUENCY ENCODING ----------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Basic frequency encodings
    for col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if col in df.columns:
            df[f'{col}_freq'] = freq_encode(df[col].astype(str))
        else:
            df[f'{col}_freq'] = 0.0

    # Interaction frequencies
    if ('neighbourhood_group' in df.columns) and ('room_type' in df.columns):
        combo = df['neighbourhood_group'].astype(str) + '|' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(combo)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if ('property_type' in df.columns) and ('room_type' in df.columns):
        combo2 = df['property_type'].astype(str) + '|' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(combo2)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- BOOLEAN TO NUMERIC ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- DROP NON-NUMERIC / UNUSED COLUMNS ----------
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target_price if present; it will be separated outside
    # Ensure only numeric columns (plus target_price if present)
    target_col = 'target_price'
    if target_col in df.columns:
        target = df[target_col]
        df_numeric = df.drop(columns=[target_col])
        df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
        df_numeric[target_col] = target
    else:
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Fill NaNs with 0
    df_numeric = df_numeric.fillna(0)

    return df_numeric
