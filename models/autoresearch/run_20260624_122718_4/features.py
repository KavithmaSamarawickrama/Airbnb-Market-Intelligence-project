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
    name_col = df['name'].fillna('').astype(str) if 'name' in df.columns else pd.Series([''] * len(df), index=df.index)
    name_lower = name_col.str.lower()

    keyword_map = {
        'kw_luxury': ['luxury'],
        'kw_cozy': ['cozy', 'cosy'],
        'kw_apartment': ['apartment', 'apt'],
        'kw_studio': ['studio'],
        'kw_downtown': ['downtown'],
        'kw_central': ['central'],
        'kw_beach': ['beach'],
        'kw_sea': ['sea'],
        'kw_ocean': ['ocean'],
        'kw_view': ['view'],
        'kw_modern': ['modern'],
        'kw_spacious': ['spacious'],
        'kw_loft': ['loft'],
        'kw_penthouse': ['penthouse'],
        'kw_villa': ['villa'],
        'kw_pool': ['pool'],
        'kw_garden': ['garden'],
        'kw_city_center': ['city center', 'citycentre', 'city-centre'],
        'kw_old_town': ['old town'],
        'kw_new': ['new'],
        'kw_renovated': ['renovated', 'refurbished'],
        'kw_balcony': ['balcony'],
        'kw_terrace': ['terrace']
    }

    kw_cols = []
    for kw_col, patterns in keyword_map.items():
        pattern_regex = '|'.join([pd.regex.escape(p) if hasattr(pd, 'regex') else p for p in patterns])
        # Use simple contains for robustness (case already lowered)
        df[kw_col] = 0
        for p in patterns:
            df[kw_col] = df[kw_col] | name_lower.str.contains(p, na=False)
        df[kw_col] = df[kw_col].astype(int)
        kw_cols.append(kw_col)

    # Aggregate text features
    df['name_len_chars'] = name_col.str.len().fillna(0).astype(float)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0).astype(float)
    df['name_kw_count'] = df[kw_cols].sum(axis=1).astype(float)

    # ---------- SPATIAL FEATURES ----------
    # Compute centroid from available lat/lon (no target usage)
    if df['latitude'].notnull().any() and df['longitude'].notnull().any():
        lat_mean = df['latitude'].mean()
        lon_mean = df['longitude'].mean()
    else:
        lat_mean = 0.0
        lon_mean = 0.0

    # Haversine distance to centroid
    def haversine(lat1, lon1, lat2, lon2):
        # convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371.0  # Earth radius in km
        return c * r

    df['dist_haversine_centroid'] = haversine(df['latitude'].fillna(lat_mean),
                                              df['longitude'].fillna(lon_mean),
                                              lat_mean, lon_mean)
    # Euclidean distance in lat/lon space (not physical distance but useful signal)
    df['dist_euclidean_centroid'] = np.sqrt((df['latitude'].fillna(lat_mean) - lat_mean) ** 2 +
                                            (df['longitude'].fillna(lon_mean) - lon_mean) ** 2)

    # ---------- RATIO & INTERACTION FEATURES ----------
    # Safe division helper
    def safe_divide(num, den):
        num = num.astype(float)
        den = den.astype(float)
        out = np.zeros_like(num, dtype=float)
        mask = den != 0
        out[mask] = num[mask] / den[mask]
        return out

    df['beds_per_bedroom'] = safe_divide(df['beds'].fillna(0), df['bedrooms'].fillna(0))
    df['accommodates_per_bed'] = safe_divide(df['accommodates'].fillna(0), df['beds'].fillna(0))
    df['accommodates_per_bedroom'] = safe_divide(df['accommodates'].fillna(0), df['bedrooms'].fillna(0))
    df['beds_per_accommodate'] = safe_divide(df['beds'].fillna(0), df['accommodates'].fillna(0))

    # Host experience
    df['host_tenure_days'] = df['host_tenure_days'].fillna(0)
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    df['host_listings_count'] = df['host_listings_count'].fillna(0)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))

    # Availability / revenue / reviews
    df['est_revenue'] = df['est_revenue'].fillna(0)
    df['total_reviews'] = df['total_reviews'].fillna(0)
    df['availability_rate'] = df['availability_rate'].fillna(0)

    df['revenue_per_review'] = safe_divide(df['est_revenue'], df['total_reviews'] + 1)
    df['revenue_per_available_day'] = safe_divide(df['est_revenue'], df['availability_rate'] * 365.0 + 1)
    df['reviews_per_year_proxy'] = safe_divide(df['total_reviews'], df['host_tenure_years'] + 0.1)

    # ---------- CATEGORICAL FREQUENCY ENCODING ----------
    def freq_encode(series):
        series = series.fillna('___MISSING___').astype(str)
        freq = series.value_counts(dropna=False)
        return series.map(freq).astype(float)

    # Basic frequency encodings
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
    if ('neighbourhood_group' in df.columns) and ('room_type' in df.columns):
        combo_ng_rt = (df['neighbourhood_group'].fillna('___MISSING___').astype(str) + '__' +
                       df['room_type'].fillna('___MISSING___').astype(str))
        df['neighbourhood_group_room_type_freq'] = freq_encode(combo_ng_rt)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if ('property_type' in df.columns) and ('room_type' in df.columns):
        combo_prop_rt = (df['property_type'].fillna('___MISSING___').astype(str) + '__' +
                         df['room_type'].fillna('___MISSING___').astype(str))
        df['property_room_type_freq'] = freq_encode(combo_prop_rt)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- BOOLEAN / OTHER CLEANUP ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].fillna(False).astype(int)

    # ---------- DROP NON-NUMERIC / UNUSED COLUMNS ----------
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=c)

    # Ensure only numeric columns (plus target if present)
    target_col = 'target_price'
    if target_col in df.columns:
        target_series = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target_series = None

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it was present
    if target_series is not None:
        df_numeric[target_col] = target_series

    # Fill any remaining NaNs with 0
    df_numeric = df_numeric.fillna(0)

    return df_numeric
