def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure expected columns exist (create if missing to avoid KeyErrors)
    for col in [
        'accommodates', 'bedrooms', 'beds', 'latitude', 'longitude',
        'is_superhost', 'host_listings_count', 'availability_rate',
        'est_revenue', 'total_reviews', 'host_tenure_days'
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # ---------- Text features from `name` ----------
    name_col = 'name'
    if name_col in df.columns:
        name_series = df[name_col].fillna('').astype(str).str.lower()
    else:
        name_series = pd.Series([''] * len(df), index=df.index)

    # Basic text stats
    df['name_len_chars'] = name_series.str.len()
    df['name_len_words'] = name_series.str.split().apply(len)

    # Keyword indicators
    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    total_kw = np.zeros(len(df), dtype=float)
    for kw in keywords:
        col_name = f'kw_{kw.replace(" ", "_")}'
        mask = name_series.str.contains(kw, regex=False)
        df[col_name] = mask.astype(int)
        total_kw += df[col_name].values
    df['name_keyword_count'] = total_kw

    # ---------- Spatial features ----------
    # Global centroid
    lat_mean = df['latitude'].mean(skipna=True)
    lon_mean = df['longitude'].mean(skipna=True)

    # Euclidean distance to centroid (approx, in degrees)
    df['dist_euclid_centroid'] = np.sqrt(
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

    df['dist_haversine_centroid'] = haversine(
        df['latitude'].fillna(lat_mean),
        df['longitude'].fillna(lon_mean),
        lat_mean,
        lon_mean,
    )

    # ---------- Capacity & layout ratios ----------
    # Safe denominators
    bedrooms_safe = df['bedrooms'].replace(0, np.nan)
    beds_safe = df['beds'].replace(0, np.nan)
    accommodates_safe = df['accommodates'].replace(0, np.nan)

    # Existing-style ratios
    df['beds_per_bedroom'] = (df['beds'] / bedrooms_safe).fillna(0)
    df['accommodates_per_bed'] = (df['accommodates'] / beds_safe).fillna(0)

    # New ratios
    df['accommodates_per_bedroom'] = (df['accommodates'] / bedrooms_safe).fillna(0)
    df['beds_per_accommodate'] = (df['beds'] / accommodates_safe).fillna(0)

    # ---------- Host behavior & experience ----------
    df['host_tenure_days'] = df['host_tenure_days'].fillna(0)
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0

    df['host_listings_count'] = df['host_listings_count'].fillna(0)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'])

    # is_superhost as int
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Demand / intensity proxies ----------
    df['availability_rate'] = df['availability_rate'].fillna(0)
    df['est_revenue'] = df['est_revenue'].fillna(0)
    df['total_reviews'] = df['total_reviews'].fillna(0)

    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'] + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'] * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'] + 0.1)

    # ---------- Frequency encoding for categoricals ----------
    def add_freq_encoding(frame, col_name):
        if col_name not in frame.columns:
            frame[col_name] = np.nan
        freq = frame[col_name].value_counts(dropna=False)
        mapping = freq / len(frame)
        frame[f'{col_name}_freq'] = frame[col_name].map(mapping).astype(float)

    for cat_col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        add_freq_encoding(df, cat_col)

    # Interaction frequency encodings
    def add_pair_freq_encoding(frame, col_a, col_b, new_name):
        if col_a not in frame.columns:
            frame[col_a] = np.nan
        if col_b not in frame.columns:
            frame[col_b] = np.nan
        pair = list(zip(frame[col_a], frame[col_b]))
        pair_series = pd.Series(pair, index=frame.index)
        freq = pair_series.value_counts(dropna=False)
        mapping = freq / len(frame)
        frame[new_name] = pair_series.map(mapping).astype(float)

    add_pair_freq_encoding(df, 'neighbourhood_group', 'room_type', 'neighbourhood_group_room_type_freq')
    add_pair_freq_encoding(df, 'property_type', 'room_type', 'property_room_type_freq')

    # ---------- Clean up columns ----------
    cols_to_drop = ['listing_id', 'name', 'host_since']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Keep target if present; ensure all non-target columns are numeric
    target_col = 'target_price'
    cols = df.columns.tolist()
    if target_col in cols:
        feature_cols = [c for c in cols if c != target_col]
    else:
        feature_cols = cols

    # Convert non-numeric feature columns to numeric via factorization as fallback
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c], _ = pd.factorize(df[c], sort=True)
            df[c] = df[c].astype(float)

    # Final NaN handling
    df = df.fillna(0)

    return df
