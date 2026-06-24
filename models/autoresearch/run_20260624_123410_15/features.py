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
        name = df[name_col].fillna('').astype(str).str.lower()
    else:
        name = pd.Series([''] * len(df), index=df.index)

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name.str.contains(kw, regex=False).astype(int)

    # Aggregate text stats
    df['name_len_chars'] = name.str.len()
    df['name_len_words'] = name.str.split().str.len().fillna(0).astype(int)
    # total keyword count
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0

    # ---------- Spatial features ----------
    # Global centroid (using current df as approximation; non-leaky w.r.t. target)
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    # Euclidean distance to centroid (approximate)
    df['dist_euclid_centroid'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

    # Haversine distance to centroid (in km)
    def haversine(lat1, lon1, lat2, lon2):
        # convert to radians
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

    df['dist_haversine_centroid'] = haversine(lat, lon, lat_mean, lon_mean)

    # ---------- Capacity / layout ratios ----------
    def safe_div(num, den):
        return num / den.replace({0: np.nan})

    # beds_per_bedroom (already in champion, recompute robustly)
    df['beds_per_bedroom'] = safe_div(df['beds'], df['bedrooms'])

    # accommodates_per_bed (already in champion, recompute robustly)
    df['accommodates_per_bed'] = safe_div(df['accommodates'], df['beds'])

    # new ratios
    df['accommodates_per_bedroom'] = safe_div(df['accommodates'], df['bedrooms'])
    df['beds_per_accommodate'] = safe_div(df['beds'], df['accommodates'])

    # ---------- Host features ----------
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].fillna(0))

    # ---------- Demand / intensity proxies ----------
    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'].fillna(0) + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'].fillna(0) * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'].replace({0: np.nan}) + 0.1)

    # ---------- Frequency encoding for categoricals ----------
    def add_freq_encoding(frame, col_name, new_col):
        if col_name not in frame.columns:
            frame[new_col] = 0.0
            return
        freq = frame[col_name].value_counts(dropna=False)
        frame[new_col] = frame[col_name].map(freq).astype(float)

    add_freq_encoding(df, 'room_type', 'room_type_freq')
    add_freq_encoding(df, 'property_type', 'property_type_freq')
    add_freq_encoding(df, 'neighbourhood', 'neighbourhood_freq')
    add_freq_encoding(df, 'neighbourhood_group', 'neighbourhood_group_freq')

    # Interaction frequency encodings
    def add_pair_freq_encoding(frame, col_a, col_b, new_col):
        if col_a not in frame.columns or col_b not in frame.columns:
            frame[new_col] = 0.0
            return
        pair = frame[col_a].astype(str) + '||' + frame[col_b].astype(str)
        freq = pair.value_counts(dropna=False)
        frame[new_col] = pair.map(freq).astype(float)

    add_pair_freq_encoding(df, 'neighbourhood_group', 'room_type', 'neighbourhood_group_room_type_freq')
    add_pair_freq_encoding(df, 'property_type', 'room_type', 'property_room_type_freq')

    # ---------- Boolean to numeric ----------
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
    else:
        target = None

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Reattach target if it was present and got dropped by numeric filter
    if target is not None and target_col not in numeric_df.columns:
        numeric_df[target_col] = target

    # Fill remaining NaNs with 0
    numeric_df = numeric_df.fillna(0)

    return numeric_df
