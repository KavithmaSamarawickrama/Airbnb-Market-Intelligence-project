def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure expected columns exist
    for col in ['accommodates', 'bedrooms', 'beds', 'latitude', 'longitude',
                'is_superhost', 'host_listings_count', 'availability_rate',
                'est_revenue', 'total_reviews', 'host_tenure_days']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---------- Text features from `name` ----------
    name_col = df['name'].fillna('').astype(str) if 'name' in df.columns else pd.Series([''] * len(df), index=df.index)
    name_lower = name_col.str.lower()

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    for kw in keywords:
        col_name = f'kw_{kw.replace(" ", "_")}'
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(float)

    # Aggregate text stats
    df['name_len_chars'] = name_col.str.len().fillna(0).astype(float)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0).astype(float)
    # total keyword count
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0.0

    # ---------- Spatial features ----------
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat = df['latitude'].astype(float)
        lon = df['longitude'].astype(float)
        lat_mean = lat.mean()
        lon_mean = lon.mean()

        # Euclidean distance to global centroid (approx, degrees space)
        df['dist_euclid_centroid'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

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

        df['dist_haversine_centroid_km'] = haversine(lat, lon, lat_mean, lon_mean)
    else:
        df['dist_euclid_centroid'] = 0.0
        df['dist_haversine_centroid_km'] = 0.0

    # ---------- Capacity & layout ratios ----------
    def safe_div(num, den):
        num = num.astype(float)
        den = den.astype(float)
        return np.where((den == 0) | np.isnan(den), 0.0, num / den)

    if 'beds' in df.columns and 'bedrooms' in df.columns:
        df['beds_per_bedroom'] = safe_div(df['beds'], df['bedrooms'])
    else:
        df['beds_per_bedroom'] = 0.0

    if 'accommodates' in df.columns and 'beds' in df.columns:
        df['accommodates_per_bed'] = safe_div(df['accommodates'], df['beds'])
    else:
        df['accommodates_per_bed'] = 0.0

    if 'accommodates' in df.columns and 'bedrooms' in df.columns:
        df['accommodates_per_bedroom'] = safe_div(df['accommodates'], df['bedrooms'])
    else:
        df['accommodates_per_bedroom'] = 0.0

    if 'beds' in df.columns and 'accommodates' in df.columns:
        df['beds_per_accommodate'] = safe_div(df['beds'], df['accommodates'])
    else:
        df['beds_per_accommodate'] = 0.0

    # ---------- Host features ----------
    if 'host_tenure_days' in df.columns:
        df['host_tenure_years'] = df['host_tenure_days'].astype(float) / 365.0
    else:
        df['host_tenure_years'] = 0.0

    if 'host_listings_count' in df.columns:
        df['host_listings_log1p'] = np.log1p(df['host_listings_count'].astype(float))
    else:
        df['host_listings_log1p'] = 0.0

    # ---------- Demand / intensity proxies ----------
    if 'est_revenue' in df.columns and 'total_reviews' in df.columns:
        df['revenue_per_review'] = safe_div(df['est_revenue'], df['total_reviews'] + 1.0)
    else:
        df['revenue_per_review'] = 0.0

    if 'est_revenue' in df.columns and 'availability_rate' in df.columns:
        df['revenue_per_available_day'] = safe_div(df['est_revenue'], df['availability_rate'] * 365.0 + 1.0)
    else:
        df['revenue_per_available_day'] = 0.0

    if 'total_reviews' in df.columns:
        df['reviews_per_year_proxy'] = safe_div(df['total_reviews'], df['host_tenure_years'] + 0.1)
    else:
        df['reviews_per_year_proxy'] = 0.0

    # ---------- Categorical frequency encoding ----------
    def add_freq_encoding(frame, col_name, new_col):
        if col_name not in frame.columns:
            frame[new_col] = 0.0
            return
        freq = frame[col_name].fillna('NA').value_counts(normalize=True)
        frame[new_col] = frame[col_name].fillna('NA').map(freq).astype(float).fillna(0.0)

    add_freq_encoding(df, 'room_type', 'room_type_freq')
    add_freq_encoding(df, 'property_type', 'property_type_freq')
    add_freq_encoding(df, 'neighbourhood', 'neighbourhood_freq')
    add_freq_encoding(df, 'neighbourhood_group', 'neighbourhood_group_freq')

    # Interaction frequency encodings
    def add_pair_freq_encoding(frame, col_a, col_b, new_col):
        if col_a not in frame.columns or col_b not in frame.columns:
            frame[new_col] = 0.0
            return
        pair = frame[col_a].fillna('NA').astype(str) + '||' + frame[col_b].fillna('NA').astype(str)
        freq = pair.value_counts(normalize=True)
        frame[new_col] = pair.map(freq).astype(float).fillna(0.0)

    add_pair_freq_encoding(df, 'neighbourhood_group', 'room_type', 'neighbourhood_group_room_type_freq')
    add_pair_freq_encoding(df, 'property_type', 'room_type', 'property_room_type_freq')

    # ---------- Boolean to numeric ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Keep target_price if present (training pipeline will handle it separately)
    target_col = None
    if 'target_price' in df.columns:
        target_col = df['target_price']
        df = df.drop(columns=['target_price'])

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it existed
    if target_col is not None:
        numeric_df['target_price'] = target_col.astype(float)

    # Fill remaining NaNs with 0
    numeric_df = numeric_df.fillna(0.0)

    return numeric_df
