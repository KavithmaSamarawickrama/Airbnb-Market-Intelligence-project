def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure target is preserved if present
    target_col = 'target_price'
    has_target = target_col in df.columns
    target_series = df[target_col] if has_target else None

    # -----------------------------
    # Text features from `name`
    # -----------------------------
    name_col = 'name'
    if name_col in df.columns:
        name_series = df[name_col].fillna('').astype(str).str.lower()

        # Keyword list (expanded beyond champion)
        keywords = [
            'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
            'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
            'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
            'new', 'renovated', 'balcony', 'terrace'
        ]

        # Binary indicators for each keyword
        for kw in keywords:
            col_name = f"kw_{kw.replace(' ', '_')}"
            df[col_name] = name_series.str.contains(kw, regex=False).astype(int)

        # Aggregate text features
        df['name_len_chars'] = name_series.str.len()
        df['name_len_words'] = name_series.str.split().apply(len)
        # Total keyword count per listing
        kw_cols = [c for c in df.columns if c.startswith('kw_')]
        if kw_cols:
            df['name_kw_count'] = df[kw_cols].sum(axis=1)
        else:
            df['name_kw_count'] = 0

    # -----------------------------
    # Spatial features
    # -----------------------------
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat = df['latitude'].astype(float)
        lon = df['longitude'].astype(float)

        # Global centroid (using available rows only)
        lat_mean = lat.mean()
        lon_mean = lon.mean()

        # Euclidean distance to centroid (approx, degrees space)
        df['dist_euclid_centroid'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

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
            r = 6371.0  # Radius of earth in kilometers
            return c * r

        df['dist_haversine_centroid_km'] = haversine(lat, lon, lat_mean, lon_mean)

    # -----------------------------
    # Capacity & layout ratios
    # -----------------------------
    for col in ['accommodates', 'bedrooms', 'beds']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # beds_per_bedroom (champion feature, recompute robustly)
    if 'beds' in df.columns and 'bedrooms' in df.columns:
        denom = df['bedrooms'].replace(0, np.nan)
        df['beds_per_bedroom'] = df['beds'] / denom

    # accommodates_per_bed (champion feature, recompute robustly)
    if 'accommodates' in df.columns and 'beds' in df.columns:
        denom = df['beds'].replace(0, np.nan)
        df['accommodates_per_bed'] = df['accommodates'] / denom

    # New: accommodates_per_bedroom
    if 'accommodates' in df.columns and 'bedrooms' in df.columns:
        denom = df['bedrooms'].replace(0, np.nan)
        df['accommodates_per_bedroom'] = df['accommodates'] / denom

    # New: beds_per_accommodate
    if 'beds' in df.columns and 'accommodates' in df.columns:
        denom = df['accommodates'].replace(0, np.nan)
        df['beds_per_accommodate'] = df['beds'] / denom

    # -----------------------------
    # Host behavior & experience
    # -----------------------------
    if 'host_tenure_days' in df.columns:
        df['host_tenure_days'] = pd.to_numeric(df['host_tenure_days'], errors='coerce')
        df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    else:
        df['host_tenure_years'] = np.nan

    if 'host_listings_count' in df.columns:
        df['host_listings_count'] = pd.to_numeric(df['host_listings_count'], errors='coerce')
        df['host_listings_log1p'] = np.log1p(df['host_listings_count'])

    # -----------------------------
    # Demand / intensity proxies
    # -----------------------------
    for col in ['est_revenue', 'total_reviews', 'availability_rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'est_revenue' in df.columns and 'total_reviews' in df.columns:
        df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'] + 1.0)

    if 'est_revenue' in df.columns and 'availability_rate' in df.columns:
        df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'] * 365.0 + 1.0)

    if 'total_reviews' in df.columns:
        # host_tenure_years may be NaN if original column missing; handle later with fillna
        df['reviews_per_year_proxy'] = df['total_reviews'] / (df.get('host_tenure_years', 0) + 0.1)

    # -----------------------------
    # Categorical frequency encoding
    # -----------------------------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Base categoricals
    cat_cols = ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']
    for col in cat_cols:
        if col in df.columns:
            df[f'{col}_freq'] = freq_encode(df[col].astype(str))

    # Interaction frequency encodings
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        combo = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(combo)

    if 'property_type' in df.columns and 'room_type' in df.columns:
        combo = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(combo)

    # -----------------------------
    # Drop non-numeric / leakage-prone columns
    # -----------------------------
    cols_to_drop = ['listing_id', 'name', 'host_since']
    # Also drop raw categoricals after encoding
    cols_to_drop += ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']

    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # -----------------------------
    # Keep only numeric columns (plus target if present)
    # -----------------------------
    # Temporarily reattach target if we saved it
    if has_target and target_col not in df.columns:
        df[target_col] = target_series

    # Separate target, select numeric, then reattach target
    if has_target:
        target_series = df[target_col]
        df = df.drop(columns=[target_col])

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Reattach target if present
    if has_target:
        numeric_df[target_col] = target_series

    # -----------------------------
    # Handle missing values (simple baseline: fill with 0)
    # -----------------------------
    numeric_df = numeric_df.fillna(0)

    return numeric_df
