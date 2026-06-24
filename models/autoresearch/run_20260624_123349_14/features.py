def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure expected columns exist
    for col in [
        'accommodates', 'bedrooms', 'beds', 'latitude', 'longitude',
        'is_superhost', 'host_listings_count', 'availability_rate',
        'est_revenue', 'total_reviews', 'host_tenure_days'
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # ---------- Text features from `name` ----------
    name = df['name'].fillna('').astype(str)
    name_lower = name.str.lower()

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    keyword_counts = []
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)
        keyword_counts.append(df[col_name])

    # Aggregate text stats
    df['name_len_chars'] = name.str.len()
    df['name_len_words'] = name.str.split().str.len().fillna(0).astype(int)
    df['name_keyword_count'] = np.sum(keyword_counts, axis=0) if keyword_counts else 0

    # ---------- Spatial features ----------
    # Global centroid (non-leaky, uses only feature distribution)
    lat_mean = df['latitude'].mean(skipna=True)
    lon_mean = df['longitude'].mean(skipna=True)

    # Euclidean distance to centroid (approximate)
    df['dist_euclid_center'] = np.sqrt(
        (df['latitude'] - lat_mean) ** 2 + (df['longitude'] - lon_mean) ** 2
    )

    # Haversine distance to centroid (in km)
    def _haversine(lat1, lon1, lat2, lon2):
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

    df['dist_haversine_center'] = _haversine(df['latitude'], df['longitude'], lat_mean, lon_mean)

    # ---------- Capacity & layout ratios ----------
    def safe_div(num, den):
        return num / den.replace({0: np.nan})

    # beds_per_bedroom
    df['beds_per_bedroom'] = safe_div(df['beds'], df['bedrooms'])
    # accommodates_per_bed
    df['accommodates_per_bed'] = safe_div(df['accommodates'], df['beds'])
    # accommodates_per_bedroom
    df['accommodates_per_bedroom'] = safe_div(df['accommodates'], df['bedrooms'])
    # beds_per_accommodate
    df['beds_per_accommodate'] = safe_div(df['beds'], df['accommodates'])

    # ---------- Host behavior features ----------
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].fillna(0))

    # ---------- Demand / intensity proxies ----------
    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'].fillna(0) + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'].fillna(0) * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'].fillna(0) + 0.1)

    # ---------- Frequency encoding for categoricals ----------
    cat_cols = ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']
    for col in cat_cols:
        if col in df.columns:
            freq = df[col].value_counts(dropna=False)
            df[f'{col}_freq'] = df[col].map(freq).fillna(0).astype(float)
        else:
            df[f'{col}_freq'] = 0.0

    # Interaction frequency encodings
    # neighbourhood_group x room_type
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        pair_freq = pair.value_counts(dropna=False)
        df['neighbourhood_group_room_type_freq'] = pair.map(pair_freq).fillna(0).astype(float)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    # property_type x room_type
    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        pair2_freq = pair2.value_counts(dropna=False)
        df['property_room_type_freq'] = pair2.map(pair2_freq).fillna(0).astype(float)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Basic numeric clean-up ----------
    # Ensure boolean is numeric
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # Drop non-numeric / identifier / raw text / date columns
    cols_to_drop = [
        'listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
        'room_type', 'property_type', 'host_since'
    ]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target_price if present; it will be handled outside
    # Ensure only numeric columns (plus target_price if present)
    target_col = 'target_price'
    if target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target = None

    df = df.apply(pd.to_numeric, errors='coerce')

    # Re-attach target if it existed
    if target is not None:
        df[target_col] = target

    # Fill remaining NaNs with 0
    df = df.fillna(0)

    return df
