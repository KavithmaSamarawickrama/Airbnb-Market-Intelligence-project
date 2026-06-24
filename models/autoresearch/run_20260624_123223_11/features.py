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

    keyword_counts = []
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_series.str.contains(kw, regex=False).astype(int)
        keyword_counts.append(df[col_name])

    if keyword_counts:
        df['name_keyword_count'] = np.vstack(keyword_counts).sum(axis=0)
    else:
        df['name_keyword_count'] = 0

    df['name_len_chars'] = name_series.str.len()
    df['name_len_words'] = name_series.str.split().apply(len)

    # ---------- SPATIAL FEATURES ----------
    # Global centroid (using available rows only)
    if df['latitude'].notnull().any() and df['longitude'].notnull().any():
        lat_mean = df['latitude'].mean()
        lon_mean = df['longitude'].mean()
    else:
        lat_mean, lon_mean = 0.0, 0.0

    df['lat_center'] = lat_mean
    df['lon_center'] = lon_mean

    # Euclidean distance to centroid (approximate)
    df['dist_euclid_center'] = np.sqrt(
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
        r = 6371  # Radius of earth in kilometers
        return r * c

    df['dist_haversine_center'] = haversine(df['latitude'], df['longitude'], lat_mean, lon_mean)

    # ---------- CAPACITY & LAYOUT RATIOS ----------
    def safe_divide(num, den):
        return np.where(den == 0, np.nan, num / den)

    df['beds_per_bedroom'] = safe_divide(df['beds'], df['bedrooms'])
    df['accommodates_per_bed'] = safe_divide(df['accommodates'], df['beds'])
    df['accommodates_per_bedroom'] = safe_divide(df['accommodates'], df['bedrooms'])
    df['beds_per_accommodate'] = safe_divide(df['beds'], df['accommodates'])

    # ---------- HOST FEATURES ----------
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].fillna(0))

    # ---------- DEMAND / INTENSITY FEATURES ----------
    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'].fillna(0) + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'].fillna(0) * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'].replace(0, np.nan) + 0.1)

    # ---------- CATEGORICAL FREQUENCY ENCODING ----------
    cat_cols = []
    for c in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if c in df.columns:
            cat_cols.append(c)

    # Basic frequency encoding
    for c in cat_cols:
        freq = df[c].value_counts(dropna=False)
        df[f'{c}_freq'] = df[c].map(freq).fillna(0).astype(float)

    # Interaction frequency encodings
    # (neighbourhood_group, room_type)
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair = list(zip(df['neighbourhood_group'], df['room_type']))
        pair_series = pd.Series(pair, index=df.index)
        pair_freq = pair_series.value_counts(dropna=False)
        df['neighbourhood_group_room_type_freq'] = pair_series.map(pair_freq).fillna(0).astype(float)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    # (property_type, room_type)
    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair2 = list(zip(df['property_type'], df['room_type']))
        pair2_series = pd.Series(pair2, index=df.index)
        pair2_freq = pair2_series.value_counts(dropna=False)
        df['property_room_type_freq'] = pair2_series.map(pair2_freq).fillna(0).astype(float)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- CLEANUP: DROP NON-NUMERIC / UNUSED COLUMNS ----------
    cols_to_drop = ['listing_id', 'name', 'host_since', 'room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']
    for c in cols_to_drop:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Keep target_price if present; ensure all other columns are numeric
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
