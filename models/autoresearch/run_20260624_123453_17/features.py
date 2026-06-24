def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # --- Basic cleaning ---
    # Ensure expected columns exist
    for col in [
        'accommodates', 'bedrooms', 'beds', 'latitude', 'longitude',
        'is_superhost', 'host_listings_count', 'availability_rate',
        'est_revenue', 'total_reviews', 'host_tenure_days'
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # --- Text features from `name` ---
    if 'name' in df.columns:
        name = df['name'].fillna('').astype(str).str.lower()
    else:
        name = pd.Series([''] * len(df), index=df.index)

    # Aggregate text stats
    df['name_len_chars'] = name.str.len()
    df['name_len_words'] = name.str.split().apply(len)

    # Keyword indicators
    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]
    kw_cols = []
    for kw in keywords:
        col_name = f'kw_{kw.replace(" ", "_")}'
        df[col_name] = name.str.contains(kw, regex=False).astype(int)
        kw_cols.append(col_name)

    # Total keyword count
    df['name_kw_count'] = df[kw_cols].sum(axis=1)

    # --- Spatial features ---
    # Global centroid
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

    df['dist_haversine_centroid'] = haversine(
        df['latitude'], df['longitude'], lat_mean, lon_mean
    )

    # --- Capacity & layout ratios ---
    # Safe denominators
    bedrooms_safe = df['bedrooms'].replace(0, np.nan)
    beds_safe = df['beds'].replace(0, np.nan)
    accommodates_safe = df['accommodates'].replace(0, np.nan)

    # Existing-style ratios
    df['beds_per_bedroom'] = beds_safe / bedrooms_safe
    df['accommodates_per_bed'] = df['accommodates'] / beds_safe

    # New ratios
    df['accommodates_per_bedroom'] = df['accommodates'] / bedrooms_safe
    df['beds_per_accommodate'] = df['beds'] / accommodates_safe

    # --- Host behavior & experience ---
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].fillna(0))

    # --- Demand / intensity proxies ---
    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'].fillna(0) + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'].fillna(0) * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'].fillna(0) / (df['host_tenure_years'].fillna(0) + 0.1)

    # --- Categorical frequency encoding ---
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Base categoricals
    cat_cols = []
    if 'room_type' in df.columns:
        df['room_type_freq'] = freq_encode(df['room_type'])
        cat_cols.append('room_type')
    if 'property_type' in df.columns:
        df['property_type_freq'] = freq_encode(df['property_type'])
        cat_cols.append('property_type')
    if 'neighbourhood' in df.columns:
        df['neighbourhood_freq'] = freq_encode(df['neighbourhood'])
        cat_cols.append('neighbourhood')
    if 'neighbourhood_group' in df.columns:
        df['neighbourhood_group_freq'] = freq_encode(df['neighbourhood_group'])
        cat_cols.append('neighbourhood_group')

    # Interaction frequency encodings
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(pair)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(pair2)
    else:
        df['property_room_type_freq'] = 0.0

    # --- Boolean to numeric ---
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # --- Drop non-numeric / ID / raw text/date columns ---
    cols_to_drop = [
        'listing_id', 'name', 'host_since',
        'room_type', 'property_type', 'neighbourhood', 'neighbourhood_group'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Keep target_price if present; it will be handled outside
    # Ensure all remaining features are numeric
    for col in df.columns:
        if col == 'target_price':
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill remaining NaNs with 0
    df = df.fillna(0)

    return df
