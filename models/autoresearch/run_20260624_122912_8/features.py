def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure expected columns exist
    for col in [
        'listing_id', 'name', 'room_type', 'property_type', 'accommodates',
        'bedrooms', 'beds', 'latitude', 'longitude', 'is_superhost',
        'host_listings_count', 'host_since', 'host_tenure_days',
        'neighbourhood', 'neighbourhood_group', 'availability_rate',
        'est_revenue', 'total_reviews', 'target_price'
    ]:
        if col not in df.columns:
            # Don't create target_price if missing; pipeline handles it
            if col == 'target_price':
                continue
            df[col] = np.nan

    # ---------- Text features from `name` ----------
    name = df['name'].fillna('').astype(str)
    name_lower = name.str.lower()

    # Keyword list
    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    # Binary flags for each keyword
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)

    # Aggregate text stats
    df['name_len_chars'] = name.str.len().fillna(0)
    df['name_len_words'] = name.str.split().apply(len).astype(float)

    # Total keyword count
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0

    # ---------- Spatial features ----------
    # Handle lat/lon safely
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)

    # Global centroid (using available rows)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    df['lat_center'] = lat_mean
    df['lon_center'] = lon_mean

    # Haversine distance to centroid (in km)
    def haversine(lat1, lon1, lat2, lon2):
        # Convert degrees to radians
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

    df['dist_center_haversine_km'] = haversine(lat, lon, lat_mean, lon_mean)

    # Simple Euclidean distance in degree space (proxy)
    df['dist_center_euclidean'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

    # ---------- Host & capacity interaction features ----------
    df['accommodates'] = df['accommodates'].astype(float)
    df['bedrooms'] = df['bedrooms'].astype(float)
    df['beds'] = df['beds'].astype(float)

    # Safe division helpers
    def safe_div(numer, denom):
        numer = numer.astype(float)
        denom = denom.astype(float)
        res = numer / denom.replace({0: np.nan})
        return res.replace([np.inf, -np.inf], np.nan)

    df['beds_per_bedroom'] = safe_div(df['beds'], df['bedrooms'])
    df['accommodates_per_bed'] = safe_div(df['accommodates'], df['beds'])
    df['accommodates_per_bedroom'] = safe_div(df['accommodates'], df['bedrooms'])
    df['beds_per_accommodate'] = safe_div(df['beds'], df['accommodates'])

    # Host features
    df['host_tenure_days'] = df['host_tenure_days'].astype(float)
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0

    df['host_listings_count'] = df['host_listings_count'].astype(float)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))

    # is_superhost as int
    if df['is_superhost'].dtype == bool:
        df['is_superhost'] = df['is_superhost'].astype(int)
    else:
        df['is_superhost'] = df['is_superhost'].fillna(False).astype(int)

    # ---------- Availability, revenue, reviews ----------
    df['availability_rate'] = df['availability_rate'].astype(float)
    df['est_revenue'] = df['est_revenue'].astype(float)
    df['total_reviews'] = df['total_reviews'].astype(float)

    df['revenue_per_review'] = safe_div(df['est_revenue'], df['total_reviews'] + 1.0)
    df['revenue_per_available_day'] = safe_div(df['est_revenue'], df['availability_rate'] * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = safe_div(df['total_reviews'], df['host_tenure_years'] + 0.1)

    # ---------- Frequency encoding for categoricals ----------
    cat_cols = ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']
    for c in cat_cols:
        if c in df.columns:
            freq = df[c].value_counts(dropna=False)
            df[f'{c}_freq'] = df[c].map(freq).astype(float)
        else:
            df[f'{c}_freq'] = 0.0

    # Interaction frequency encodings
    # neighbourhood_group x room_type
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        combo = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        freq_combo = combo.value_counts(dropna=False)
        df['neighbourhood_group_room_type_freq'] = combo.map(freq_combo).astype(float)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    # property_type x room_type
    if 'property_type' in df.columns and 'room_type' in df.columns:
        combo2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        freq_combo2 = combo2.value_counts(dropna=False)
        df['property_room_type_freq'] = combo2.map(freq_combo2).astype(float)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = [
        'listing_id', 'name', 'room_type', 'property_type', 'neighbourhood',
        'neighbourhood_group', 'host_since'
    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Keep target_price if present; it will be separated by the training pipeline
    # Ensure all remaining features except target_price are numeric
    for col in df.columns:
        if col == 'target_price':
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill remaining NaNs with 0
    df = df.fillna(0)

    return df
