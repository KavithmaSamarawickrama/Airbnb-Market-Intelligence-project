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
    name_col = df['name'].fillna('').astype(str) if 'name' in df.columns else pd.Series([''] * len(df), index=df.index)
    name_lower = name_col.str.lower()

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)

    # Aggregate text stats
    df['name_len_chars'] = name_col.str.len().fillna(0).astype(float)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0).astype(float)
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0.0

    # ---------- Spatial features ----------
    # Global centroid (mean lat/lon)
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    df['lat_center'] = lat_mean
    df['lon_center'] = lon_mean

    # Euclidean distance to centroid (approximate)
    df['dist_euclid_center'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

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

    df['dist_haversine_center_km'] = haversine(lat, lon, lat_mean, lon_mean)

    # ---------- Host & capacity interaction features ----------
    bedrooms = df['bedrooms'].astype(float)
    beds = df['beds'].astype(float)
    accommodates = df['accommodates'].astype(float)

    def safe_div(num, den):
        den_safe = den.replace({0: np.nan}) if isinstance(den, pd.Series) else den
        res = num / den_safe
        return res.replace([np.inf, -np.inf], np.nan)

    df['beds_per_bedroom'] = safe_div(beds, bedrooms)
    df['accommodates_per_bed'] = safe_div(accommodates, beds)
    df['accommodates_per_bedroom'] = safe_div(accommodates, bedrooms)
    df['beds_per_accommodate'] = safe_div(beds, accommodates)

    # Host tenure in years and log host listings
    df['host_tenure_years'] = safe_div(df['host_tenure_days'], 365.0)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].fillna(0))

    # ---------- Availability, revenue, and review intensity ----------
    est_rev = df['est_revenue'].astype(float)
    total_reviews = df['total_reviews'].astype(float)
    avail_rate = df['availability_rate'].astype(float)

    df['revenue_per_review'] = safe_div(est_rev, total_reviews + 1.0)
    df['revenue_per_available_day'] = safe_div(est_rev, avail_rate * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = safe_div(total_reviews, df['host_tenure_years'] + 0.1)

    # ---------- Categorical frequency encoding ----------
    cat_cols = []
    for c in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if c in df.columns:
            cat_cols.append(c)

    for c in cat_cols:
        freq = df[c].value_counts(dropna=False)
        df[f'{c}_freq'] = df[c].map(freq).astype(float)

    # Interaction frequency encodings
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        freq_pair = pair.value_counts(dropna=False)
        df['neighbourhood_group_room_type_freq'] = pair.map(freq_pair).astype(float)
    else:
        df['neighbourhood_group_room_type_freq'] = np.nan

    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        freq_pair2 = pair2.value_counts(dropna=False)
        df['property_room_type_freq'] = pair2.map(freq_pair2).astype(float)
    else:
        df['property_room_type_freq'] = np.nan

    # ---------- Basic boolean/numeric cleanup ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target if present
    target = None
    if 'target_price' in df.columns:
        target = df['target_price']
        df = df.drop(columns=['target_price'])

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it existed
    if target is not None:
        numeric_df['target_price'] = target

    # Fill remaining NaNs with 0
    numeric_df = numeric_df.fillna(0)

    return numeric_df
