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
        name_series = df[name_col].fillna('').astype(str).str.lower()
    else:
        name_series = pd.Series([''] * len(df), index=df.index)

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_series.str.contains(kw, regex=False).astype(int)

    # Aggregate text stats
    df['name_len_chars'] = name_series.str.len()
    df['name_len_words'] = name_series.str.split().apply(len)
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    df['name_kw_count'] = df[kw_cols].sum(axis=1) if kw_cols else 0

    # ---------- Spatial features ----------
    # Global centroid (dataset-level, no target usage)
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    # Euclidean distance to centroid (approx, degrees)
    df['dist_euclid_centroid'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

    # Haversine distance to centroid (km)
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

    # ---------- Capacity & layout ratios ----------
    accommodates = df['accommodates'].astype(float)
    bedrooms = df['bedrooms'].astype(float)
    beds = df['beds'].astype(float)

    # Existing-style ratios
    df['beds_per_bedroom'] = np.where((bedrooms > 0) & np.isfinite(bedrooms), beds / bedrooms, 0.0)
    df['accommodates_per_bed'] = np.where((beds > 0) & np.isfinite(beds), accommodates / beds, 0.0)

    # New ratios
    df['accommodates_per_bedroom'] = np.where((bedrooms > 0) & np.isfinite(bedrooms), accommodates / bedrooms, 0.0)
    df['beds_per_accommodate'] = np.where((accommodates > 0) & np.isfinite(accommodates), beds / accommodates, 0.0)

    # ---------- Host behavior & experience ----------
    df['host_tenure_days'] = df['host_tenure_days'].astype(float).fillna(0.0)
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0

    df['host_listings_count'] = df['host_listings_count'].astype(float).fillna(0.0)
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'])

    # ---------- Demand / intensity proxies ----------
    df['est_revenue'] = df['est_revenue'].astype(float).fillna(0.0)
    df['total_reviews'] = df['total_reviews'].astype(float).fillna(0.0)
    df['availability_rate'] = df['availability_rate'].astype(float).fillna(0.0)

    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'] + 1.0)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'] * 365.0 + 1.0)
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'] + 0.1)

    # ---------- Categorical frequency encoding ----------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Basic categoricals
    for cat_col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')
            df[f'{cat_col}_freq'] = freq_encode(df[cat_col])
        else:
            df[f'{cat_col}_freq'] = 0.0

    # Interaction frequency encodings
    if ('neighbourhood_group' in df.columns) and ('room_type' in df.columns):
        pair = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(pair)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if ('property_type' in df.columns) and ('room_type' in df.columns):
        pair2 = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(pair2)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Binary casting for is_superhost ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float).fillna(0.0)

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=c)

    # Keep target if present
    target_col = 'target_price'
    if target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target = None

    # Ensure only numeric columns remain
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it existed
    if target is not None:
        numeric_df[target_col] = target

    # Fill any remaining NaNs with 0
    numeric_df = numeric_df.fillna(0.0)

    return numeric_df
