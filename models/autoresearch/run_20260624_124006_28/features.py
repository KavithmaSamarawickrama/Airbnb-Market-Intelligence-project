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

    keyword_list = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    total_kw_count = np.zeros(len(df), dtype=float)
    for kw in keyword_list:
        col_name = f"kw_{kw.replace(' ', '_')}"
        mask = name_lower.str.contains(kw, regex=False)
        df[col_name] = mask.astype(float)
        total_kw_count += df[col_name].values

    df['name_len_chars'] = name_col.str.len().fillna(0).astype(float)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0).astype(float)
    df['name_kw_total'] = total_kw_count

    # ---------- Spatial features ----------
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat = df['latitude'].astype(float)
        lon = df['longitude'].astype(float)
        lat_mean = lat.mean()
        lon_mean = lon.mean()

        # Euclidean distance in degrees (approximate)
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

    # ---------- Capacity & layout ratios ----------
    eps = 1e-6
    if 'beds' in df.columns and 'bedrooms' in df.columns:
        df['beds_per_bedroom'] = df['beds'] / (df['bedrooms'].replace(0, np.nan) + eps)
    if 'accommodates' in df.columns and 'beds' in df.columns:
        df['accommodates_per_bed'] = df['accommodates'] / (df['beds'].replace(0, np.nan) + eps)
    if 'accommodates' in df.columns and 'bedrooms' in df.columns:
        df['accommodates_per_bedroom'] = df['accommodates'] / (df['bedrooms'].replace(0, np.nan) + eps)
    if 'beds' in df.columns and 'accommodates' in df.columns:
        df['beds_per_accommodate'] = df['beds'] / (df['accommodates'].replace(0, np.nan) + eps)

    # ---------- Host behavior & experience ----------
    if 'host_tenure_days' in df.columns:
        df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    if 'host_listings_count' in df.columns:
        df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))

    # ---------- Demand / intensity proxies ----------
    if 'est_revenue' in df.columns and 'total_reviews' in df.columns:
        df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'] + 1.0)
    if 'est_revenue' in df.columns and 'availability_rate' in df.columns:
        df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'] * 365.0 + 1.0)
    if 'total_reviews' in df.columns and 'host_tenure_days' in df.columns:
        host_years = df['host_tenure_days'] / 365.0
        df['reviews_per_year_proxy'] = df['total_reviews'] / (host_years + 0.1)

    # ---------- Categorical frequency encoding ----------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Basic frequency encodings
    if 'room_type' in df.columns:
        df['room_type_freq'] = freq_encode(df['room_type'].astype(str))
    if 'property_type' in df.columns:
        df['property_type_freq'] = freq_encode(df['property_type'].astype(str))
    if 'neighbourhood' in df.columns:
        df['neighbourhood_freq'] = freq_encode(df['neighbourhood'].astype(str))
    if 'neighbourhood_group' in df.columns:
        df['neighbourhood_group_freq'] = freq_encode(df['neighbourhood_group'].astype(str))

    # Interaction frequency encodings
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        pair_ng_rt = df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(pair_ng_rt)
    if 'property_type' in df.columns and 'room_type' in df.columns:
        pair_prop_rt = df['property_type'].astype(str) + '||' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(pair_prop_rt)

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
        df = df.drop(columns=[target_col])
    else:
        target = None

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Reattach target if it was present
    if target is not None:
        numeric_df[target_col] = target

    # Fill remaining NaNs with 0
    numeric_df = numeric_df.fillna(0.0)

    return numeric_df
