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

    keyword_counts = []
    for kw in keywords:
        col_name = f'kw_{kw.replace(" ", "_")}'
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)
        keyword_counts.append(df[col_name])

    # Aggregate text stats
    df['name_len_chars'] = name_col.str.len()
    df['name_len_words'] = name_col.str.split().apply(len)
    if keyword_counts:
        df['name_keyword_count'] = np.vstack(keyword_counts).sum(axis=0)
    else:
        df['name_keyword_count'] = 0

    # ---------- Spatial features ----------
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat = df['latitude'].astype(float)
        lon = df['longitude'].astype(float)
        lat_mean = lat.mean()
        lon_mean = lon.mean()

        # Euclidean distance to global centroid (approx, in degrees)
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

        df['dist_haversine_centroid'] = haversine(lat, lon, lat_mean, lon_mean)
    else:
        df['dist_euclid_centroid'] = 0.0
        df['dist_haversine_centroid'] = 0.0

    # ---------- Capacity & layout ratios ----------
    accommodates = df['accommodates'] if 'accommodates' in df.columns else pd.Series(0, index=df.index)
    bedrooms = df['bedrooms'] if 'bedrooms' in df.columns else pd.Series(0.0, index=df.index)
    beds = df['beds'] if 'beds' in df.columns else pd.Series(0.0, index=df.index)

    # Existing-style ratios (recompute robustly)
    df['beds_per_bedroom'] = np.where((bedrooms > 0) & np.isfinite(bedrooms), beds / bedrooms, 0.0)
    df['accommodates_per_bed'] = np.where((beds > 0) & np.isfinite(beds), accommodates / beds, 0.0)

    # New ratios
    df['accommodates_per_bedroom'] = np.where((bedrooms > 0) & np.isfinite(bedrooms), accommodates / bedrooms, 0.0)
    df['beds_per_accommodate'] = np.where((accommodates > 0) & np.isfinite(accommodates), beds / accommodates, 0.0)

    # ---------- Host behavior & experience ----------
    if 'host_tenure_days' in df.columns:
        df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    else:
        df['host_tenure_years'] = 0.0

    if 'host_listings_count' in df.columns:
        df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))
    else:
        df['host_listings_log1p'] = 0.0

    # ---------- Demand / intensity proxies ----------
    est_rev = df['est_revenue'] if 'est_revenue' in df.columns else pd.Series(0.0, index=df.index)
    total_reviews = df['total_reviews'] if 'total_reviews' in df.columns else pd.Series(0.0, index=df.index)
    availability_rate = df['availability_rate'] if 'availability_rate' in df.columns else pd.Series(0.0, index=df.index)
    host_tenure_years = df['host_tenure_years']

    df['revenue_per_review'] = np.where((total_reviews + 1) > 0, est_rev / (total_reviews + 1), 0.0)
    df['revenue_per_available_day'] = np.where((availability_rate * 365 + 1) > 0,
                                               est_rev / (availability_rate * 365 + 1), 0.0)
    df['reviews_per_year_proxy'] = np.where((host_tenure_years + 0.1) > 0,
                                            total_reviews / (host_tenure_years + 0.1), 0.0)

    # ---------- Categorical frequency encoding ----------
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # Simple frequency encodings
    for cat_col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if cat_col in df.columns:
            df[f'{cat_col}_freq'] = freq_encode(df[cat_col].astype(str))
        else:
            df[f'{cat_col}_freq'] = 0.0

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

    # ---------- Boolean to numeric ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Drop non-numeric / identifier / raw text/date columns ----------
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target_price if present; ensure all other columns are numeric
    target_col = 'target_price'
    cols = list(df.columns)
    if target_col in cols:
        cols.remove(target_col)
        feature_cols = cols
    else:
        feature_cols = cols

    # Coerce to numeric (safety) and fill NaNs
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.fillna(0.0)

    return df
