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

    # ---------- TEXT FEATURES FROM `name` ----------
    name_col = df['name'].fillna('').astype(str) if 'name' in df.columns else pd.Series([''] * len(df), index=df.index)
    name_lower = name_col.str.lower()

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central', 'beach',
        'sea', 'ocean', 'view', 'modern', 'spacious', 'loft', 'penthouse',
        'villa', 'pool', 'garden', 'city center', 'old town', 'new',
        'renovated', 'balcony', 'terrace'
    ]

    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)

    # Aggregate text features
    df['name_len_chars'] = name_col.str.len()
    df['name_len_words'] = name_col.str.split().apply(len)
    kw_cols = [c for c in df.columns if c.startswith('kw_')]
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0

    # ---------- SPATIAL FEATURES ----------
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat = df['latitude'].astype(float)
        lon = df['longitude'].astype(float)
        lat_mean = lat.mean()
        lon_mean = lon.mean()

        # Euclidean distance to global centroid (approximate)
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

        df['dist_haversine_center'] = haversine(lat, lon, lat_mean, lon_mean)
    else:
        df['dist_euclid_center'] = 0.0
        df['dist_haversine_center'] = 0.0

    # ---------- CAPACITY & LAYOUT RATIOS ----------
    def safe_div(num, den):
        num = num.astype(float)
        den = den.astype(float)
        return num / den.replace(0, np.nan)

    if {'beds', 'bedrooms'}.issubset(df.columns):
        df['beds_per_bedroom'] = safe_div(df['beds'], df['bedrooms'])
    else:
        df['beds_per_bedroom'] = np.nan

    if {'accommodates', 'beds'}.issubset(df.columns):
        df['accommodates_per_bed'] = safe_div(df['accommodates'], df['beds'])
    else:
        df['accommodates_per_bed'] = np.nan

    if {'accommodates', 'bedrooms'}.issubset(df.columns):
        df['accommodates_per_bedroom'] = safe_div(df['accommodates'], df['bedrooms'])
    else:
        df['accommodates_per_bedroom'] = np.nan

    if {'beds', 'accommodates'}.issubset(df.columns):
        df['beds_per_accommodate'] = safe_div(df['beds'], df['accommodates'])
    else:
        df['beds_per_accommodate'] = np.nan

    # ---------- HOST FEATURES ----------
    if 'host_tenure_days' in df.columns:
        df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    else:
        df['host_tenure_years'] = np.nan

    if 'host_listings_count' in df.columns:
        df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))
    else:
        df['host_listings_log1p'] = 0.0

    # ---------- DEMAND / INTENSITY FEATURES ----------
    if {'est_revenue', 'total_reviews'}.issubset(df.columns):
        df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'].fillna(0) + 1.0)
    else:
        df['revenue_per_review'] = np.nan

    if {'est_revenue', 'availability_rate'}.issubset(df.columns):
        avail_days = df['availability_rate'].fillna(0) * 365.0
        df['revenue_per_available_day'] = df['est_revenue'] / (avail_days + 1.0)
    else:
        df['revenue_per_available_day'] = np.nan

    if 'total_reviews' in df.columns:
        tenure_years = df.get('host_tenure_years', pd.Series(np.nan, index=df.index)).fillna(0)
        df['reviews_per_year_proxy'] = df['total_reviews'] / (tenure_years + 0.1)
    else:
        df['reviews_per_year_proxy'] = np.nan

    # ---------- CATEGORICAL FREQUENCY ENCODING ----------
    def add_freq_encoding(frame, col_name, new_col):
        if col_name in frame.columns:
            freq = frame[col_name].fillna('NA').value_counts(normalize=True)
            frame[new_col] = frame[col_name].fillna('NA').map(freq).astype(float)
        else:
            frame[new_col] = 0.0

    add_freq_encoding(df, 'room_type', 'room_type_freq')
    add_freq_encoding(df, 'property_type', 'property_type_freq')
    add_freq_encoding(df, 'neighbourhood', 'neighbourhood_freq')
    add_freq_encoding(df, 'neighbourhood_group', 'neighbourhood_group_freq')

    # Interaction frequency encodings
    if {'neighbourhood_group', 'room_type'}.issubset(df.columns):
        combo = df['neighbourhood_group'].fillna('NA').astype(str) + '|' + df['room_type'].fillna('NA').astype(str)
        freq_combo = combo.value_counts(normalize=True)
        df['neighbourhood_group_room_type_freq'] = combo.map(freq_combo).astype(float)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if {'property_type', 'room_type'}.issubset(df.columns):
        combo2 = df['property_type'].fillna('NA').astype(str) + '|' + df['room_type'].fillna('NA').astype(str)
        freq_combo2 = combo2.value_counts(normalize=True)
        df['property_room_type_freq'] = combo2.map(freq_combo2).astype(float)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- BOOLEAN TO NUMERIC ----------
    if 'is_superhost' in df.columns:
        if df['is_superhost'].dtype == bool:
            df['is_superhost'] = df['is_superhost'].astype(int)
        else:
            # handle strings like 't'/'f' or 'True'/'False'
            df['is_superhost'] = df['is_superhost'].map({True: 1, False: 0, 't': 1, 'f': 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

    # ---------- DROP NON-NUMERIC / UNUSED COLUMNS ----------
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=c)

    # Keep target if present
    target = None
    if 'target_price' in df.columns:
        target = df['target_price']
        df = df.drop(columns=['target_price'])

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it existed
    if target is not None:
        numeric_df['target_price'] = target

    # Fill remaining NaNs with 0
    numeric_df = numeric_df.fillna(0)

    return numeric_df
