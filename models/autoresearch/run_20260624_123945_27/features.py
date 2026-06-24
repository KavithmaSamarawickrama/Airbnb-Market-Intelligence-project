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

    # ---------- TEXT FEATURES FROM name ----------
    if 'name' in df.columns:
        name = df['name'].fillna('').astype(str).str.lower()

        keywords = [
            'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
            'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
            'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
            'new', 'renovated', 'balcony', 'terrace'
        ]

        # Simple keyword flags (substring search)
        for kw in keywords:
            col_name = f"kw_{kw.replace(' ', '_')}"
            df[col_name] = name.str.contains(kw, regex=False).astype(int)

        # Aggregate text stats
        df['name_len_chars'] = name.str.len()
        df['name_len_words'] = name.str.split().str.len().fillna(0).astype(int)

        # Total keyword count per listing
        kw_cols = [c for c in df.columns if c.startswith('kw_')]
        if kw_cols:
            df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        # If name missing, create empty placeholders to keep schema stable
        df['name_len_chars'] = 0
        df['name_len_words'] = 0
        df['name_kw_count'] = 0

    # ---------- SPATIAL FEATURES ----------
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat = df['latitude'].astype(float)
        lon = df['longitude'].astype(float)

        lat_mean = lat.mean()
        lon_mean = lon.mean()

        # Euclidean distance to global centroid (approx, degrees space)
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

        df['dist_haversine_centroid_km'] = haversine(lat, lon, lat_mean, lon_mean)
    else:
        df['dist_euclid_centroid'] = 0.0
        df['dist_haversine_centroid_km'] = 0.0

    # ---------- CAPACITY / LAYOUT RATIOS ----------
    # Existing-style ratios
    if {'beds', 'bedrooms'}.issubset(df.columns):
        denom = df['bedrooms'].replace({0: np.nan})
        df['beds_per_bedroom'] = (df['beds'] / denom).replace([np.inf, -np.inf], np.nan)
    else:
        df['beds_per_bedroom'] = np.nan

    if {'accommodates', 'beds'}.issubset(df.columns):
        denom = df['beds'].replace({0: np.nan})
        df['accommodates_per_bed'] = (df['accommodates'] / denom).replace([np.inf, -np.inf], np.nan)
    else:
        df['accommodates_per_bed'] = np.nan

    # New ratios
    if {'accommodates', 'bedrooms'}.issubset(df.columns):
        denom = df['bedrooms'].replace({0: np.nan})
        df['accommodates_per_bedroom'] = (df['accommodates'] / denom).replace([np.inf, -np.inf], np.nan)
    else:
        df['accommodates_per_bedroom'] = np.nan

    if {'beds', 'accommodates'}.issubset(df.columns):
        denom = df['accommodates'].replace({0: np.nan})
        df['beds_per_accommodate'] = (df['beds'] / denom).replace([np.inf, -np.inf], np.nan)
    else:
        df['beds_per_accommodate'] = np.nan

    # ---------- HOST BEHAVIOR / EXPERIENCE ----------
    if 'host_tenure_days' in df.columns:
        df['host_tenure_years'] = df['host_tenure_days'] / 365.0
    else:
        df['host_tenure_years'] = np.nan

    if 'host_listings_count' in df.columns:
        df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))
    else:
        df['host_listings_log1p'] = 0.0

    # ---------- DEMAND / INTENSITY PROXIES ----------
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
        tenure_years = df.get('host_tenure_years', pd.Series(0, index=df.index)).fillna(0)
        df['reviews_per_year_proxy'] = df['total_reviews'].fillna(0) / (tenure_years + 0.1)
    else:
        df['reviews_per_year_proxy'] = np.nan

    # ---------- CATEGORICAL FREQUENCY ENCODING ----------
    def add_freq_encoding(frame, col_name, new_col):
        if col_name in frame.columns:
            freq = frame[col_name].fillna('NA').value_counts(normalize=False)
            frame[new_col] = frame[col_name].fillna('NA').map(freq).astype(float)
        else:
            frame[new_col] = 0.0

    # Single-column frequencies
    add_freq_encoding(df, 'room_type', 'room_type_freq')
    add_freq_encoding(df, 'property_type', 'property_type_freq')
    add_freq_encoding(df, 'neighbourhood', 'neighbourhood_freq')
    add_freq_encoding(df, 'neighbourhood_group', 'neighbourhood_group_freq')

    # Interaction frequencies
    if {'neighbourhood_group', 'room_type'}.issubset(df.columns):
        pair = df['neighbourhood_group'].fillna('NA').astype(str) + '||' + df['room_type'].fillna('NA').astype(str)
        freq_pair = pair.value_counts()
        df['neighbourhood_group_room_type_freq'] = pair.map(freq_pair).astype(float)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if {'property_type', 'room_type'}.issubset(df.columns):
        pair2 = df['property_type'].fillna('NA').astype(str) + '||' + df['room_type'].fillna('NA').astype(str)
        freq_pair2 = pair2.value_counts()
        df['property_room_type_freq'] = pair2.map(freq_pair2).astype(float)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- BOOLEAN TO NUMERIC ----------
    if 'is_superhost' in df.columns:
        if df['is_superhost'].dtype == bool:
            df['is_superhost'] = df['is_superhost'].astype(int)
        else:
            # Map common string representations
            df['is_superhost'] = df['is_superhost'].map({True: 1, False: 0, 't': 1, 'f': 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

    # ---------- DROP NON-NUMERIC / UNUSED COLUMNS ----------
    cols_to_drop = ['listing_id', 'name', 'host_since', 'room_type', 'property_type',
                    'neighbourhood', 'neighbourhood_group']
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

    # Keep target if present; ensure all other columns are numeric
    target_col = 'target_price'
    cols = list(df.columns)
    if target_col in cols:
        feature_cols = [c for c in cols if c != target_col]
        df_features = df[feature_cols]
        df_target = df[[target_col]]
    else:
        df_features = df
        df_target = None

    # Convert non-numeric columns to numeric where possible, then drop any remaining non-numeric
    for c in df_features.columns:
        if not np.issubdtype(df_features[c].dtype, np.number):
            df_features[c] = pd.to_numeric(df_features[c], errors='coerce')

    # After coercion, drop any still-non-numeric columns
    numeric_cols = [c for c in df_features.columns if np.issubdtype(df_features[c].dtype, np.number)]
    df_features = df_features[numeric_cols]

    # Reattach target if it existed
    if df_target is not None:
        df = pd.concat([df_features, df_target], axis=1)
    else:
        df = df_features

    # ---------- HANDLE MISSING VALUES ----------
    df = df.fillna(0)

    return df
