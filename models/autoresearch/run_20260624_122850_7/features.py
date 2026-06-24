def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure expected columns exist (create if missing to avoid KeyErrors)
    expected_cols = [
        'listing_id', 'name', 'room_type', 'property_type', 'accommodates',
        'bedrooms', 'beds', 'latitude', 'longitude', 'is_superhost',
        'host_listings_count', 'host_since', 'host_tenure_days',
        'neighbourhood', 'neighbourhood_group', 'availability_rate',
        'est_revenue', 'total_reviews', 'target_price'
    ]
    for c in expected_cols:
        if c not in df.columns:
            # create neutral defaults for missing columns
            if c == 'is_superhost':
                df[c] = False
            elif c in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group', 'name', 'host_since']:
                df[c] = ''
            else:
                df[c] = 0.0

    # --- Text features from `name` ---
    name = df['name'].fillna('').astype(str).str.lower()

    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]

    # Binary flags for each keyword
    for kw in keywords:
        col_name = f'kw_{kw.replace(" ", "_")}'
        df[col_name] = name.str.contains(kw, regex=False).astype(int)

    # Aggregate text features
    df['name_len_chars'] = name.str.len()
    df['name_len_words'] = name.str.split().str.len().fillna(0).astype(float)
    kw_cols = [f'kw_{kw.replace(" ", "_")}' for kw in keywords]
    df['name_kw_count'] = df[kw_cols].sum(axis=1)

    # --- Spatial features ---
    # Use available lat/lon; handle missing with mean later
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)

    # Global centroid (mean lat/lon over current df)
    lat_mean = lat.replace(0, np.nan).mean()
    lon_mean = lon.replace(0, np.nan).mean()
    if np.isnan(lat_mean):
        lat_mean = 0.0
    if np.isnan(lon_mean):
        lon_mean = 0.0

    # Haversine distance to centroid
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # km
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df['dist_haversine_to_centroid'] = haversine(lat, lon, lat_mean, lon_mean)

    # Euclidean distance (rough, in degrees)
    df['dist_euclidean_to_centroid'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

    # --- Host and capacity interaction features ---
    df['accommodates'] = pd.to_numeric(df['accommodates'], errors='coerce')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['beds'] = pd.to_numeric(df['beds'], errors='coerce')

    # Safe denominators
    bedrooms_safe = df['bedrooms'].replace(0, np.nan)
    beds_safe = df['beds'].replace(0, np.nan)
    accommodates_safe = df['accommodates'].replace(0, np.nan)

    df['beds_per_bedroom'] = (df['beds'] / bedrooms_safe).replace([np.inf, -np.inf], np.nan)
    df['accommodates_per_bed'] = (df['accommodates'] / beds_safe).replace([np.inf, -np.inf], np.nan)
    df['accommodates_per_bedroom'] = (df['accommodates'] / bedrooms_safe).replace([np.inf, -np.inf], np.nan)
    df['beds_per_accommodate'] = (df['beds'] / accommodates_safe).replace([np.inf, -np.inf], np.nan)

    # Host tenure and scale
    df['host_tenure_days'] = pd.to_numeric(df['host_tenure_days'], errors='coerce')
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0

    df['host_listings_count'] = pd.to_numeric(df['host_listings_count'], errors='coerce')
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))

    # --- Availability, revenue, reviews ---
    df['availability_rate'] = pd.to_numeric(df['availability_rate'], errors='coerce')
    df['est_revenue'] = pd.to_numeric(df['est_revenue'], errors='coerce')
    df['total_reviews'] = pd.to_numeric(df['total_reviews'], errors='coerce')

    df['revenue_per_review'] = (df['est_revenue'] / (df['total_reviews'] + 1.0)).replace([np.inf, -np.inf], np.nan)
    df['revenue_per_available_day'] = (
        df['est_revenue'] / (df['availability_rate'].clip(lower=0) * 365.0 + 1.0)
    ).replace([np.inf, -np.inf], np.nan)

    df['reviews_per_year_proxy'] = (
        df['total_reviews'] / (df['host_tenure_years'].abs() + 0.1)
    ).replace([np.inf, -np.inf], np.nan)

    # --- Categorical frequency encoding ---
    def freq_encode(series):
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    for col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if col in df.columns:
            df[col] = df[col].fillna('')
            df[f'{col}_freq'] = freq_encode(df[col].astype(str))

    # Interaction frequencies
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        combo_ng_rt = (df['neighbourhood_group'].astype(str) + '||' + df['room_type'].astype(str))
        df['neighbourhood_group_room_type_freq'] = freq_encode(combo_ng_rt)

    if 'property_type' in df.columns and 'room_type' in df.columns:
        combo_prop_rt = (df['property_type'].astype(str) + '||' + df['room_type'].astype(str))
        df['property_room_type_freq'] = freq_encode(combo_prop_rt)

    # --- Boolean to numeric ---
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # --- Drop non-numeric / identifier / raw text/date columns ---
    cols_to_drop = [
        'listing_id', 'name', 'host_since',
        'room_type', 'property_type', 'neighbourhood', 'neighbourhood_group'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Ensure only numeric columns (plus target if present)
    target_col = 'target_price'
    if target_col in df.columns:
        target = df[target_col]
        df_numeric = df.drop(columns=[target_col])
        df_numeric = df_numeric.select_dtypes(include=[np.number])
        df_numeric[target_col] = target
    else:
        df_numeric = df.select_dtypes(include=[np.number])

    # Fill remaining NaNs with 0
    df_numeric = df_numeric.fillna(0.0)

    return df_numeric
