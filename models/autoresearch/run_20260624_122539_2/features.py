def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # --- Basic cleanup ---
    # Ensure expected columns exist
    for col in ['accommodates', 'bedrooms', 'beds', 'latitude', 'longitude',
                'is_superhost', 'host_listings_count', 'availability_rate',
                'est_revenue', 'total_reviews', 'host_tenure_days']:
        if col not in df.columns:
            df[col] = np.nan

    # --- Text features from `name` ---
    if 'name' in df.columns:
        name = df['name'].fillna('').astype(str).str.lower()
    else:
        name = pd.Series([''] * len(df), index=df.index)

    # Basic length features
    df['name_len_chars'] = name.str.len()
    df['name_len_words'] = name.str.split().apply(len)

    # Keyword list (binary flags)
    keyword_map = {
        'kw_luxury': ['luxury', 'luxurious'],
        'kw_cozy': ['cozy', 'cosy'],
        'kw_apartment': ['apartment', 'apt'],
        'kw_studio': ['studio'],
        'kw_downtown': ['downtown'],
        'kw_central': ['central', 'city center', 'citycentre', 'city centre'],
        'kw_beach': ['beach'],
        'kw_sea': ['sea'],
        'kw_ocean': ['ocean'],
        'kw_view': ['view', 'views'],
        'kw_modern': ['modern'],
        'kw_spacious': ['spacious', 'large'],
        'kw_loft': ['loft'],
        'kw_penthouse': ['penthouse'],
        'kw_villa': ['villa'],
        'kw_pool': ['pool'],
        'kw_garden': ['garden'],
        'kw_old_town': ['old town'],
        'kw_new': ['new', 'brand new', 'newly'],
        'kw_renovated': ['renovated', 'refurbished'],
        'kw_balcony': ['balcony'],
        'kw_terrace': ['terrace']
    }

    keyword_cols = []
    for col_name, patterns in keyword_map.items():
        pattern_regex = '|'.join([pd.regex.escape(p) if hasattr(pd, 'regex') else p for p in patterns])
        # Use simple contains for robustness (no external regex module assumptions)
        df[col_name] = 0
        for p in patterns:
            df[col_name] = df[col_name] | name.str.contains(p, na=False)
        df[col_name] = df[col_name].astype(int)
        keyword_cols.append(col_name)

    # Aggregate keyword count
    if keyword_cols:
        df['name_keyword_count'] = df[keyword_cols].sum(axis=1)
    else:
        df['name_keyword_count'] = 0

    # --- Geographic features ---
    # Handle missing lat/lon
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Approximate city center as mean lat/lon of available points
    if df['latitude'].notnull().any() and df['longitude'].notnull().any():
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
    else:
        center_lat, center_lon = 0.0, 0.0

    # Euclidean distance in degrees (proxy for distance)
    df['dist_to_center'] = np.sqrt((df['latitude'] - center_lat) ** 2 + (df['longitude'] - center_lon) ** 2)

    # Polar coordinates relative to center
    df['rel_lat'] = df['latitude'] - center_lat
    df['rel_lon'] = df['longitude'] - center_lon
    df['dist_radial'] = np.sqrt(df['rel_lat'] ** 2 + df['rel_lon'] ** 2)
    df['angle_radial'] = np.arctan2(df['rel_lat'], df['rel_lon'])

    # --- Host features ---
    df['host_listings_count'] = pd.to_numeric(df['host_listings_count'], errors='coerce')
    df['host_listings_count'] = df['host_listings_count'].fillna(0)

    # Log-like bucket for host scale
    df['host_listings_bucket'] = np.log1p(df['host_listings_count'])

    # Host tenure in years
    df['host_tenure_days'] = pd.to_numeric(df['host_tenure_days'], errors='coerce')
    df['host_tenure_days'] = df['host_tenure_days'].fillna(0)
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0

    # Ensure is_superhost is numeric (0/1)
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)
    else:
        df['is_superhost'] = 0.0

    # --- Capacity and room features ---
    df['accommodates'] = pd.to_numeric(df['accommodates'], errors='coerce')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['beds'] = pd.to_numeric(df['beds'], errors='coerce')

    # Ratios with safe denominators
    df['beds_per_bedroom'] = df['beds'] / df['bedrooms'].replace({0: np.nan})
    df['beds_per_bedroom'] = df['beds_per_bedroom'].replace([np.inf, -np.inf], np.nan)

    df['accommodates_per_bed'] = df['accommodates'] / df['beds'].replace({0: np.nan})
    df['accommodates_per_bed'] = df['accommodates_per_bed'].replace([np.inf, -np.inf], np.nan)

    df['accommodates_per_bedroom'] = df['accommodates'] / df['bedrooms'].replace({0: np.nan})
    df['accommodates_per_bedroom'] = df['accommodates_per_bedroom'].replace([np.inf, -np.inf], np.nan)

    df['beds_per_accommodate'] = df['beds'] / df['accommodates'].replace({0: np.nan})
    df['beds_per_accommodate'] = df['beds_per_accommodate'].replace([np.inf, -np.inf], np.nan)

    # --- Availability and revenue structure ---
    df['availability_rate'] = pd.to_numeric(df['availability_rate'], errors='coerce')
    df['est_revenue'] = pd.to_numeric(df['est_revenue'], errors='coerce')
    df['total_reviews'] = pd.to_numeric(df['total_reviews'], errors='coerce')

    df['availability_rate'] = df['availability_rate'].fillna(0)
    df['est_revenue'] = df['est_revenue'].fillna(0)
    df['total_reviews'] = df['total_reviews'].fillna(0)

    # Revenue per review (baseline already used; keep and ensure robustness)
    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'] + 1.0)

    # Revenue per available day (avoid division by zero)
    df['revenue_per_available_day'] = df['est_revenue'] / (df['availability_rate'] * 365.0 + 1.0)

    # Reviews per year proxy
    df['reviews_per_year_proxy'] = df['total_reviews'] / (df['host_tenure_years'] + 0.1)

    # --- Categorical encodings (frequency-based) ---
    def freq_encode(series):
        # Frequency encoding using value counts (no target, so no leakage)
        vc = series.value_counts(dropna=False)
        return series.map(vc).astype(float)

    # room_type
    if 'room_type' in df.columns:
        df['room_type_freq'] = freq_encode(df['room_type'].astype(str))
    else:
        df['room_type_freq'] = 0.0

    # property_type
    if 'property_type' in df.columns:
        df['property_type_freq'] = freq_encode(df['property_type'].astype(str))
    else:
        df['property_type_freq'] = 0.0

    # neighbourhood
    if 'neighbourhood' in df.columns:
        df['neighbourhood_freq'] = freq_encode(df['neighbourhood'].astype(str))
    else:
        df['neighbourhood_freq'] = 0.0

    # neighbourhood_group
    if 'neighbourhood_group' in df.columns:
        df['neighbourhood_group_freq'] = freq_encode(df['neighbourhood_group'].astype(str))
    else:
        df['neighbourhood_group_freq'] = 0.0

    # Combined categorical interactions
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:
        combo_ng_rt = df['neighbourhood_group'].astype(str) + '|' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(combo_ng_rt)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    if 'property_type' in df.columns and 'room_type' in df.columns:
        combo_prop_rt = df['property_type'].astype(str) + '|' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(combo_prop_rt)
    else:
        df['property_room_type_freq'] = 0.0

    # --- Final cleanup ---
    # Drop non-numeric, ID, and raw text/date columns
    cols_to_drop = ['listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
                    'room_type', 'property_type', 'host_since']
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=c)

    # Ensure only numeric columns (plus target if present)
    target_col = 'target_price'
    cols = list(df.columns)
    if target_col in cols:
        feature_cols = [c for c in cols if c != target_col]
        # Keep only numeric features
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_features + [target_col]]
    else:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_features]

    # Fill remaining NaNs with 0 for model robustness
    df = df.fillna(0)

    return df
