def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # --- Drop obvious identifiers ---
    if 'listing_id' in df.columns:
        df = df.drop(columns=['listing_id'])

    # --- Basic safety: ensure expected columns exist ---
    # Fill missing numeric columns if absent
    numeric_defaults = {
        'accommodates': 0,
        'bedrooms': 0.0,
        'beds': 0.0,
        'latitude': 0.0,
        'longitude': 0.0,
        'is_superhost': 0,
        'host_listings_count': 0,
        'host_tenure_days': 0.0,
        'availability_rate': 0.0,
        'est_revenue': 0.0,
        'total_reviews': 0
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default

    # --- Text features from `name` ---
    name_col = df['name'].fillna('').astype(str) if 'name' in df.columns else pd.Series([''] * len(df), index=df.index)
    name_lower = name_col.str.lower()

    # Keyword list (binary flags)
    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'citycentre',
        'old town', 'new', 'renovated', 'balcony', 'terrace'
    ]

    kw_cols = []
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_lower.str.contains(kw, regex=False).astype(int)
        kw_cols.append(col_name)

    # Aggregate text features
    df['name_len_chars'] = name_col.str.len().fillna(0).astype(float)
    df['name_len_words'] = name_col.str.split().apply(len).fillna(0).astype(float)
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0

    # --- Spatial features ---
    # Ensure latitude/longitude are numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    lat_mean = df['latitude'].mean(skipna=True)
    lon_mean = df['longitude'].mean(skipna=True)

    # Euclidean distance to centroid (approximate)
    df['dist_euclid_centroid'] = np.sqrt(
        (df['latitude'] - lat_mean) ** 2 + (df['longitude'] - lon_mean) ** 2
    )

    # Haversine distance to centroid (in km)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in km
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df['dist_haversine_centroid_km'] = haversine(df['latitude'], df['longitude'], lat_mean, lon_mean)

    # --- Host and capacity interaction features ---
    # Safe ratios
    bedrooms = df['bedrooms'].replace(0, np.nan)
    beds = df['beds'].replace(0, np.nan)
    accommodates = df['accommodates'].replace(0, np.nan)

    df['beds_per_bedroom'] = (df['beds'] / bedrooms).replace([np.inf, -np.inf], np.nan)
    df['accommodates_per_bed'] = (df['accommodates'] / beds).replace([np.inf, -np.inf], np.nan)
    df['accommodates_per_bedroom'] = (df['accommodates'] / bedrooms).replace([np.inf, -np.inf], np.nan)
    df['beds_per_accommodate'] = (df['beds'] / accommodates).replace([np.inf, -np.inf], np.nan)

    # Host tenure in years
    df['host_tenure_years'] = df['host_tenure_days'] / 365.0

    # Host listings log1p
    df['host_listings_log1p'] = np.log1p(df['host_listings_count'].clip(lower=0))

    # --- Availability, revenue, and review intensity ---
    df['revenue_per_review'] = df['est_revenue'] / (df['total_reviews'].fillna(0) + 1.0)

    # Avoid division by zero for availability_rate
    avail_days = df['availability_rate'].fillna(0) * 365.0
    df['revenue_per_available_day'] = df['est_revenue'] / (avail_days + 1.0)

    # Reviews per year proxy
    tenure_years_safe = df['host_tenure_years'].replace(0, np.nan)
    df['reviews_per_year_proxy'] = (df['total_reviews'] / (tenure_years_safe + 0.1)).replace([np.inf, -np.inf], np.nan)

    # --- Categorical frequency encoding ---
    def freq_encode(series):
        # frequency (relative) encoding
        vc = series.value_counts(dropna=False)
        freq = vc / len(series) if len(series) > 0 else vc
        return series.map(freq).astype(float)

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

    # Interaction frequencies
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

    # --- Binary encoding for is_superhost if boolean ---
    if 'is_superhost' in df.columns:
        if df['is_superhost'].dtype == bool:
            df['is_superhost'] = df['is_superhost'].astype(int)
        else:
            # handle 't'/'f' or similar
            df['is_superhost'] = df['is_superhost'].map({True: 1, False: 0, 't': 1, 'f': 0}).fillna(0).astype(int)

    # --- Drop raw non-numeric columns ---
    cols_to_drop = ['name', 'room_type', 'property_type', 'neighbourhood', 'neighbourhood_group', 'host_since']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # --- Keep only numeric columns (plus target_price if present) ---
    target_col = 'target_price'
    cols = list(df.columns)
    if target_col in cols:
        target_series = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target_series = None

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Reattach target if it existed
    if target_series is not None:
        df_numeric[target_col] = target_series

    # --- Final NaN handling ---
    df_numeric = df_numeric.fillna(0)

    return df_numeric
