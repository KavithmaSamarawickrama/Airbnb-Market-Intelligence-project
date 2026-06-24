def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Ensure expected columns exist
    for col in [
        'accommodates', 'bedrooms', 'beds', 'latitude', 'longitude',
        'is_superhost', 'host_listings_count', 'availability_rate',
        'est_revenue', 'total_reviews', 'host_tenure_days'
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # ---------- Text features from `name` ----------
    if 'name' in df.columns:
        name = df['name'].fillna('').astype(str).str.lower()
    else:
        name = pd.Series([''] * len(df), index=df.index)

    # Basic text stats
    df['name_len_chars'] = name.str.len()
    df['name_len_words'] = name.str.split().apply(len)

    # Keyword indicators
    keywords = [
        'luxury', 'cozy', 'apartment', 'studio', 'downtown', 'central',
        'beach', 'sea', 'ocean', 'view', 'modern', 'spacious', 'loft',
        'penthouse', 'villa', 'pool', 'garden', 'city center', 'old town',
        'new', 'renovated', 'balcony', 'terrace'
    ]
    kw_cols = []
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name.str.contains(kw, regex=False).astype(int)
        kw_cols.append(col_name)

    # Total keyword count
    if kw_cols:
        df['name_kw_count'] = df[kw_cols].sum(axis=1)
    else:
        df['name_kw_count'] = 0

    # ---------- Spatial features ----------
    # Global centroid (non-leaky: uses only feature distribution)
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    # Euclidean distance to centroid (approximate)
    df['dist_euclid_centroid'] = np.sqrt((lat - lat_mean) ** 2 + (lon - lon_mean) ** 2)

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

    df['dist_haversine_centroid'] = haversine(lat, lon, lat_mean, lon_mean)

    # ---------- Capacity & layout ratios ----------
    accommodates = df['accommodates'].astype(float)
    bedrooms = df['bedrooms'].astype(float)
    beds = df['beds'].astype(float)

    # Champion ratios (recompute robustly)
    df['beds_per_bedroom'] = np.where((bedrooms > 0) & np.isfinite(bedrooms), beds / bedrooms, 0.0)
    df['accommodates_per_bed'] = np.where((beds > 0) & np.isfinite(beds), accommodates / beds, 0.0)

    # New ratios
    df['accommodates_per_bedroom'] = np.where((bedrooms > 0) & np.isfinite(bedrooms), accommodates / bedrooms, 0.0)
    df['beds_per_accommodate'] = np.where((accommodates > 0) & np.isfinite(accommodates), beds / accommodates, 0.0)

    # ---------- Host features ----------
    host_tenure_days = df['host_tenure_days'].astype(float)
    df['host_tenure_years'] = host_tenure_days / 365.0

    host_listings = df['host_listings_count'].astype(float).fillna(0)
    df['host_listings_log1p'] = np.log1p(np.clip(host_listings, a_min=0, a_max=None))

    # ---------- Demand / intensity proxies ----------
    est_revenue = df['est_revenue'].astype(float).fillna(0)
    total_reviews = df['total_reviews'].astype(float).fillna(0)
    availability_rate = df['availability_rate'].astype(float).fillna(0)

    # Revenue per review
    df['revenue_per_review'] = est_revenue / (total_reviews + 1.0)

    # Revenue per available day (avoid division by zero)
    df['revenue_per_available_day'] = est_revenue / (availability_rate * 365.0 + 1.0)

    # Reviews per year proxy
    host_tenure_years = df['host_tenure_years'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['reviews_per_year_proxy'] = total_reviews / (host_tenure_years + 0.1)

    # ---------- Categorical frequency encodings ----------
    def freq_encode(series):
        # Map each category to its relative frequency
        vc = series.value_counts(dropna=False)
        freq = vc / vc.sum()
        return series.map(freq).astype(float)

    # Base categoricals
    for col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if col in df.columns:
            df[f'{col}_freq'] = freq_encode(df[col].astype(str))
        else:
            df[f'{col}_freq'] = 0.0

    # Interaction: neighbourhood_group x room_type
    if ('neighbourhood_group' in df.columns) and ('room_type' in df.columns):
        inter_ng_rt = df['neighbourhood_group'].astype(str) + '|' + df['room_type'].astype(str)
        df['neighbourhood_group_room_type_freq'] = freq_encode(inter_ng_rt)
    else:
        df['neighbourhood_group_room_type_freq'] = 0.0

    # Interaction: property_type x room_type
    if ('property_type' in df.columns) and ('room_type' in df.columns):
        inter_pt_rt = df['property_type'].astype(str) + '|' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(inter_pt_rt)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Basic boolean handling ----------
    if 'is_superhost' in df.columns:
        # Convert boolean to 0/1 if needed
        if df['is_superhost'].dtype == bool:
            df['is_superhost'] = df['is_superhost'].astype(int)
        else:
            # Coerce to numeric, treating non-numeric as NaN
            df['is_superhost'] = pd.to_numeric(df['is_superhost'], errors='coerce').fillna(0).astype(int)

    # ---------- Drop non-numeric / ID / raw text/date columns ----------
    cols_to_drop = [
        'listing_id', 'name', 'neighbourhood', 'neighbourhood_group',
        'room_type', 'property_type', 'host_since'
    ]
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=c)

    # Keep target_price if present; it will be handled outside
    # Ensure only numeric columns remain (plus target_price if present)
    target_col = 'target_price'
    if target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])
    else:
        target = None

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Re-attach target if it existed
    if target is not None:
        df_numeric[target_col] = target

    # Fill any remaining NaNs with 0
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_numeric
