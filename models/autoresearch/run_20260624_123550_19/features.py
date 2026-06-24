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

    keyword_cols = []
    for kw in keywords:
        col_name = f"kw_{kw.replace(' ', '_')}"
        df[col_name] = name_series.str.contains(kw, regex=False).astype(int)
        keyword_cols.append(col_name)

    # Aggregate text stats
    df['name_len_chars'] = name_series.str.len()
    df['name_len_words'] = name_series.str.split().apply(len)
    df['name_keyword_count'] = df[keyword_cols].sum(axis=1)

    # ---------- Spatial features ----------
    # Fill lat/lon NaNs with column means for centroid computation
    lat = df['latitude'].astype(float)
    lon = df['longitude'].astype(float)
    lat_mean = lat.mean(skipna=True)
    lon_mean = lon.mean(skipna=True)

    lat_filled = lat.fillna(lat_mean)
    lon_filled = lon.fillna(lon_mean)

    # Euclidean distance to global centroid (approximate)
    df['dist_euclid_centroid'] = np.sqrt((lat_filled - lat_mean) ** 2 + (lon_filled - lon_mean) ** 2)

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

    df['dist_haversine_centroid_km'] = haversine(lat_filled, lon_filled, lat_mean, lon_mean)

    # ---------- Capacity & layout ratios ----------
    accommodates = df['accommodates'].astype(float)
    bedrooms = df['bedrooms'].astype(float)
    beds = df['beds'].astype(float)

    # Avoid division by zero using where
    df['beds_per_bedroom'] = np.where(bedrooms > 0, beds / bedrooms, 0.0)
    df['accommodates_per_bed'] = np.where(beds > 0, accommodates / beds, 0.0)
    df['accommodates_per_bedroom'] = np.where(bedrooms > 0, accommodates / bedrooms, 0.0)
    df['beds_per_accommodate'] = np.where(accommodates > 0, beds / accommodates, 0.0)

    # ---------- Host features ----------
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

    # ---------- Categorical frequency encodings ----------
    def freq_encode(series):
        counts = series.value_counts(dropna=False)
        return series.map(counts).astype(float)

    # Base categoricals
    for col in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[f'{col}_freq'] = freq_encode(df[col])
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
        inter_prop_rt = df['property_type'].astype(str) + '|' + df['room_type'].astype(str)
        df['property_room_type_freq'] = freq_encode(inter_prop_rt)
    else:
        df['property_room_type_freq'] = 0.0

    # ---------- Binary / numeric cleanup ----------
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # ---------- Drop non-numeric / ID / raw text / date columns ----------
    cols_to_drop = [
        'listing_id', 'name', 'host_since',
        'room_type', 'property_type', 'neighbourhood', 'neighbourhood_group'
    ]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_to_drop, errors='ignore')

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
