def feature_engineering(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Drop identifier
    if 'listing_id' in df.columns:
        df = df.drop(columns=['listing_id'])

    # Ensure target is preserved if present
    target_col = 'target_price'
    has_target = target_col in df.columns
    target_series = df[target_col] if has_target else None

    # --- Simple text features from name ---
    name_col = 'name'
    if name_col in df.columns:
        name = df[name_col].fillna('').astype(str).str.lower()
        keywords = {
            'kw_luxury': ['luxury', 'luxurious'],
            'kw_cozy': ['cozy', 'cosy'],
            'kw_apartment': ['apartment', 'apt'],
            'kw_studio': ['studio'],
            'kw_downtown': ['downtown', 'city center', 'city centre', 'center', 'centre'],
            'kw_beach': ['beach', 'seaside', 'ocean'],
            'kw_view': ['view', 'views'],
            'kw_modern': ['modern', 'contemporary'],
        }
        for new_col, patterns in keywords.items():
            pattern_regex = '|'.join([pd.regex.escape(p) if hasattr(pd, 'regex') else p for p in patterns])
            # Use simple contains for compatibility
            df[new_col] = 0
            for p in patterns:
                df[new_col] = df[new_col] | name.str.contains(p, na=False)
            df[new_col] = df[new_col].astype(int)
        # Drop raw text column
        df = df.drop(columns=[name_col])

    # --- Drop raw datetime column (we have host_tenure_days) ---
    if 'host_since' in df.columns:
        df = df.drop(columns=['host_since'])

    # --- Frequency encoding for categoricals ---
    cat_cols = []
    for c in ['room_type', 'property_type', 'neighbourhood', 'neighbourhood_group']:
        if c in df.columns:
            cat_cols.append(c)

    for c in cat_cols:
        # Compute frequency of each category
        freq = df[c].value_counts(dropna=False)
        freq_map = freq / len(df)
        new_col = f"{c}_freq"
        df[new_col] = df[c].map(freq_map).fillna(0).astype(float)

    # Drop original categorical columns
    df = df.drop(columns=cat_cols, errors='ignore')

    # --- Derived ratio features ---
    # beds_per_bedroom
    if 'beds' in df.columns and 'bedrooms' in df.columns:
        beds = df['beds'].astype(float)
        bedrooms = df['bedrooms'].astype(float)
        denom = bedrooms.replace(0, np.nan)
        df['beds_per_bedroom'] = (beds / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # accommodates_per_bed
    if 'accommodates' in df.columns and 'beds' in df.columns:
        accommodates = df['accommodates'].astype(float)
        beds = df['beds'].astype(float)
        denom = beds.replace(0, np.nan)
        df['accommodates_per_bed'] = (accommodates / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # revenue_per_review
    if 'est_revenue' in df.columns and 'total_reviews' in df.columns:
        est_rev = df['est_revenue'].astype(float)
        reviews = df['total_reviews'].astype(float)
        df['revenue_per_review'] = (est_rev / (reviews + 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- Boolean to numeric ---
    if 'is_superhost' in df.columns:
        df['is_superhost'] = df['is_superhost'].astype(float)

    # --- Ensure all remaining non-numeric columns are dropped ---
    # Temporarily reattach target to avoid dropping it
    if has_target:
        df[target_col] = target_series

    # Select numeric columns plus target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if has_target and target_col not in numeric_cols:
        numeric_cols.append(target_col)

    df_numeric = df[numeric_cols].copy()

    # --- Handle missing values: simple fill with 0 ---
    df_numeric = df_numeric.fillna(0.0)

    return df_numeric
