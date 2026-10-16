"""
Data loader and preprocessing utilities for the Water Quality Dashboard.
Handles file loading, encoding fallbacks, missing value standardization,
and dataset summary calculations.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Standard expected features in order of model usage
NUMERIC_FEATURE_COLS = [
    'Temp',
    'D.O. (mg/l)',
    'PH',
    'CONDUCTIVITY (\u00b5mhos/cm)',
    'B.O.D. (mg/l)',
    'NITRATENAN N+ NITRITENANN (mg/l)',
    'FECAL COLIFORM (MPN/100ml)',
    'TOTAL COLIFORM (MPN/100ml)Mean'
]

# Standard metadata columns
META_COLS = ['STATION CODE', 'LOCATIONS', 'STATE', 'year']


def find_conductivity_column(df: pd.DataFrame) -> Optional[str]:
    """Finds the conductivity column regardless of micro-character encoding."""
    for col in df.columns:
        if 'CONDUCTIVITY' in col.upper():
            return col
    return None


@st.cache_data(show_spinner="Loading water monitoring dataset...")
def load_raw_dataset(csv_path: str = "water_dataX.csv") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Loads the water dataset with encoding fallback handling.
    Returns (DataFrame, error_message).
    """
    if not os.path.exists(csv_path):
        return None, f"Dataset file '{csv_path}' not found in working directory."

    encodings_to_try = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
    df = None
    last_err = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            logger.info(f"Successfully loaded '{csv_path}' with encoding '{enc}'.")
            break
        except Exception as e:
            last_err = str(e)
            continue

    if df is None:
        return None, f"Failed to decode '{csv_path}'. Last error: {last_err}"

    return df, None


def clean_water_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names, cleans missing value representations ('NAN', 'NA', etc.),
    and converts numeric measurements to float64.
    """
    df = df_raw.copy()

    # Standardize conductivity column name for consistent programmatic access
    cond_col = find_conductivity_column(df)
    if cond_col and cond_col != 'CONDUCTIVITY (\u00b5mhos/cm)':
        # Keep original mapping in mind, but normalize internally to match model_features.json
        df = df.rename(columns={cond_col: 'CONDUCTIVITY (\u00b5mhos/cm)'})

    # Standard missing string representations in environmental datasets
    nan_strings = ['NAN', 'NA', 'nan', 'null', 'None', '', ' ', '?', '#N/A']

    # Identify numeric candidate columns
    numeric_targets = [
        'Temp',
        'D.O. (mg/l)',
        'PH',
        'CONDUCTIVITY (\u00b5mhos/cm)',
        'B.O.D. (mg/l)',
        'NITRATENAN N+ NITRITENANN (mg/l)',
        'FECAL COLIFORM (MPN/100ml)',
        'TOTAL COLIFORM (MPN/100ml)Mean'
    ]

    for col in numeric_targets:
        if col in df.columns:
            # Clean string replacements
            df[col] = df[col].astype(str).str.strip().replace(nan_strings, np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert year if present
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Clean text columns
    for text_col in ['STATION CODE', 'LOCATIONS', 'STATE']:
        if text_col in df.columns:
            df[text_col] = df[text_col].astype(str).str.strip().replace(nan_strings, np.nan)

    return df


@st.cache_data
def load_feature_list(json_path: str = "model_features.json") -> Tuple[Optional[List[str]], Optional[str]]:
    """Loads expected feature list from JSON metadata."""
    if not os.path.exists(json_path):
        return None, f"Feature metadata file '{json_path}' not found."

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        return features, None
    except Exception as e:
        return None, f"Failed to load '{json_path}': {e}"


def compute_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates missing value count and percentage for every column."""
    total_rows = len(df)
    summary_data = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = (missing_count / total_rows * 100.0) if total_rows > 0 else 0.0
        non_missing = total_rows - missing_count

        summary_data.append({
            'Column': col,
            'Non-Null Count': non_missing,
            'Missing Count': missing_count,
            'Missing (%)': round(missing_pct, 2),
            'Data Type': str(df[col].dtype)
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df.sort_values(by='Missing (%)', ascending=False).reset_index(drop=True)


def compute_numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes distribution statistics (mean, std, min, quartiles, max) for numeric columns."""
    numeric_cols = [
        'Temp',
        'D.O. (mg/l)',
        'PH',
        'CONDUCTIVITY (\u00b5mhos/cm)',
        'B.O.D. (mg/l)',
        'NITRATENAN N+ NITRITENANN (mg/l)',
        'FECAL COLIFORM (MPN/100ml)',
        'TOTAL COLIFORM (MPN/100ml)Mean'
    ]
    existing_cols = [c for c in numeric_cols if c in df.columns]

    stats_list = []
    for c in existing_cols:
        series = df[c].dropna()
        if len(series) > 0:
            stats_list.append({
                'Parameter': c,
                'Valid Count': int(len(series)),
                'Mean': round(float(series.mean()), 2),
                'Std Dev': round(float(series.std()), 2),
                'Min': round(float(series.min()), 2),
                '25%': round(float(series.quantile(0.25)), 2),
                'Median (50%)': round(float(series.median()), 2),
                '75%': round(float(series.quantile(0.75)), 2),
                'Max': round(float(series.max()), 2)
            })

    return pd.DataFrame(stats_list)


def prepare_labelled_evaluation_data(
    df: pd.DataFrame, 
    model_expected_features: List[str]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Extracts the complete rows for model features and derives the ground-truth target
    ('Within pH range' for 6.5 <= pH <= 8.5 vs 'Outside pH range') as per standard
    drinking/surface water guidelines.
    """
    if 'PH' not in df.columns:
        return None, None

    # Map required columns
    cols_to_check = []
    cond_col = find_conductivity_column(df)
    
    col_mapping = {}
    for mf in model_expected_features:
        if 'CONDUCTIVITY' in mf.upper() and cond_col:
            col_mapping[mf] = cond_col
            cols_to_check.append(cond_col)
        elif mf in df.columns:
            col_mapping[mf] = mf
            cols_to_check.append(mf)

    if len(cols_to_check) != len(model_expected_features):
        logger.warning("Could not match all expected model features in dataset.")
        return None, None

    cols_to_check.append('PH')
    df_clean = df.dropna(subset=cols_to_check).copy()

    if len(df_clean) == 0:
        return None, None

    # Ground truth: permissible pH range [6.5, 8.5]
    y_true = np.where(
        (df_clean['PH'] >= 6.5) & (df_clean['PH'] <= 8.5),
        'Within pH range',
        'Outside pH range'
    )

    # Extract X with exact feature names expected by model
    X_clean = pd.DataFrame()
    for mf in model_expected_features:
        actual_col = col_mapping[mf]
        X_clean[mf] = df_clean[actual_col].values

    return X_clean, pd.Series(y_true, index=df_clean.index, name='GroundTruth_pH_Range')
