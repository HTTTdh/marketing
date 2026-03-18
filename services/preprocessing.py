"""
services/preprocessing.py
Handles data validation, cleaning, and scaling.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import streamlit as st


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    """Return list of missing required columns. Empty list = OK."""
    return [c for c in required if c not in df.columns]


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return names of all numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Impute missing values:
      - Numeric  → median
      - Categorical → mode
    Returns cleaned df and a report dict.
    """
    report: dict = {}
    df = df.copy()

    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            strategy = "median"
        else:
            fill_val = df[col].mode()[0]
            strategy = "mode"
        df[col] = df[col].fillna(fill_val)
        report[col] = {"missing": int(n_missing), "strategy": strategy, "fill_value": fill_val}

    return df, report


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def scale_features(df_json: str, feature_cols: list) -> Tuple[np.ndarray, object]:
    """
    StandardScaler on selected columns.
    Accepts JSON-serialised df so that st.cache_data can hash it.
    Returns (scaled_array, scaler).
    """
    import io as _io
    df = pd.read_json(_io.StringIO(df_json))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_upload(file) -> pd.DataFrame:
    """
    Parse an uploaded file (.csv or .xlsx) and return a DataFrame.
    Raises ValueError with a user-friendly message on failure.
    """
    name: str = file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file type: '{file.name}'. Please upload .csv or .xlsx.")
    except Exception as exc:
        raise ValueError(f"Could not read file '{file.name}': {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file contains no data.")

    return df
