"""
services/preprocessing.py
Handles data validation, cleaning, and scaling.
"""

from __future__ import annotations

import io
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
def scale_features(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[np.ndarray, object]:
    """
    StandardScaler on selected columns.
    Returns (scaled_array, scaler).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _parse_upload_cached(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Cached parser for uploaded file bytes."""
    lower_name = file_name.lower()

    if lower_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if lower_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    raise ValueError(f"Unsupported file type: '{file_name}'. Please upload .csv or .xlsx.")


def parse_upload(file) -> pd.DataFrame:
    """
    Parse an uploaded file (.csv or .xlsx) and return a DataFrame.
    Raises ValueError with a user-friendly message on failure.
    """
    try:
        file_bytes = file.getvalue()
        df = _parse_upload_cached(file.name, file_bytes)
    except Exception as exc:
        raise ValueError(f"Could not read file '{file.name}': {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file contains no data.")

    return df
