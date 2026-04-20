"""
services/clustering.py
AHC clustering, PCA, silhouette score, anomaly detection, elbow method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Tuple, List
import streamlit as st


# ---------------------------------------------------------------------------
# Elbow Method (KMeans inertia) — help choose optimal k
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_elbow(scaled_data: np.ndarray, k_range: List[int] | None = None) -> Tuple[List[int], List[float]]:
    """
    Compute KMeans inertia for a range of k values.
    Returns (k_values, inertia_values) for plotting the Elbow curve.
    """
    if k_range is None:
        k_range = list(range(1, 15))
    sample_data = scaled_data
    if len(scaled_data) > 5000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(scaled_data), size=5000, replace=False)
        sample_data = scaled_data[sample_idx]
    inertias: List[float] = []
    for k in k_range:
        if k <= 1:
            centroid = sample_data.mean(axis=0)
            inertia = float(((sample_data - centroid) ** 2).sum())
        else:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                random_state=42,
                algorithm="lloyd",
            )
            km.fit(sample_data)
            inertia = float(km.inertia_)
        inertias.append(inertia)
    for i in range(1, len(inertias)):
        if inertias[i] > inertias[i - 1]:
            inertias[i] = inertias[i - 1]
    return k_range, inertias


# ---------------------------------------------------------------------------
# Linkage (for dendrogram)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_linkage(scaled_data: np.ndarray, method: str = "ward") -> np.ndarray:
    """Compute scipy linkage matrix."""
    return linkage(scaled_data, method=method)


# ---------------------------------------------------------------------------
# Cluster assignment
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def assign_clusters(
    scaled_data: np.ndarray,
    n_clusters: int,
    linkage_method: str = "ward",
) -> np.ndarray:
    """Fit AgglomerativeClustering and return integer label array."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    return model.fit_predict(scaled_data)


# ---------------------------------------------------------------------------
# Cluster statistics
# ---------------------------------------------------------------------------

def compute_cluster_stats(df: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """
    Return a DataFrame with mean, std, count per cluster.
    Numeric columns only.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)

    stats = df.groupby(cluster_col)[numeric_cols].agg(["mean", "std", "count"])
    stats.columns = ["_".join(c) for c in stats.columns]
    stats = stats.reset_index()
    return stats


def cluster_profiles(df: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Per-cluster mean of numeric columns — used for heatmap / AI prompt."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)
    return df.groupby(cluster_col)[numeric_cols].mean()


# ---------------------------------------------------------------------------
# Silhouette score
# ---------------------------------------------------------------------------

def compute_silhouette(scaled_data: np.ndarray, labels: np.ndarray) -> float:
    """Return silhouette score or -1 if not computable."""
    try:
        if len(set(labels)) < 2:
            return -1.0
        return float(silhouette_score(scaled_data, labels))
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# PCA reduction
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_pca(scaled_data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce dimensionality to n_components with PCA."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(scaled_data)


def _fuzzy_c_means(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuzzy C-Means clustering. Returns (centers, membership_matrix).
    membership[i, c] = degree to which sample i belongs to cluster c.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    U = rng.random((n_samples, n_clusters))
    U = U / U.sum(axis=1, keepdims=True)

    centers = np.zeros((n_clusters, X.shape[1]))
    for _ in range(max_iter):
        Um = U ** m
        denom = Um.sum(axis=0)[:, None]
        denom = np.where(denom == 0, 1e-10, denom)
        centers = (Um.T @ X) / denom

        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        dist = np.fmax(dist, 1e-10)
        inv_dist = dist ** (-2.0 / (m - 1))
        U_new = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new

    return centers, U


@st.cache_data(show_spinner=False)
def compute_fcm_pca_projection(
    scaled_data: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuzzy C-Means + PCA plotting flow:
      - PCA on scaled data to 2D
      - FCM on scaled data -> hard labels (argmax of membership)
      - Project FCM centers into PCA space
    Returns (pca_coords, cluster_labels, centroid_coords).
    """
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(scaled_data)

    centers, U = _fuzzy_c_means(scaled_data, n_clusters=n_clusters)
    cluster_labels = U.argmax(axis=1)
    centroid_coords = pca.transform(centers)
    return pca_coords, cluster_labels, centroid_coords


# ---------------------------------------------------------------------------
# Anomaly detection (bonus)
# ---------------------------------------------------------------------------

def detect_anomalies(scaled_data: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Use IsolationForest to flag anomalies.
    Returns boolean array: True = anomaly.
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(scaled_data)
    return preds == -1  # -1 means anomaly in sklearn
