"""
services/visualization.py
All chart/plot functions. Returns figures for Streamlit rendering.
Preserves: dendrogram, PCA scatter, seaborn heatmap, plotly bar/box.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
from typing import Optional
# ---------------------------------------------------------------------------
# Elbow Method chart (matplotlib)
# ---------------------------------------------------------------------------

def plot_elbow(k_values: list, inertia_values: list, recommended_k: int | None = None) -> plt.Figure:
    """
    Return a matplotlib elbow chart matching the reference style.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#e5e5e5")
    ax.set_facecolor("white")
    ax.plot(k_values, inertia_values, marker="o", linewidth=1.6, color="#1f77b4", markersize=6)
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("WCSS")

    if recommended_k is not None and recommended_k in k_values:
        idx = k_values.index(recommended_k)
        ax.scatter(
            [recommended_k],
            [inertia_values[idx]],
            s=140,
            color="orange",
            edgecolors="black",
            zorder=3,
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dendrogram (matplotlib)
# ---------------------------------------------------------------------------

def plot_dendrogram(
    linkage_matrix: np.ndarray,
    title: str = "Bieu do cay Phan cum Phan cap",
    truncate_mode: Optional[str] = "lastp",
    p: int = 30,
    color_threshold: Optional[float] = None,
) -> plt.Figure:
    """Return matplotlib Figure with dendrogram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ddata = dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=color_threshold,
        above_threshold_color="#888",
    )
    ax.set_title(title, color="#1f2937", fontsize=14, pad=12)
    ax.set_xlabel("Chi so Mau / Kich thuoc Cum", color="#5b6475", fontsize=10)
    ax.set_ylabel("Khoang cach", color="#5b6475", fontsize=10)
    ax.tick_params(colors="#374151")
    ax.grid(True, axis="y", alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#d9e0ef")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PCA Scatter (matplotlib)
# ---------------------------------------------------------------------------

def plot_pca(
    pca_coords: np.ndarray,
    labels: np.ndarray,
    centroid_coords: Optional[np.ndarray] = None,
    anomaly_mask: Optional[np.ndarray] = None,
) -> plt.Figure:
    """2D PCA scatter for Fuzzy C-Means clustering."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#e5e5e5")
    ax.set_facecolor("white")

    unique_labels = np.unique(labels)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]

    for i, cluster_id in enumerate(unique_labels):
        cluster_mask = labels == cluster_id
        ax.scatter(
            pca_coords[cluster_mask, 0],
            pca_coords[cluster_mask, 1],
            s=40,
            color=palette[i % len(palette)],
            label=f"Cluster {int(cluster_id) + 1}",
            alpha=0.85,
        )

    if centroid_coords is not None and len(centroid_coords) > 0:
        ax.scatter(
            centroid_coords[:, 0],
            centroid_coords[:, 1],
            c="#b22222",
            s=180,
            marker="X",
            linewidths=2.5,
            label="Centroids",
            zorder=5,
        )

    if anomaly_mask is not None and anomaly_mask.any():
        ax.scatter(
            pca_coords[anomaly_mask, 0],
            pca_coords[anomaly_mask, 1],
            s=70,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Anomaly",
            zorder=6,
        )

    ax.set_title("Fuzzy C-Means Clustering (PCA Visualization)")
    ax.set_ylabel("PCA 2")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Seaborn Heatmap (matplotlib)
# ---------------------------------------------------------------------------

def plot_heatmap(profiles: pd.DataFrame) -> plt.Figure:
    """Return seaborn heatmap of cluster mean feature values."""
    fig, ax = plt.subplots(figsize=(max(10, len(profiles.columns)), max(4, len(profiles) + 1)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.heatmap(
        profiles,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.5,
        linecolor="#e5e7eb",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title("Ho so Dac trung Cum (Trung binh)", color="#1f2937", fontsize=13, pad=12)
    ax.set_xlabel("Dac trung", color="#5b6475")
    ax.set_ylabel("Cum", color="#5b6475")
    ax.tick_params(colors="#374151", labelsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster Size Distribution (Plotly bar)
# ---------------------------------------------------------------------------

def plot_cluster_distribution(labels: np.ndarray) -> go.Figure:
    """Bar chart of customer count per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    fig = px.bar(
        x=[f"Cum {u}" for u in unique],
        y=counts,
        labels={"x": "Cum", "y": "So luong Khach hang"},
        title="So luong Khach hang moi Cum",
        template="plotly_white",
        color=counts,
        color_continuous_scale="Blues",
        text=counts,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1f2937",
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Cluster Comparison Radar Chart (Plotly) — bonus
# ---------------------------------------------------------------------------

def plot_cluster_comparison(profiles: pd.DataFrame) -> go.Figure:
    """Radar chart comparing all clusters across features."""
    features = profiles.columns.tolist()
    fig = go.Figure()

    # Normalise for radar (0-1 per feature)
    norm = (profiles - profiles.min()) / (profiles.max() - profiles.min() + 1e-9)

    colors = px.colors.qualitative.Bold
    for i, cluster_id in enumerate(norm.index):
        values = norm.loc[cluster_id].tolist()
        values += values[:1]  # close the polygon
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=features + [features[0]],
                fill="toself",
                name=f"Cum {cluster_id}",
                line_color=colors[i % len(colors)],
                opacity=0.7,
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0, 1], color="#374151", gridcolor="#d9e0ef"),
            angularaxis=dict(color="#374151", gridcolor="#d9e0ef"),
        ),
        showlegend=True,
        template="plotly_white",
        title="So sanh Cum (Dac trung Chuan hoa)",
        paper_bgcolor="white",
        font_color="#1f2937",
    )
    return fig


# ---------------------------------------------------------------------------
# Box plots per cluster (Plotly)
# ---------------------------------------------------------------------------

def plot_feature_boxplots(df: pd.DataFrame, feature_cols: list, cluster_col: str = "Cluster") -> go.Figure:
    """Faceted box plots for selected numeric features split by cluster."""
    df_plot = df[[cluster_col] + feature_cols].copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(str)
    melted = df_plot.melt(id_vars=cluster_col, var_name="Feature", value_name="Value")

    fig = px.box(
        melted,
        x="Feature",
        y="Value",
        color=cluster_col,
        title="Phan phoi Dac trung theo Cum",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#1f2937",
        legend_title_text="Cum",
    )
    return fig
