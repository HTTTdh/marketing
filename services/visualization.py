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
# Dendrogram (matplotlib)
# ---------------------------------------------------------------------------

def plot_dendrogram(
    linkage_matrix: np.ndarray,
    title: str = "Hierarchical Clustering Dendrogram",
    truncate_mode: Optional[str] = "lastp",
    p: int = 30,
    color_threshold: Optional[float] = None,
) -> plt.Figure:
    """Return matplotlib Figure with dendrogram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ddata = dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=color_threshold,
        above_threshold_color="#888",
    )
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_xlabel("Sample Index / Cluster Size", color="#aaa", fontsize=10)
    ax.set_ylabel("Distance", color="#aaa", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PCA Scatter (Plotly)
# ---------------------------------------------------------------------------

def plot_pca(
    pca_coords: np.ndarray,
    labels: np.ndarray,
    anomaly_mask: Optional[np.ndarray] = None,
) -> go.Figure:
    """2D PCA scatter coloured by cluster, with optional anomaly overlay."""
    df = pd.DataFrame(
        {"PC1": pca_coords[:, 0], "PC2": pca_coords[:, 1], "Cluster": labels.astype(str)}
    )
    if anomaly_mask is not None:
        df["Anomaly"] = anomaly_mask

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="PCA – Customer Clusters",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={"PC1": ":.3f", "PC2": ":.3f", "Cluster": True},
    )

    if anomaly_mask is not None:
        anomaly_df = df[df["Anomaly"]]
        fig.add_trace(
            go.Scatter(
                x=anomaly_df["PC1"],
                y=anomaly_df["PC2"],
                mode="markers",
                marker=dict(symbol="x", size=10, color="red", line=dict(width=2)),
                name="Anomaly",
            )
        )

    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        legend_title_text="Cluster",
    )
    return fig


# ---------------------------------------------------------------------------
# Seaborn Heatmap (matplotlib)
# ---------------------------------------------------------------------------

def plot_heatmap(profiles: pd.DataFrame) -> plt.Figure:
    """Return seaborn heatmap of cluster mean feature values."""
    fig, ax = plt.subplots(figsize=(max(10, len(profiles.columns)), max(4, len(profiles) + 1)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    sns.heatmap(
        profiles,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.5,
        linecolor="#222",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title("Cluster Feature Profiles (Mean)", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Features", color="#aaa")
    ax.set_ylabel("Cluster", color="#aaa")
    ax.tick_params(colors="white", labelsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cluster Size Distribution (Plotly bar)
# ---------------------------------------------------------------------------

def plot_cluster_distribution(labels: np.ndarray) -> go.Figure:
    """Bar chart of customer count per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    fig = px.bar(
        x=[f"Cluster {u}" for u in unique],
        y=counts,
        labels={"x": "Cluster", "y": "Number of Customers"},
        title="Customer Count per Cluster",
        template="plotly_dark",
        color=counts,
        color_continuous_scale="Viridis",
        text=counts,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
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
                name=f"Cluster {cluster_id}",
                line_color=colors[i % len(colors)],
                opacity=0.7,
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="#1a1a2e",
            radialaxis=dict(visible=True, range=[0, 1], color="white"),
            angularaxis=dict(color="white"),
        ),
        showlegend=True,
        template="plotly_dark",
        title="Cluster Comparison (Normalised Features)",
        paper_bgcolor="#0e1117",
        font_color="white",
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
        title="Feature Distribution by Cluster",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        legend_title_text="Cluster",
    )
    return fig
