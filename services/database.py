"""
services/database.py
SQLite persistence layer using SQLAlchemy.
Table: analyses
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "analyses.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=ENGINE, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    number_of_customers = Column(Integer, nullable=False)
    number_of_clusters = Column(Integer, nullable=False)
    clustering_method = Column(String(50), default="ward")
    cluster_stats = Column(Text, nullable=True)   # JSON string
    ai_insights = Column(Text, nullable=True)     # JSON string
    # Extended data for full visualization replay
    labels_json = Column(Text, nullable=True)
    pca_coords_json = Column(Text, nullable=True)
    linkage_matrix_json = Column(Text, nullable=True)
    feature_cols_json = Column(Text, nullable=True)
    df_result_json = Column(Text, nullable=True)
    overall_analysis = Column(Text, nullable=True)
    silhouette_score = Column(String(20), nullable=True)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist, and migrate schema for new columns."""
    Base.metadata.create_all(bind=ENGINE)
    _migrate_add_columns()


def _migrate_add_columns() -> None:
    """Add new columns to existing tables if they are missing."""
    new_cols = [
        ("labels_json", "TEXT"),
        ("pca_coords_json", "TEXT"),
        ("linkage_matrix_json", "TEXT"),
        ("feature_cols_json", "TEXT"),
        ("df_result_json", "TEXT"),
        ("overall_analysis", "TEXT"),
        ("silhouette_score", "VARCHAR(20)"),
    ]
    with ENGINE.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(analyses)"))
        existing = {row[1] for row in result.fetchall()}
        for col_name, col_type in new_cols:
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE analyses ADD COLUMN {col_name} {col_type}"))
        conn.commit()


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def save_analysis(
    filename: str,
    number_of_customers: int,
    number_of_clusters: int,
    clustering_method: str,
    cluster_stats: dict,
    ai_insights: dict,
    *,
    labels: list | None = None,
    pca_coords: list | None = None,
    linkage_matrix: list | None = None,
    feature_cols: list | None = None,
    df_result: dict | None = None,
    overall_analysis: dict | None = None,
    silhouette_score: float | None = None,
) -> str:
    """Persist a new analysis record. Returns the generated UUID."""
    record = Analysis(
        id=str(uuid.uuid4()),
        filename=filename,
        number_of_customers=number_of_customers,
        number_of_clusters=number_of_clusters,
        clustering_method=clustering_method,
        cluster_stats=json.dumps(cluster_stats, default=str),
        ai_insights=json.dumps(ai_insights, default=str),
        labels_json=json.dumps(labels, default=str) if labels is not None else None,
        pca_coords_json=json.dumps(pca_coords, default=str) if pca_coords is not None else None,
        linkage_matrix_json=json.dumps(linkage_matrix, default=str) if linkage_matrix is not None else None,
        feature_cols_json=json.dumps(feature_cols) if feature_cols is not None else None,
        df_result_json=json.dumps(df_result, default=str) if df_result is not None else None,
        overall_analysis=json.dumps(overall_analysis, default=str) if overall_analysis is not None else None,
        silhouette_score=str(silhouette_score) if silhouette_score is not None else None,
    )
    with SessionLocal() as session:
        session.add(record)
        session.commit()
        return record.id


def list_analyses() -> List[dict]:
    """Return all analyses as list of dicts, ordered by newest first."""
    with SessionLocal() as session:
        rows = session.query(Analysis).order_by(Analysis.created_at.desc()).all()
        return [_to_dict(r) for r in rows]


def get_analysis(analysis_id: str) -> Optional[dict]:
    """Return a single analysis by ID, or None."""
    with SessionLocal() as session:
        row = session.query(Analysis).filter(Analysis.id == analysis_id).first()
        return _to_dict(row) if row else None


def delete_analysis(analysis_id: str) -> bool:
    """Delete an analysis by ID. Returns True if deleted."""
    with SessionLocal() as session:
        row = session.query(Analysis).filter(Analysis.id == analysis_id).first()
        if row:
            session.delete(row)
            session.commit()
            return True
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(row: Analysis) -> dict:
    return {
        "id": row.id,
        "filename": row.filename,
        "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else "",
        "number_of_customers": row.number_of_customers,
        "number_of_clusters": row.number_of_clusters,
        "clustering_method": row.clustering_method,
        "cluster_stats": json.loads(row.cluster_stats) if row.cluster_stats else {},
        "ai_insights": json.loads(row.ai_insights) if row.ai_insights else {},
        "labels": json.loads(row.labels_json) if row.labels_json else None,
        "pca_coords": json.loads(row.pca_coords_json) if row.pca_coords_json else None,
        "linkage_matrix": json.loads(row.linkage_matrix_json) if row.linkage_matrix_json else None,
        "feature_cols": json.loads(row.feature_cols_json) if row.feature_cols_json else None,
        "df_result": json.loads(row.df_result_json) if row.df_result_json else None,
        "overall_analysis": json.loads(row.overall_analysis) if row.overall_analysis else None,
        "silhouette_score": float(row.silhouette_score) if row.silhouette_score else None,
    }
