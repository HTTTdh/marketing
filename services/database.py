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


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=ENGINE)


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
    }
