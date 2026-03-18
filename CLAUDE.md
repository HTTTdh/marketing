# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Vietnamese-language Streamlit web app for customer segmentation using Agglomerative Hierarchical Clustering (AHC) with AI-powered insights via Google Gemini. Users upload CSV/Excel data, configure clustering parameters, and get visualizations + AI marketing analysis per segment.

## Commands

```bash
# Run locally (requires .venv with dependencies)
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Run with Docker
docker-compose up --build
```

No test suite exists. No linter is configured.

## Architecture

**Entry point:** `app.py` — sets page config, custom CSS, sidebar navigation, loads `.env`, and routes to view functions.

**Routing pattern:** `app.py` imports and calls `render()` from `views/analyze.py` or `views/history.py` based on sidebar selection. The `pages/` directory contains thin wrappers that just call the same `render()` functions (Streamlit multipage compatibility).

### Layer Structure

- **`views/`** — Page-level render functions. Each exports a `render()` function that orchestrates the full page UI and workflow.
  - `analyze.py` — Main 9-step analysis pipeline: upload → preprocess → configure → cluster → metrics → visualize → AI insights → save → export (CSV/Excel/PDF)
  - `history.py` — Browse/view/delete saved analyses from SQLite

- **`services/`** — Business logic, stateless functions (no Streamlit UI code except `@st.cache_data` decorators):
  - `preprocessing.py` — File parsing (CSV/Excel), missing value imputation (median/mode), StandardScaler feature scaling
  - `clustering.py` — Scipy/sklearn AHC: linkage matrix, cluster assignment, silhouette score, PCA reduction, IsolationForest anomaly detection
  - `visualization.py` — Returns matplotlib/plotly figures: dendrogram, PCA scatter, heatmap, bar chart, box plots, radar chart. All use dark theme (`#0e1117` background)
  - `ai_service.py` — Google Gemini API calls for per-cluster and cross-cluster analysis. Returns structured JSON. Falls back to rule-based insights when API key missing or call fails. Includes JSON repair for truncated Gemini responses.
  - `database.py` — SQLAlchemy + SQLite (`analyses.db` at project root). Single `analyses` table stores results as JSON strings.

### Key Patterns

- AI service always has a rule-based fallback — the app works without a Gemini API key.
- `scale_features()` accepts JSON-serialized DataFrames for Streamlit cache compatibility.
- Analysis results are cached in `st.session_state["analysis_result"]` with a signature dict to detect config changes.
- UI text is in Vietnamese.

## Environment

Requires `GEMINI_API_KEY` in `.env` for AI features (optional — rule-based fallback exists). See `.env.example`. Python 3.11. Deployed on Render (`render.yaml`).
