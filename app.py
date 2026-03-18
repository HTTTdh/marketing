"""
app.py
Entry point for the Customer Segmentation Streamlit App.
Handles page config, sidebar navigation, and .env loading.
"""

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env, overriding any existing system variables
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Customer Segmentation AI — powered by AHC + OpenAI",
    },
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
    }
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #2d3748;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2e;
        border-radius: 6px 6px 0 0;
    }
    /* Buttons */
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    /* Expander headers */
    details summary {
        background-color: #1a1f2e;
        border-radius: 6px;
        padding: 8px 12px;
    }
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Segmentation AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        options=["🔍 Analyze", "📚 History"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        **Tech Stack**
        - Streamlit · Pandas · NumPy
        - Scipy · Scikit-learn
        - OpenAI GPT-4o-mini
        - SQLite · SQLAlchemy
        - Plotly · Seaborn
        """
    )
    st.markdown("---")
    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    if api_key_set:
        st.success("🔑 Gemini API Key: Loaded")
    else:
        st.warning("🔑 Gemini API Key: Not Found\n\nAdd key to `.env` file to enable AI features.")

# ─── Route to Pages ───────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")

if page == "🔍 Analyze":
    from views.analyze import render
    render(api_key=api_key)
else:
    from views.history import render
    render()
