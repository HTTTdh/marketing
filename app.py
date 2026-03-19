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
        "About": "AI Phân khúc Khách hàng — sử dụng AHC + Gemini",
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
    st.markdown("## 🔬 AI Phân khúc")
    st.markdown("---")
    page = st.radio(
        "Điều hướng",
        options=["🔍 Phân tích", "📚 Lịch sử"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        **Ngăn xếp công nghệ**
        - Streamlit · Pandas · NumPy
        - Scipy · Scikit-learn
        - OpenAI GPT-4o-mini
        - SQLite · SQLAlchemy
        - Plotly · Seaborn
        """
    )
    st.markdown("---")
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    if api_key_set:
        st.success("🔑 Khóa API OpenAI: Đã tải")
    else:
        st.warning("🔑 Khóa API OpenAI: Không tìm thấy\n\nThêm khóa vào tệp `.env` để bật các tính năng AI.")

# ─── Route to Pages ───────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")

if page == "🔍 Phân tích":
    from views.analyze import render
    render(api_key=api_key)
else:
    from views.history import render
    render()
