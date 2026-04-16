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
    :root {
        --bg-main: #f7f8fc;
        --bg-card: #ffffff;
        --bg-soft: #eef2ff;
        --border: #d9e0ef;
        --text-main: #1f2937;
        --text-muted: #5b6475;
        --accent: #315efb;
        --accent-2: #5aa9ff;
    }
    html, body, [class*="css"] {
        color: var(--text-main);
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: transparent !important;
    }
    /* Light sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f5f7fb 100%);
        border-right: 1px solid var(--border);
    }
    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at top left, #eef4ff 0%, transparent 28%),
            linear-gradient(180deg, #f9fbff 0%, #f4f6fb 100%);
        color: var(--text-main);
    }
    .block-container {
        color: var(--text-main);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
        color: var(--text-main) !important;
        border-color: var(--border) !important;
    }
    .stDataFrame, .stTable, [data-testid="stDataFrame"] {
        background: white !important;
        color: var(--text-main) !important;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"],
    [data-testid="stCaptionContainer"],
    label,
    .stSelectbox label,
    .stMultiSelect label,
    .stNumberInput label,
    .stCheckbox label {
        color: var(--text-main) !important;
    }
    [data-baseweb="tag"] {
        background-color: var(--bg-soft) !important;
        color: var(--text-main) !important;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: var(--bg-card);
        border-radius: 14px;
        padding: 14px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.88);
        border: 1px solid var(--border);
        border-radius: 10px 10px 0 0;
        color: var(--text-main);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-soft);
        border-bottom-color: transparent;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--border);
    }
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
        border: none;
        font-weight: 600;
        letter-spacing: 0.2px;
        color: white;
    }
    /* Expander headers */
    details summary {
        background-color: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 8px 12px;
    }
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }
    /* Inputs / containers */
    [data-testid="stFileUploader"],
    [data-testid="stSelectbox"],
    [data-testid="stMultiSelect"],
    [data-testid="stNumberInput"],
    [data-testid="stTextInput"],
    [data-testid="stDateInput"] {
        background: transparent;
    }
    [data-testid="stMarkdownContainer"] p {
        color: var(--text-main);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
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
