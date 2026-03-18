"""
views/analyze.py
Main analysis page: upload → preprocess → cluster → visualize → AI insights → save → export.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

from services.preprocessing import (
    parse_upload,
    handle_missing,
    scale_features,
    get_numeric_columns,
)
from services.clustering import (
    compute_linkage,
    assign_clusters,
    compute_cluster_stats,
    cluster_profiles,
    compute_silhouette,
    compute_pca,
    detect_anomalies,
)
from services.visualization import (
    plot_dendrogram,
    plot_pca,
    plot_heatmap,
    plot_cluster_distribution,
    plot_cluster_comparison,
    plot_feature_boxplots,
)
from services.ai_service import analyze_all_clusters, analyze_overall
from services.database import save_analysis, init_db


# ---------------------------------------------------------------------------
# PDF export helper
# ---------------------------------------------------------------------------

def _generate_pdf_report(
    filename: str,
    n_customers: int,
    n_clusters: int,
    method: str,
    ai_insights: dict,
    overall_analysis: dict | None = None,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Customer Segmentation Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"File: {filename}   |   Customers: {n_customers}   |   Clusters: {n_clusters}   |   Method: {method}", ln=True, align="C")
    pdf.ln(6)

    # Overall analysis section in PDF
    if overall_analysis:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_fill_color(10, 60, 100)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, "  Overall Cross-Cluster Analysis", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        key_contrast = overall_analysis.get("key_contrast", "")
        if key_contrast:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Key Contrast:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            key_contrast = key_contrast.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, key_contrast)
            pdf.ln(2)

        overall_strategy = overall_analysis.get("overall_strategy", "")
        if overall_strategy:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Overall Business Strategy:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            overall_strategy = overall_strategy.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, overall_strategy)
            pdf.ln(2)

        priority_actions = overall_analysis.get("priority_actions", [])
        if priority_actions:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Priority Actions:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for action in priority_actions:
                action = action.encode("latin-1", errors="replace").decode("latin-1")
                pdf.cell(0, 6, f"  - {action}", ln=True)
        pdf.ln(6)

    for cluster_id, insight in ai_insights.items():
        pdf.set_font("Helvetica", "B", 13)
        name = insight.get("segment_name", f"Cluster {cluster_id}")
        name = name.encode("latin-1", errors="replace").decode("latin-1")
        pdf.set_fill_color(30, 30, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, f"  Cluster {cluster_id}: {name}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)

        for label, key in [
            ("Description", "description"),
            ("Behavior Insight", "behavior_insight"),
            ("Marketing Strategy", "marketing_strategy"),
        ]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"{label}:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            text = insight.get(key, "N/A")
            text = text.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, text)
            pdf.ln(1)

        campaigns = insight.get("suggested_campaigns", [])
        if campaigns:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Suggested Campaigns:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for c in campaigns:
                c = c.encode("latin-1", errors="replace").decode("latin-1")
                pdf.cell(0, 6, f"  - {c}", ln=True)
        pdf.ln(5)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def _df_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Cluster Results")
    return output.getvalue()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(api_key: str | None = None):
    init_db()
    st.title("🔬 Customer Segmentation Analysis")
    st.markdown("Upload your customer data and discover hidden segments using Agglomerative Hierarchical Clustering.")

    # -------------------------------------------------------------------------
    # STEP 1: File Upload
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📂 Step 1 — Upload Data")
    uploaded = st.file_uploader(
        "Upload customer data (.csv or .xlsx)",
        type=["csv", "xlsx", "xls"],
        help="File must contain at least 2 numeric columns for clustering.",
    )

    if not uploaded:
        st.info("👆 Upload a file to get started.")
        return

    try:
        df_raw = parse_upload(uploaded)
    except ValueError as e:
        st.error(str(e))
        return

    st.success(f"✅ Loaded **{len(df_raw):,}** rows × **{len(df_raw.columns)}** columns from **{uploaded.name}**")

    # -------------------------------------------------------------------------
    # STEP 2: Data Preview & Missing Value Handling
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("👀 Step 2 — Data Preview")
    with st.expander("Show raw data sample", expanded=False):
        st.dataframe(df_raw.head(50), width='stretch')

    df_clean, missing_report = handle_missing(df_raw.copy())

    if missing_report:
        st.warning(f"**Missing values detected and imputed in {len(missing_report)} column(s):**")
        miss_df = pd.DataFrame(missing_report).T.reset_index().rename(columns={"index": "Column"})
        st.dataframe(miss_df, width='stretch')
    else:
        st.success("✅ No missing values detected.")

    # -------------------------------------------------------------------------
    # STEP 3: Feature Selection & Clustering Configuration
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("⚙️ Step 3 — Configure Clustering")

    numeric_cols = get_numeric_columns(df_clean)
    if len(numeric_cols) < 2:
        st.error("❌ Dataset must have at least 2 numeric columns for clustering.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        feature_cols = st.multiselect(
            "Select features for clustering",
            options=numeric_cols,
            default=numeric_cols[:min(len(numeric_cols), 6)],
            help="Choose 2+ numeric columns.",
        )
    with col2:
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
    with col3:
        linkage_method = st.selectbox(
            "Linkage method",
            options=["ward", "complete", "average", "single"],
            index=0,
            help="'ward' minimises within-cluster variance (recommended).",
        )

    show_anomalies = st.checkbox("🔍 Detect anomalies (IsolationForest)", value=False)

    if len(feature_cols) < 2:
        st.warning("⚠️ Please select at least 2 features.")
        return

    # -------------------------------------------------------------------------
    # STEP 4: Run Analysis
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🚀 Step 4 — Run Analysis")

    run_clicked = st.button("▶ Run Clustering Analysis", width='stretch', type="primary")

    analysis_signature = {
        "uploaded_name": uploaded.name,
        "uploaded_size": uploaded.size,
        "feature_cols": tuple(feature_cols),
        "n_clusters": int(n_clusters),
        "linkage_method": linkage_method,
        "show_anomalies": bool(show_anomalies),
    }

    has_cached_result = "analysis_result" in st.session_state
    cached_result = st.session_state.get("analysis_result") if has_cached_result else None
    cached_matches_current = bool(cached_result and cached_result.get("signature") == analysis_signature)

    if run_clicked:
        progress = st.progress(0, text="Starting analysis…")
        status = st.empty()

        with st.spinner("Running analysis…"):
            status.text("⚙️ Scaling features…")
            progress.progress(10, "Scaling features…")
            scaled, _ = scale_features(df_clean.to_json(), feature_cols)

            status.text("🔗 Computing linkage matrix…")
            progress.progress(20, "Computing linkage…")
            linkage_matrix = compute_linkage(scaled, method=linkage_method)

            status.text("🎯 Assigning clusters…")
            progress.progress(35, "Assigning clusters…")
            labels = assign_clusters(scaled, n_clusters, linkage_method)

            status.text("📐 Computing PCA…")
            progress.progress(48, "PCA projection…")
            pca_coords = compute_pca(scaled)

            anomaly_mask = None
            if show_anomalies:
                status.text("🔍 Detecting anomalies…")
                progress.progress(55, "Detecting anomalies…")
                anomaly_mask = detect_anomalies(scaled)

            status.text("📊 Computing cluster statistics…")
            progress.progress(65, "Computing statistics…")
            df_result = df_clean.copy()
            df_result["Cluster"] = labels
            profiles = cluster_profiles(df_result, "Cluster")
            silhouette = compute_silhouette(scaled, labels)

            status.text("🤖 Generating per-cluster AI insights…")
            progress.progress(75, "Per-cluster AI analysis…")
            ai_insights = analyze_all_clusters(profiles, api_key=api_key)

            # NEW: overall cross-cluster analysis
            status.text("🧠 Generating overall cross-cluster analysis…")
            progress.progress(90, "Overall AI analysis…")
            overall_analysis = analyze_overall(
                profiles,
                ai_insights,
                api_key=api_key,
            )

            progress.progress(100, "Done!")
            status.empty()
            time.sleep(0.3)
            progress.empty()

        st.session_state["analysis_result"] = {
            "signature": analysis_signature,
            "df_result": df_result,
            "profiles": profiles,
            "labels": labels,
            "linkage_matrix": linkage_matrix,
            "pca_coords": pca_coords,
            "anomaly_mask": anomaly_mask,
            "silhouette": silhouette,
            "ai_insights": ai_insights,
            "overall_analysis": overall_analysis,
        }
        cached_result = st.session_state["analysis_result"]
        st.success("✅ Analysis complete!")

    elif not cached_matches_current:
        if has_cached_result:
            st.info("⚠️ Analysis settings changed. Click **Run Clustering Analysis** to refresh results before saving/exporting.")
        return

    df_result = cached_result["df_result"]
    profiles = cached_result["profiles"]
    labels = cached_result["labels"]
    linkage_matrix = cached_result.get("linkage_matrix")
    pca_coords = cached_result["pca_coords"]
    anomaly_mask = cached_result["anomaly_mask"]
    silhouette = cached_result["silhouette"]
    ai_insights = cached_result["ai_insights"]
    overall_analysis = cached_result.get("overall_analysis", {})

    if linkage_matrix is None:
        st.info("⚠️ Cached analysis format is outdated. Click **Run Clustering Analysis** once to refresh visualizations.")
        return

    # -------------------------------------------------------------------------
    # STEP 5: Metrics
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Step 5 — Summary Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Total Customers", f"{len(df_result):,}")
    m2.metric("🔵 Clusters", n_clusters)
    m3.metric("📐 Silhouette Score", f"{silhouette:.3f}" if silhouette >= 0 else "N/A")
    m4.metric("📊 Features Used", len(feature_cols))

    if show_anomalies and anomaly_mask is not None:
        n_anom = int(anomaly_mask.sum())
        st.info(f"⚠️ **{n_anom}** potential anomalies detected ({n_anom/len(df_result)*100:.1f}% of customers).")

    # -------------------------------------------------------------------------
    # STEP 6: Visualizations
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Step 6 — Visualizations")

    tab_dend, tab_pca, tab_heat, tab_dist, tab_box, tab_radar = st.tabs([
        "🌲 Dendrogram", "🔵 PCA", "🌡️ Heatmap",
        "📊 Distribution", "📦 Box Plots", "🕸️ Radar"
    ])

    with tab_dend:
        st.subheader("Hierarchical Clustering Dendrogram")
        fig_dend = plot_dendrogram(linkage_matrix)
        st.pyplot(fig_dend, width='stretch')

    with tab_pca:
        st.subheader("PCA — 2D Cluster Projection")
        fig_pca = plot_pca(pca_coords, labels, anomaly_mask)
        st.plotly_chart(fig_pca, width='stretch')

    with tab_heat:
        st.subheader("Feature Profile Heatmap")
        fig_heat = plot_heatmap(profiles)
        st.pyplot(fig_heat, width='stretch')

    with tab_dist:
        st.subheader("Customer Distribution per Cluster")
        fig_dist = plot_cluster_distribution(labels)
        st.plotly_chart(fig_dist, width='stretch')

    with tab_box:
        st.subheader("Feature Distribution by Cluster")
        fig_box = plot_feature_boxplots(df_result, feature_cols)
        st.plotly_chart(fig_box, width='stretch')

    with tab_radar:
        st.subheader("Cluster Comparison (Radar)")
        fig_radar = plot_cluster_comparison(profiles)
        st.plotly_chart(fig_radar, width='stretch')

    # -------------------------------------------------------------------------
    # STEP 7: AI Insights
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Step 7 — AI-Powered Cluster Insights")

    if not os.getenv("GEMINI_API_KEY"):
        st.info("💡 **Tip:** Set `GEMINI_API_KEY` in your `.env` file for real Gemini AI insights. Currently showing rule-based analysis.")

    # --- 7A: Overall Cross-Cluster Analysis (NEW) ---
    if overall_analysis:
        st.markdown("### 🧠 Overall Analysis")

        # Key contrast callout
        key_contrast = overall_analysis.get("key_contrast", "")
        if key_contrast:
            st.info(f"**💡 Key Contrast across segments:** {key_contrast}")

        # Cluster comparison table
        comparisons = overall_analysis.get("cluster_comparison", [])
        if comparisons:
            st.markdown("#### 🔍 Cross-Cluster Comparison")
            comp_df = pd.DataFrame(comparisons)
            comp_df.columns = ["Aspect", "Summary"]
            st.table(comp_df)

        # Overall strategy
        overall_strategy = overall_analysis.get("overall_strategy", "")
        if overall_strategy:
            st.markdown("#### 🏢 Overall Business Strategy")
            st.success(overall_strategy)

        # Priority actions
        priority_actions = overall_analysis.get("priority_actions", [])
        if priority_actions:
            st.markdown("#### 🎯 Priority Actions")
            for i, action in enumerate(priority_actions, start=1):
                st.markdown(f"**{i}.** {action}")

        st.markdown("---")

    # --- 7B: Per-Cluster Insights (existing) ---
    st.markdown("### 📋 Per-Cluster Breakdown")
    for cluster_id, insight in ai_insights.items():
        segment_name = insight.get("segment_name", f"Cluster {cluster_id}")
        with st.expander(f"🔵 Cluster {cluster_id}: **{segment_name}**", expanded=True):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown("**📝 Description**")
                st.write(insight.get("description", ""))
                st.markdown("**🧠 Behavior Insight**")
                st.write(insight.get("behavior_insight", ""))
            with col_b:
                st.markdown("**🎯 Marketing Strategy**")
                st.write(insight.get("marketing_strategy", ""))
                campaigns = insight.get("suggested_campaigns", [])
                if campaigns:
                    st.markdown("**📣 Suggested Campaigns**")
                    for c in campaigns:
                        st.markdown(f"  - {c}")

    # -------------------------------------------------------------------------
    # STEP 8: Save to Database
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💾 Step 8 — Save Analysis")

    if st.button("💾 Save to History", width='stretch'):
        with st.spinner("Saving…"):
            cluster_stats_json = {
                str(cid): profiles.loc[cid].to_dict() for cid in profiles.index
            }
            analysis_id = save_analysis(
                filename=uploaded.name,
                number_of_customers=len(df_result),
                number_of_clusters=n_clusters,
                clustering_method=linkage_method,
                cluster_stats=cluster_stats_json,
                ai_insights={str(k): v for k, v in ai_insights.items()},
            )
        st.success(f"✅ Analysis saved! (ID: `{analysis_id}`)")

    # -------------------------------------------------------------------------
    # STEP 9: Export
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Step 9 — Export Results")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download CSV",
            data=csv_bytes,
            file_name=f"clusters_{uploaded.name.rsplit('.', 1)[0]}.csv",
            mime="text/csv",
            width='stretch',
        )

    with exp2:
        excel_bytes = _df_to_excel(df_result)
        st.download_button(
            label="📊 Download Excel",
            data=excel_bytes,
            file_name=f"clusters_{uploaded.name.rsplit('.', 1)[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
        )

    with exp3:
        pdf_bytes = _generate_pdf_report(
            filename=uploaded.name,
            n_customers=len(df_result),
            n_clusters=n_clusters,
            method=linkage_method,
            ai_insights={str(k): v for k, v in ai_insights.items()},
            overall_analysis=overall_analysis,
        )
        st.download_button(
            label="📋 Download PDF Report",
            data=pdf_bytes,
            file_name=f"report_{uploaded.name.rsplit('.', 1)[0]}.pdf",
            mime="application/pdf",
            width='stretch',
        )