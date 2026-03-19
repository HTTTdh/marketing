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

def _get_font_path():
    """Locate DejaVuSans font from matplotlib for Unicode/Vietnamese support."""
    import matplotlib
    font_dir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    return os.path.join(font_dir, "DejaVuSans.ttf"), os.path.join(font_dir, "DejaVuSans-Bold.ttf")


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

    # Register Unicode font for Vietnamese
    font_regular, font_bold = _get_font_path()
    pdf.add_font("VN", "", font_regular, uni=True)
    pdf.add_font("VN", "B", font_bold, uni=True)

    pdf.add_page()
    pdf.set_font("VN", "B", 16)
    pdf.cell(0, 10, "Báo cáo Phân khúc Khách hàng", ln=True, align="C")
    pdf.set_font("VN", "", 10)
    pdf.cell(0, 7, f"Tệp: {filename}   |   Khách hàng: {n_customers}   |   Cụm: {n_clusters}   |   Phương pháp: {method}", ln=True, align="C")
    pdf.ln(6)

    # Overall analysis section in PDF
    if overall_analysis:
        pdf.set_font("VN", "B", 14)
        pdf.set_fill_color(10, 60, 100)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, "  Phân tích Tổng thể Liên cụm", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        key_contrast = overall_analysis.get("key_contrast", "")
        if key_contrast:
            pdf.set_font("VN", "B", 10)
            pdf.cell(0, 6, "Tương phản chính:", ln=True)
            pdf.set_font("VN", "", 10)
            pdf.multi_cell(0, 6, key_contrast)
            pdf.ln(2)

        overall_strategy = overall_analysis.get("overall_strategy", "")
        if overall_strategy:
            pdf.set_font("VN", "B", 10)
            pdf.cell(0, 6, "Chiến lược kinh doanh tổng thể:", ln=True)
            pdf.set_font("VN", "", 10)
            pdf.multi_cell(0, 6, overall_strategy)
            pdf.ln(2)

        priority_actions = overall_analysis.get("priority_actions", [])
        if priority_actions:
            pdf.set_font("VN", "B", 10)
            pdf.cell(0, 6, "Hành động ưu tiên:", ln=True)
            pdf.set_font("VN", "", 10)
            for action in priority_actions:
                pdf.cell(0, 6, f"  - {action}", ln=True)
        pdf.ln(6)

    for cluster_id, insight in ai_insights.items():
        pdf.set_font("VN", "B", 13)
        name = insight.get("segment_name", f"Cụm {cluster_id}")
        pdf.set_fill_color(30, 30, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 9, f"  Cụm {cluster_id}: {name}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)

        for label, key in [
            ("Mô tả", "description"),
            ("Thông tin hành vi", "behavior_insight"),
            ("Chiến lược tiếp thị", "marketing_strategy"),
        ]:
            pdf.set_font("VN", "B", 10)
            pdf.cell(0, 6, f"{label}:", ln=True)
            pdf.set_font("VN", "", 10)
            text = insight.get(key, "N/A")
            pdf.multi_cell(0, 6, text)
            pdf.ln(1)

        campaigns = insight.get("suggested_campaigns", [])
        if campaigns:
            pdf.set_font("VN", "B", 10)
            pdf.cell(0, 6, "Các chiến dịch đề xuất:", ln=True)
            pdf.set_font("VN", "", 10)
            for c in campaigns:
                pdf.cell(0, 6, f"  - {c}", ln=True)
        pdf.ln(5)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def _df_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Ket qua Phan cum")
    return output.getvalue()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(api_key: str | None = None):
    init_db()
    st.title("🔬 Phân tích Phân khúc Khách hàng")
    st.markdown("Tải lên dữ liệu khách hàng của bạn và khám phá các phân khúc tiềm ẩn bằng Phân cụm phân cấp tập hợp (Agglomerative Hierarchical Clustering).")

    # -------------------------------------------------------------------------
    # STEP 1: File Upload
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📂 Bước 1 — Tải lên Dữ liệu")
    uploaded = st.file_uploader(
        "Tải lên dữ liệu khách hàng (.csv hoặc .xlsx)",
        type=["csv", "xlsx", "xls"],
        help="Tệp phải chứa ít nhất 2 cột số để phân cụm.",
    )

    if not uploaded:
        st.info("👆 Tải lên một tệp để bắt đầu.")
        return

    try:
        df_raw = parse_upload(uploaded)
    except ValueError as e:
        st.error(str(e))
        return

    st.success(f"✅ Đã tải **{len(df_raw):,}** hàng × **{len(df_raw.columns)}** cột từ **{uploaded.name}**")

    # -------------------------------------------------------------------------
    # STEP 2: Data Preview & Missing Value Handling
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("👀 Bước 2 — Xem trước Dữ liệu")
    with st.expander("Hiển thị mẫu dữ liệu thô", expanded=False):
        st.dataframe(df_raw.head(50), width="stretch")

    df_clean, missing_report = handle_missing(df_raw.copy())

    if missing_report:
        st.warning(f"**Phát hiện và điền các giá trị bị thiếu trong {len(missing_report)} cột:**")
        miss_df = pd.DataFrame(missing_report).T.reset_index().rename(columns={"index": "Cột"})
        st.dataframe(miss_df, width="stretch")
    else:
        st.success("✅ Không phát hiện giá trị bị thiếu.")

    # -------------------------------------------------------------------------
    # STEP 3: Feature Selection & Clustering Configuration
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("⚙️ Bước 3 — Cấu hình Phân cụm")

    numeric_cols = get_numeric_columns(df_clean)
    if len(numeric_cols) < 2:
        st.error("❌ Tập dữ liệu phải có ít nhất 2 cột số để phân cụm.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        feature_cols = st.multiselect(
            "Chọn các đặc trưng để phân cụm",
            options=numeric_cols,
            default=numeric_cols[:min(len(numeric_cols), 6)],
            help="Chọn 2+ cột số.",
        )
    with col2:
        n_clusters = st.slider("Số lượng cụm", min_value=2, max_value=10, value=3)
    with col3:
        linkage_method = st.selectbox(
            "Phương pháp liên kết",
            options=["ward", "complete", "average", "single"],
            index=0,
            help="'ward' giảm thiểu phương sai trong cụm (khuyến nghị).",
        )

    show_anomalies = st.checkbox("🔍 Phát hiện bất thường (IsolationForest)", value=False)

    if len(feature_cols) < 2:
        st.warning("⚠️ Vui lòng chọn ít nhất 2 đặc trưng.")
        return

    # -------------------------------------------------------------------------
    # STEP 4: Run Analysis
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🚀 Bước 4 — Chạy phân tích")

    run_clicked = st.button("▶ Chạy phân tích phân cụm", width="stretch", type="primary")

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
        progress = st.progress(0, text="Bắt đầu phân tích…")
        status = st.empty()

        with st.spinner("Đang chạy phân tích…"):
            status.text("⚙️ Đang chuẩn hóa đặc trưng…")
            progress.progress(10, "Đang chuẩn hóa đặc trưng…")
            scaled, _ = scale_features(df_clean.to_json(), feature_cols)

            status.text("🔗 Đang tính toán ma trận liên kết…")
            progress.progress(20, "Đang tính toán liên kết…")
            linkage_matrix = compute_linkage(scaled, method=linkage_method)

            status.text("🎯 Đang gán cụm…")
            progress.progress(35, "Đang gán cụm…")
            labels = assign_clusters(scaled, n_clusters, linkage_method)

            status.text("📐 Đang tính toán PCA…")
            progress.progress(48, "Đang chiếu PCA…")
            pca_coords = compute_pca(scaled)

            anomaly_mask = None
            if show_anomalies:
                status.text("🔍 Đang phát hiện bất thường…")
                progress.progress(55, "Đang phát hiện bất thường…")
                anomaly_mask = detect_anomalies(scaled)

            status.text("📊 Đang tính toán thống kê cụm…")
            progress.progress(65, "Đang tính toán thống kê…")
            df_result = df_clean.copy()
            df_result["Cluster"] = labels
            profiles = cluster_profiles(df_result, "Cluster")
            silhouette = compute_silhouette(scaled, labels)

            status.text("🤖 Đang tạo thông tin chi tiết từ AI cho mỗi cụm…")
            progress.progress(75, "Phân tích AI cho mỗi cụm…")
            ai_insights = analyze_all_clusters(profiles, api_key=api_key)

            # NEW: overall cross-cluster analysis
            status.text("🧠 Đang tạo phân tích tổng thể liên cụm…")
            progress.progress(90, "Phân tích AI tổng thể…")
            overall_analysis = analyze_overall(
                profiles,
                ai_insights,
                api_key=api_key,
            )

            progress.progress(100, "Hoàn thành!")
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
        st.success("✅ Phân tích hoàn tất!")

    elif not cached_matches_current:
        if has_cached_result:
            st.info("⚠️ Cài đặt phân tích đã thay đổi. Nhấp vào **Chạy phân tích phân cụm** để làm mới kết quả trước khi lưu/xuất.")
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
        st.info("⚠️ Định dạng phân tích được lưu trong bộ nhớ cache đã lỗi thời. Nhấp vào **Chạy phân tích phân cụm** một lần để làm mới hình ảnh trực quan.")
        return

    # -------------------------------------------------------------------------
    # STEP 5: Metrics
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Bước 5 — Các chỉ số Tóm tắt")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Tổng số khách hàng", f"{len(df_result):,}")
    m2.metric("🔵 Số cụm", n_clusters)
    m3.metric("📐 Điểm Silhouette", f"{silhouette:.3f}" if silhouette >= 0 else "N/A")
    m4.metric("📊 Đặc trưng đã sử dụng", len(feature_cols))

    if show_anomalies and anomaly_mask is not None:
        n_anom = int(anomaly_mask.sum())
        st.info(f"⚠️ **{n_anom}** bất thường tiềm ẩn được phát hiện ({n_anom/len(df_result)*100:.1f}% khách hàng).")

    # -------------------------------------------------------------------------
    # STEP 6: Visualizations
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Bước 6 — Trực quan hóa")

    tab_dend, tab_pca, tab_heat, tab_dist, tab_box, tab_radar = st.tabs([
        "🌲 Biểu đồ cây", "🔵 PCA", "🌡️ Biểu đồ nhiệt",
        "📊 Phân phối", "📦 Biểu đồ hộp", "🕸️ Biểu đồ Radar"
    ])

    with tab_dend:
        st.subheader("Biểu đồ cây Phân cụm Phân cấp")
        fig_dend = plot_dendrogram(linkage_matrix)
        st.pyplot(fig_dend, width="stretch")

    with tab_pca:
        st.subheader("PCA — Chiếu cụm 2D")
        fig_pca = plot_pca(pca_coords, labels, anomaly_mask)
        st.plotly_chart(fig_pca, width="stretch")

    with tab_heat:
        st.subheader("Biểu đồ nhiệt Hồ sơ Đặc trưng")
        fig_heat = plot_heatmap(profiles)
        st.pyplot(fig_heat, width="stretch")

    with tab_dist:
        st.subheader("Phân phối khách hàng mỗi cụm")
        fig_dist = plot_cluster_distribution(labels)
        st.plotly_chart(fig_dist, width="stretch")

    with tab_box:
        st.subheader("Phân phối đặc trưng theo cụm")
        fig_box = plot_feature_boxplots(df_result, feature_cols)
        st.plotly_chart(fig_box, width="stretch")

    with tab_radar:
        st.subheader("So sánh cụm (Radar)")
        fig_radar = plot_cluster_comparison(profiles)
        st.plotly_chart(fig_radar, width="stretch")

    # -------------------------------------------------------------------------
    # STEP 7: AI Insights
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Bước 7 — Thông tin chi tiết về cụm từ AI")

    if not os.getenv("OPENAI_API_KEY"):
        st.info("💡 **Mẹo:** Đặt `OPENAI_API_KEY` trong tệp `.env` của bạn để có thông tin chi tiết từ OpenAI thực sự. Hiện đang hiển thị phân tích dựa trên quy tắc.")

    # --- 7A: Overall Cross-Cluster Analysis (NEW) ---
    if overall_analysis:
        st.markdown("### 🧠 Phân tích Tổng thể")

        # Key contrast callout
        key_contrast = overall_analysis.get("key_contrast", "")
        if key_contrast:
            st.info(f"**💡 Tương phản chính giữa các phân khúc:** {key_contrast}")

        # Cluster comparison table
        comparisons = overall_analysis.get("cluster_comparison", [])
        if comparisons:
            st.markdown("#### 🔍 So sánh liên cụm")
            comp_df = pd.DataFrame(comparisons)
            if len(comp_df.columns) == 2:
                comp_df.columns = ["Khía cạnh", "Tóm tắt"]
            elif len(comp_df.columns) >= 2:
                comp_df = comp_df.iloc[:, :2]
                comp_df.columns = ["Khía cạnh", "Tóm tắt"]
            st.table(comp_df)

        # Overall strategy
        overall_strategy = overall_analysis.get("overall_strategy", "")
        if overall_strategy:
            st.markdown("#### 🏢 Chiến lược kinh doanh tổng thể")
            st.success(overall_strategy)

        # Priority actions
        priority_actions = overall_analysis.get("priority_actions", [])
        if priority_actions:
            st.markdown("#### 🎯 Hành động ưu tiên")
            for i, action in enumerate(priority_actions, start=1):
                st.markdown(f"**{i}.** {action}")

        st.markdown("---")

    # --- 7B: Per-Cluster Insights (existing) ---
    st.markdown("### 📋 Phân tích chi tiết mỗi cụm")
    for cluster_id, insight in ai_insights.items():
        segment_name = insight.get("segment_name", f"Cụm {cluster_id}")
        with st.expander(f"🔵 Cụm {cluster_id}: **{segment_name}**", expanded=True):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown("**📝 Mô tả**")
                st.write(insight.get("description", ""))
                st.markdown("**🧠 Thông tin chi tiết về hành vi**")
                st.write(insight.get("behavior_insight", ""))
            with col_b:
                st.markdown("**🎯 Chiến lược tiếp thị**")
                st.write(insight.get("marketing_strategy", ""))
                campaigns = insight.get("suggested_campaigns", [])
                if campaigns:
                    st.markdown("**📣 Các chiến dịch được đề xuất**")
                    for c in campaigns:
                        st.markdown(f"  - {c}")

    # -------------------------------------------------------------------------
    # STEP 8: Save to Database
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💾 Bước 8 — Lưu Phân tích")

    if st.button("💾 Lưu vào Lịch sử", width="stretch"):
        with st.spinner("Đang lưu…"):
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
        st.success(f"✅ Phân tích đã được lưu! (ID: `{analysis_id}`)")

    # -------------------------------------------------------------------------
    # STEP 9: Export
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Bước 9 — Xuất Kết quả")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Tải xuống CSV",
            data=csv_bytes,
            file_name=f"clusters_{uploaded.name.rsplit('.', 1)[0]}.csv",
            mime="text/csv",
            width="stretch",
        )

    with exp2:
        excel_bytes = _df_to_excel(df_result)
        st.download_button(
            label="📊 Tải xuống Excel",
            data=excel_bytes,
            file_name=f"clusters_{uploaded.name.rsplit('.', 1)[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
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
            label="📋 Tải xuống Báo cáo PDF",
            data=pdf_bytes,
            file_name=f"report_{uploaded.name.rsplit('.', 1)[0]}.pdf",
            mime="application/pdf",
            width="stretch",
        )
