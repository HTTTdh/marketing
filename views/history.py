"""
views/history.py
History page: list, view details (with full visualizations), and delete saved analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from services.database import list_analyses, delete_analysis, init_db
from services.visualization import (
    plot_dendrogram,
    plot_pca,
    plot_heatmap,
    plot_cluster_distribution,
    plot_cluster_comparison,
    plot_feature_boxplots,
)


def render():
    init_db()
    st.title("📚 Lịch sử Phân tích")
    st.markdown("Duyệt, xem lại và quản lý tất cả các phân tích phân khúc khách hàng đã lưu trước đó.")

    records = list_analyses()

    if not records:
        st.info("📭 Chưa có phân tích nào được lưu. Chạy phân tích từ trang **Phân tích** và nhấp vào *Lưu vào Lịch sử*.")
        return

    st.metric("Tổng số Phân tích đã lưu", len(records))
    st.markdown("---")

    for rec in records:
        header = (
            f"📁 **{rec['filename']}**  |  "
            f"🗓 {rec['created_at']}  |  "
            f"👥 {rec['number_of_customers']:,} khách hàng  |  "
            f"🔵 {rec['number_of_clusters']} cụm  |  "
            f"🔗 Phương pháp: {rec['clustering_method']}"
        )

        with st.expander(header, expanded=False):
            act_col, del_col = st.columns([6, 1])
            with del_col:
                if st.button("🗑️ Xóa", key=f"del_{rec['id']}", type="secondary"):
                    if delete_analysis(rec["id"]):
                        st.success("Đã xóa!")
                        st.rerun()
                    else:
                        st.error("Không thể xóa.")

            # -----------------------------------------------------------------
            # Metrics
            # -----------------------------------------------------------------
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Khách hàng", f"{rec['number_of_customers']:,}")
            m2.metric("Số cụm", rec["number_of_clusters"])
            m3.metric("Phương pháp", rec["clustering_method"].capitalize())
            sil = rec.get("silhouette_score")
            m4.metric("Điểm Silhouette", f"{sil:.3f}" if sil is not None else "N/A")

            # -----------------------------------------------------------------
            # Prepare data from saved record
            # -----------------------------------------------------------------
            cluster_stats = rec.get("cluster_stats", {})
            ai_insights = rec.get("ai_insights", {})
            overall_analysis = rec.get("overall_analysis") or {}
            labels_list = rec.get("labels")
            pca_list = rec.get("pca_coords")
            linkage_list = rec.get("linkage_matrix")
            feature_cols = rec.get("feature_cols")
            df_result_dict = rec.get("df_result")

            labels = np.array(labels_list) if labels_list is not None else None
            pca_coords = np.array(pca_list) if pca_list is not None else None
            linkage_matrix = np.array(linkage_list) if linkage_list is not None else None
            df_result = pd.DataFrame(df_result_dict) if df_result_dict is not None else None

            has_full_data = all(x is not None for x in [labels, pca_coords, linkage_matrix, feature_cols, df_result])

            # -----------------------------------------------------------------
            # Cluster Stats Table
            # -----------------------------------------------------------------
            if cluster_stats:
                st.markdown("### 📊 Hồ sơ Đặc trưng Cụm")
                try:
                    profiles = pd.DataFrame(cluster_stats).T
                    profiles.index.name = "Cụm"
                    profiles = profiles.apply(pd.to_numeric, errors="coerce")
                    st.dataframe(profiles.round(3), width="stretch")
                except Exception as e:
                    profiles = None
                    st.warning(f"Không thể hiển thị bảng hồ sơ: {e}")
            else:
                profiles = None

            # -----------------------------------------------------------------
            # Visualizations — all 6 tabs (matching analyze.py)
            # -----------------------------------------------------------------
            if has_full_data and profiles is not None:
                st.markdown("### 📊 Trực quan hóa")
                tab_dend, tab_pca, tab_heat, tab_dist, tab_box, tab_radar = st.tabs([
                    "🌲 Biểu đồ cây", "🔵 PCA", "🌡️ Biểu đồ nhiệt",
                    "📊 Phân phối", "📦 Biểu đồ hộp", "🕸️ Biểu đồ Radar"
                ])

                with tab_dend:
                    st.subheader("Biểu đồ cây Phân cụm Phân cấp")
                    try:
                        fig_dend = plot_dendrogram(linkage_matrix)
                        st.pyplot(fig_dend, width="stretch")
                    except Exception as e:
                        st.warning(f"Không thể hiển thị biểu đồ cây: {e}")

                with tab_pca:
                    st.subheader("PCA — Chiếu cụm 2D")
                    try:
                        fig_pca = plot_pca(pca_coords, labels)
                        st.plotly_chart(fig_pca, width="stretch")
                    except Exception as e:
                        st.warning(f"Không thể hiển thị PCA: {e}")

                with tab_heat:
                    st.subheader("Biểu đồ nhiệt Hồ sơ Đặc trưng")
                    try:
                        fig_heat = plot_heatmap(profiles)
                        st.pyplot(fig_heat, width="stretch")
                    except Exception as e:
                        st.warning(f"Không thể hiển thị biểu đồ nhiệt: {e}")

                with tab_dist:
                    st.subheader("Phân phối khách hàng mỗi cụm")
                    try:
                        fig_dist = plot_cluster_distribution(labels)
                        st.plotly_chart(fig_dist, width="stretch")
                    except Exception as e:
                        st.warning(f"Không thể hiển thị phân phối: {e}")

                with tab_box:
                    st.subheader("Phân phối đặc trưng theo cụm")
                    try:
                        fig_box = plot_feature_boxplots(df_result, feature_cols)
                        st.plotly_chart(fig_box, width="stretch")
                    except Exception as e:
                        st.warning(f"Không thể hiển thị biểu đồ hộp: {e}")

                with tab_radar:
                    st.subheader("So sánh cụm (Radar)")
                    try:
                        fig_radar = plot_cluster_comparison(profiles)
                        st.plotly_chart(fig_radar, width="stretch")
                    except Exception as e:
                        st.warning(f"Không thể hiển thị radar: {e}")

            elif profiles is not None:
                # Fallback for old records without full data
                st.markdown("### 📊 Trực quan hóa")
                st.caption("⚠️ Bản ghi cũ — chỉ hiển thị biểu đồ nhiệt và radar. Lưu lại phân tích để có đầy đủ biểu đồ.")
                tab_heat, tab_radar = st.tabs(["🌡️ Biểu đồ nhiệt", "🕸️ Biểu đồ Radar"])
                with tab_heat:
                    try:
                        fig_heat = plot_heatmap(profiles)
                        st.pyplot(fig_heat, width="stretch")
                    except Exception as e:
                        st.warning(f"Không thể hiển thị biểu đồ: {e}")
                with tab_radar:
                    try:
                        fig_radar = plot_cluster_comparison(profiles)
                        st.plotly_chart(fig_radar, width="stretch")
                    except Exception as e:
                        st.warning(f"Không thể hiển thị biểu đồ: {e}")

            # -----------------------------------------------------------------
            # AI Insights — Overall Analysis
            # -----------------------------------------------------------------
            if overall_analysis:
                st.markdown("### 🧠 Phân tích Tổng thể")

                key_contrast = overall_analysis.get("key_contrast", "")
                if key_contrast:
                    st.info(f"**💡 Tương phản chính giữa các phân khúc:** {key_contrast}")

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

                overall_strategy = overall_analysis.get("overall_strategy", "")
                if overall_strategy:
                    st.markdown("#### 🏢 Chiến lược kinh doanh tổng thể")
                    st.success(overall_strategy)

                priority_actions = overall_analysis.get("priority_actions", [])
                if priority_actions:
                    st.markdown("#### 🎯 Hành động ưu tiên")
                    for i, action in enumerate(priority_actions, start=1):
                        st.markdown(f"**{i}.** {action}")

                st.markdown("---")

            # -----------------------------------------------------------------
            # AI Insights — Per-Cluster
            # -----------------------------------------------------------------
            if ai_insights:
                st.markdown("### 🤖 Thông tin chi tiết từ AI")
                for cluster_id, insight in ai_insights.items():
                    segment_name = insight.get("segment_name", f"Cụm {cluster_id}")
                    st.markdown(f"#### 🔵 Cụm {cluster_id}: {segment_name}")
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
                    st.markdown("---")

            if st.toggle("Hiển thị dữ liệu JSON thô", key=f"raw_{rec['id']}", value=False):
                st.json(rec)
