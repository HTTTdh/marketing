"""
views/history.py
History page: list, view details, and delete saved analyses.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.database import list_analyses, delete_analysis, init_db
from services.visualization import plot_heatmap, plot_cluster_comparison


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

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Khách hàng", f"{rec['number_of_customers']:,}")
            m2.metric("Số cụm", rec["number_of_clusters"])
            m3.metric("Phương pháp", rec["clustering_method"].capitalize())
            m4.metric("Đã lưu", rec["created_at"].split(" ")[0])

            cluster_stats = rec.get("cluster_stats", {})
            ai_insights = rec.get("ai_insights", {})

            if cluster_stats:
                st.markdown("### 📊 Hồ sơ Đặc trưng Cụm")
                try:
                    profiles = pd.DataFrame(cluster_stats).T
                    profiles.index.name = "Cụm"
                    profiles = profiles.apply(pd.to_numeric, errors="coerce")
                    st.dataframe(profiles.round(3), width="stretch")

                    tab_heat, tab_radar = st.tabs(["🌡️ Biểu đồ nhiệt", "🕸️ So sánh Radar"])
                    with tab_heat:
                        fig_heat = plot_heatmap(profiles)
                        st.pyplot(fig_heat, width="stretch")
                    with tab_radar:
                        fig_radar = plot_cluster_comparison(profiles)
                        st.plotly_chart(fig_radar, width="stretch")
                except Exception as e:
                    st.warning(f"Không thể hiển thị biểu đồ: {e}")

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
