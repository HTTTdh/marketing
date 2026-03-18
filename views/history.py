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
    st.title("📚 Analysis History")
    st.markdown("Browse, review, and manage all previously saved customer segmentation analyses.")

    records = list_analyses()

    if not records:
        st.info("📭 No analyses saved yet. Run an analysis from the **Analyze** page and click *Save to History*.")
        return

    st.metric("Total Saved Analyses", len(records))
    st.markdown("---")

    for rec in records:
        header = (
            f"📁 **{rec['filename']}**  |  "
            f"🗓 {rec['created_at']}  |  "
            f"👥 {rec['number_of_customers']:,} customers  |  "
            f"🔵 {rec['number_of_clusters']} clusters  |  "
            f"🔗 Method: {rec['clustering_method']}"
        )

        with st.expander(header, expanded=False):
            act_col, del_col = st.columns([6, 1])
            with del_col:
                if st.button("🗑️ Delete", key=f"del_{rec['id']}", type="secondary"):
                    if delete_analysis(rec["id"]):
                        st.success("Deleted!")
                        st.rerun()
                    else:
                        st.error("Could not delete.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Customers", f"{rec['number_of_customers']:,}")
            m2.metric("Clusters", rec["number_of_clusters"])
            m3.metric("Method", rec["clustering_method"].capitalize())
            m4.metric("Saved", rec["created_at"].split(" ")[0])

            cluster_stats = rec.get("cluster_stats", {})
            ai_insights = rec.get("ai_insights", {})

            if cluster_stats:
                st.markdown("### 📊 Cluster Feature Profiles")
                try:
                    profiles = pd.DataFrame(cluster_stats).T
                    profiles.index.name = "Cluster"
                    profiles = profiles.apply(pd.to_numeric, errors="coerce")
                    st.dataframe(profiles.round(3), width='stretch')

                    tab_heat, tab_radar = st.tabs(["🌡️ Heatmap", "🕸️ Radar Comparison"])
                    with tab_heat:
                        fig_heat = plot_heatmap(profiles)
                        st.pyplot(fig_heat, width='stretch')
                    with tab_radar:
                        fig_radar = plot_cluster_comparison(profiles)
                        st.plotly_chart(fig_radar, width='stretch')
                except Exception as e:
                    st.warning(f"Could not render charts: {e}")

            if ai_insights:
                st.markdown("### 🤖 AI Insights")
                for cluster_id, insight in ai_insights.items():
                    segment_name = insight.get("segment_name", f"Cluster {cluster_id}")
                    st.markdown(f"#### 🔵 Cluster {cluster_id}: {segment_name}")
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
                    st.markdown("---")

            # with st.expander("🔧 Raw JSON data", expanded=False):
            #     st.json(rec)
            if st.toggle("Show raw JSON data", key=f"raw_{rec['id']}", value=False):
                st.json(rec)
