"""
Streamlit Dashboard -- Trader Performance vs Market Sentiment
=============================================================
Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Trader vs Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3a3a5c;
        padding: 15px 20px;
        border-radius: 12px;
        color: white;
    }
    div[data-testid="stMetric"] label {color: #a0a0c0 !important;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {color: #ffffff !important;}
    h1, h2, h3 {color: #e0e0ff;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2f;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #a0a0c0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3a3a5c !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

FEAR_C, GREED_C, NEUT_C = "#e74c3c", "#27ae60", "#3498db"
SENT_COLORS = {"Fear": FEAR_C, "Greed": GREED_C, "Neutral": NEUT_C}

# ---------- Load Data ----------
@st.cache_data
def load_merged():
    return pd.read_csv("merged_daily_analysis.csv", parse_dates=["date"])

@st.cache_data
def load_clusters():
    return pd.read_csv("trader_clusters.csv")

@st.cache_data
def load_segments():
    return pd.read_csv("trader_segments.csv")

try:
    df = load_merged()
    clusters = load_clusters()
    segments = load_segments()
except FileNotFoundError as e:
    st.error(f"Data files not found. Run `python trader_sentiment_analysis.py` first.\n\n{e}")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.title("Filters")
selected_sentiments = st.sidebar.multiselect(
    "Sentiment", options=["Fear", "Neutral", "Greed"],
    default=["Fear", "Neutral", "Greed"])
selected_segments = st.sidebar.multiselect(
    "Leverage Segment", options=df["Seg_Leverage"].dropna().unique().tolist(),
    default=df["Seg_Leverage"].dropna().unique().tolist())

mask = (df["Sentiment"].isin(selected_sentiments)) & \
       (df["Seg_Leverage"].isin(selected_segments))
fdf = df[mask].copy()

# ======================================================================
# HEADER
# ======================================================================
st.title("Bitcoin Sentiment vs Trader Performance")
st.caption("Analyzing Hyperliquid trader behavior across Fear & Greed regimes")

# ---------- KPI row ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PnL", f"${fdf['Daily_PnL'].sum():,.0f}")
c2.metric("Avg Win Rate", f"{fdf['Win_Rate'].mean():.1%}")
c3.metric("Unique Traders", f"{fdf['Account'].nunique()}")
c4.metric("Total Trades", f"{fdf['Total_Trades'].sum():,.0f}")
c5.metric("Date Range", f"{fdf['date'].min().date()} - {fdf['date'].max().date()}")

st.markdown("---")

# ======================================================================
# TABS
# ======================================================================
tab_a, tab_b1, tab_b2, tab_b3, tab_c, tab_bonus = st.tabs([
    "Part A: Data Overview",
    "B.1  Performance",
    "B.2  Behavior",
    "B.3  Segments",
    "Part C: Strategy Rules",
    "Bonus: Model & Clusters",
])

# ---------- TAB A  -- Data Overview ----------
with tab_a:
    st.header("Part A -- Data Preparation Summary")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Historical Trader Data")
        st.write(f"- **Rows**: {df.shape[0]:,}")
        st.write(f"- **Columns**: {df.shape[1]}")
        st.write(f"- **Unique Accounts**: {df['Account'].nunique()}")
        st.write(f"- **Date Range**: {df['date'].min().date()} to {df['date'].max().date()}")
    with col_r:
        st.subheader("Sentiment Distribution")
        sent_counts = df.groupby("Sentiment")["date"].nunique().reset_index()
        sent_counts.columns = ["Sentiment", "Days"]
        fig = px.pie(sent_counts, names="Sentiment", values="Days",
                     color="Sentiment",
                     color_discrete_map=SENT_COLORS,
                     hole=0.45)
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Key Metrics Computed")
    st.dataframe(
        df[["date", "Account", "Daily_PnL", "Win_Rate", "Total_Trades",
            "Avg_Trade_Size_USD", "Long_Short_Ratio", "Leverage_Proxy",
            "Sentiment"]].head(20).style.format({
                "Daily_PnL": "${:,.2f}", "Win_Rate": "{:.2%}",
                "Avg_Trade_Size_USD": "${:,.2f}",
                "Long_Short_Ratio": "{:.2f}", "Leverage_Proxy": "{:.2f}"}),
        width="stretch")

# ---------- TAB B.1  -- Performance ----------
with tab_b1:
    st.header("B.1 -- Performance: Fear vs Greed")

    perf = fdf.groupby("Sentiment").agg(
        Mean_PnL=("Daily_PnL", "mean"),
        Median_PnL=("Daily_PnL", "median"),
        Std_PnL=("Daily_PnL", "std"),
        Mean_WR=("Win_Rate", "mean"),
        Count=("Daily_PnL", "count"),
    ).reindex(["Fear", "Neutral", "Greed"]).dropna()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(perf.reset_index(), x="Sentiment", y="Mean_PnL",
                     color="Sentiment", color_discrete_map=SENT_COLORS,
                     title="Average Daily PnL by Sentiment")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(perf.reset_index(), x="Sentiment", y="Mean_WR",
                     color="Sentiment", color_discrete_map=SENT_COLORS,
                     title="Average Win Rate by Sentiment")
        fig.update_layout(showlegend=False, yaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")

    # PnL distribution
    st.subheader("PnL Distribution by Sentiment")
    fig = go.Figure()
    for s in ["Fear", "Neutral", "Greed"]:
        vals = fdf.loc[fdf["Sentiment"] == s, "Daily_PnL"].clip(-50000, 50000)
        fig.add_trace(go.Histogram(x=vals, name=s,
                                    marker_color=SENT_COLORS[s],
                                    opacity=0.5, nbinsx=80))
    fig.update_layout(barmode="overlay", xaxis_title="Daily PnL ($)",
                      yaxis_title="Frequency", height=400)
    st.plotly_chart(fig, width="stretch")

    # Drawdown
    st.subheader("Drawdown Proxy by Sentiment")
    if "Drawdown" in fdf.columns:
        dd_acct = fdf.groupby(["Account", "Sentiment"])["Drawdown"].min().reset_index()
        fig = px.box(dd_acct, x="Sentiment", y="Drawdown",
                     color="Sentiment", color_discrete_map=SENT_COLORS,
                     category_orders={"Sentiment": ["Fear", "Neutral", "Greed"]})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(perf.style.format({
        "Mean_PnL": "${:,.2f}", "Median_PnL": "${:,.2f}",
        "Std_PnL": "${:,.2f}", "Mean_WR": "{:.3f}",
        "Count": "{:,.0f}"}), width="stretch")

# ---------- TAB B.2  -- Behavior ----------
with tab_b2:
    st.header("B.2 -- Behavioral Changes by Sentiment")

    beh = fdf.groupby("Sentiment").agg(
        Avg_Trades=("Total_Trades", "mean"),
        Avg_Size=("Avg_Trade_Size_USD", "mean"),
        Avg_LS=("Long_Short_Ratio", "mean"),
        Avg_Volume=("Total_Volume_USD", "mean"),
        Avg_Lev=("Leverage_Proxy", "mean"),
    ).reindex(["Fear", "Neutral", "Greed"]).dropna().reset_index()

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Trade Frequency", "Trade Size (USD)",
                                        "Long/Short Ratio", "Leverage Proxy"])

    for i, (col, row_i, col_i) in enumerate([
        ("Avg_Trades", 1, 1), ("Avg_Size", 1, 2),
        ("Avg_LS", 2, 1), ("Avg_Lev", 2, 2)]):
        fig.add_trace(go.Bar(
            x=beh["Sentiment"], y=beh[col],
            marker_color=[SENT_COLORS.get(s, "#888") for s in beh["Sentiment"]],
            showlegend=False
        ), row=row_i, col=col_i)

    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

    # Long/Short bias
    st.subheader("Long vs Short Trades by Sentiment")
    ls_data = fdf.groupby("Sentiment")[["Long_Trades", "Short_Trades"]].sum().reindex(
        ["Fear", "Neutral", "Greed"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ls_data["Sentiment"], y=ls_data["Long_Trades"],
                         name="Long", marker_color="#27ae60"))
    fig.add_trace(go.Bar(x=ls_data["Sentiment"], y=ls_data["Short_Trades"],
                         name="Short", marker_color="#e74c3c"))
    fig.update_layout(barmode="group", height=400)
    st.plotly_chart(fig, width="stretch")

# ---------- TAB B.3  -- Segments ----------
with tab_b3:
    st.header("B.3 -- Trader Segmentation")

    seg_choice = st.selectbox("Select Segment Type",
                               ["Seg_Leverage", "Seg_Frequency", "Seg_Consistency"])

    # Segment PnL by sentiment
    seg_perf = fdf.groupby([seg_choice, "Sentiment"])["Daily_PnL"].mean().reset_index()
    fig = px.bar(seg_perf, x="Sentiment", y="Daily_PnL", color=seg_choice,
                 barmode="group", title=f"Avg Daily PnL by {seg_choice}",
                 category_orders={"Sentiment": ["Fear", "Neutral", "Greed"]})
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")

    # Win rate heatmap
    st.subheader("Win Rate Heatmap")
    wr_pivot = fdf.groupby([seg_choice, "Sentiment"])["Win_Rate"].mean().unstack()
    cols_order = [c for c in ["Fear", "Neutral", "Greed"] if c in wr_pivot.columns]
    wr_pivot = wr_pivot[cols_order]
    fig = px.imshow(wr_pivot.values, x=cols_order, y=wr_pivot.index.tolist(),
                    text_auto=".3f", color_continuous_scale="RdYlGn",
                    zmin=0.3, zmax=0.95,
                    labels=dict(x="Sentiment", y=seg_choice, color="Win Rate"))
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

    # Segment summary table
    st.subheader("Segment Summary")
    seg_summary = fdf.groupby(seg_choice).agg(
        Traders=("Account", "nunique"),
        Avg_PnL=("Daily_PnL", "mean"),
        Avg_WR=("Win_Rate", "mean"),
        Avg_Size=("Avg_Trade_Size_USD", "mean"),
        Total_Trades=("Total_Trades", "sum"),
    ).round(2)
    st.dataframe(seg_summary.style.format({
        "Avg_PnL": "${:,.2f}", "Avg_WR": "{:.3f}",
        "Avg_Size": "${:,.2f}", "Total_Trades": "{:,.0f}"}),
        width="stretch")

# ---------- TAB C  -- Strategy Rules ----------
with tab_c:
    st.header("Part C -- Actionable Strategy Rules")

    # Compute data-driven values
    ll_fear = fdf.loc[(fdf["Seg_Leverage"] == "Low Leverage") &
                       (fdf["Sentiment"] == "Fear"), "Daily_PnL"].mean()
    ll_greed = fdf.loc[(fdf["Seg_Leverage"] == "Low Leverage") &
                        (fdf["Sentiment"] == "Greed"), "Daily_PnL"].mean()
    inf_fear_wr = fdf.loc[(fdf["Seg_Frequency"] == "Infrequent") &
                           (fdf["Sentiment"] == "Fear"), "Win_Rate"].mean()
    inf_greed_wr = fdf.loc[(fdf["Seg_Frequency"] == "Infrequent") &
                            (fdf["Sentiment"] == "Greed"), "Win_Rate"].mean()

    st.success(f"""
    **RULE 1: During Fear periods, Low-Leverage traders should increase position sizes.**

    - **Segment**: Low Leverage (below-median avg trade size)
    - **Evidence**: Low-lev traders avg PnL during Fear = ${ll_fear:,.2f} vs Greed = ${ll_greed:,.2f}
      ({((ll_fear - ll_greed) / (abs(ll_greed) + 1e-9)) * 100:+.0f}% difference)
    - **Action**: When Fear & Greed Index < 40, increase trade frequency by 10-20%.
      These traders thrive in volatile, fear-driven conditions.
    """)

    st.warning(f"""
    **RULE 2: During Fear periods, Infrequent traders should reduce activity and
    wait for Neutral/Greed to re-enter.**

    - **Segment**: Infrequent traders (below-median total trade count)
    - **Evidence**: Infrequent trader win rate Fear = {inf_fear_wr:.3f} vs Greed = {inf_greed_wr:.3f}.
      They likely panic-trade during Fear.
    - **Action**: Set a minimum F&G threshold of 45 before new entries.
      Use limit orders instead of market orders during Fear.
    """)

# ---------- TAB BONUS ----------
with tab_bonus:
    st.header("Bonus -- Predictive Model & Clustering")

    tab_model, tab_cluster = st.tabs(["Predictive Model", "Clustering"])

    with tab_model:
        st.subheader("Next-Day PnL Direction Prediction")

        # Model comparison table
        mc_path = "model_comparison.csv"
        if os.path.exists(mc_path):
            mc = pd.read_csv(mc_path)
            st.markdown("#### Model Comparison")

            # Highlight best model
            best_idx = mc["Balanced_Accuracy"].idxmax()
            best_model = mc.loc[best_idx, "Model"]
            st.info(f"**Best Model**: {best_model} (selected by highest Balanced Accuracy)")

            # Metrics as cards
            m1, m2, m3, m4 = st.columns(4)
            for i, (_, row) in enumerate(mc.iterrows()):
                col = [m1, m2, m3, m4][i]
                col.metric(
                    row["Model"],
                    f"{row['Balanced_Accuracy']:.1%}",
                    f"F1: {row['Macro_F1']:.3f} | AUC: {row['ROC_AUC']:.3f}",
                )

            st.dataframe(mc.style.format({
                "Threshold": "{:.2f}", "Accuracy": "{:.4f}",
                "Balanced_Accuracy": "{:.4f}", "Macro_F1": "{:.4f}",
                "ROC_AUC": "{:.4f}"}), width="stretch")

        # Charts
        chart_path = os.path.join("charts", "Bonus_model_results.png")
        if os.path.exists(chart_path):
            st.image(chart_path, width="stretch")
        else:
            st.info("Run `python trader_sentiment_analysis.py` to generate model charts.")

        st.markdown("""
        **Methodology:**
        - **Target**: Whether next-day aggregate PnL is positive (Profit) or negative (Loss)
        - **Features (34)**: Fear & Greed value, momentum (3d/5d/10d), rolling PnL stats, EMA,
          PnL streak, lag features (1d-3d), L/S ratio, leverage proxy, sentiment dummies, day of week
        - **Class Imbalance**: Handled via SMOTE oversampling on training set
        - **Models**: Logistic Regression, Random Forest (400 trees), Gradient Boosting (300 trees), XGBoost (300 trees)
        - **Threshold Tuning**: Optimal threshold per model via macro-F1 sweep (0.30 - 0.70)
        - **Evaluation**: Balanced Accuracy, Macro F1, ROC-AUC (chronological 80/20 split)
        - **Cross-Validation**: 5-fold stratified on training data for robustness estimate
        """)

    with tab_cluster:
        st.subheader("Trader Behavioral Archetypes (K-Means)")

        # Display cluster charts
        for chart_name in ["Bonus_clustering.png", "Bonus_cluster_profiles.png"]:
            p = os.path.join("charts", chart_name)
            if os.path.exists(p):
                st.image(p, width="stretch")

        if not clusters.empty:
            st.subheader("Cluster Summary")
            cl_summary = clusters.groupby("Cluster").agg(
                Traders=("Account", "count"),
                Avg_PnL=("avg_pnl", "mean"),
                Avg_WR=("avg_wr", "mean"),
                Avg_Size=("avg_size", "mean"),
                Total_Trades=("total_trades", "mean"),
                Avg_LS=("avg_ls", "mean"),
            ).round(2)
            st.dataframe(cl_summary.style.format({
                "Avg_PnL": "${:,.2f}", "Avg_WR": "{:.3f}",
                "Avg_Size": "${:,.2f}", "Total_Trades": "{:,.0f}",
                "Avg_LS": "{:.2f}"}), width="stretch")

            # Interactive PCA scatter
            if "PCA1" in clusters.columns:
                fig = px.scatter(clusters, x="PCA1", y="PCA2",
                                 color=clusters["Cluster"].astype(str),
                                 hover_data=["Account", "avg_pnl", "avg_wr",
                                             "total_trades"],
                                 title="Trader Clusters (PCA Projection)",
                                 color_discrete_sequence=px.colors.qualitative.Set1)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("Data: Hyperliquid historical trades + Bitcoin Fear & Greed Index")
