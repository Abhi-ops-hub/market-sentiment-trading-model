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
    page_title="Sentiment x Performance",
    page_icon="https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f4c8.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  PREMIUM CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* -------- global -------- */
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 1.5rem 2rem 1rem 2rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
[data-testid="stSidebar"] * { color: #c0c0e0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #ffffff !important; }

/* -------- header banner -------- */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero-banner h1 {
    font-size: 2.4rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #a78bfa, #818cf8, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-banner p { color: #9ca3af; font-size: 1.05rem; margin: 6px 0 0 0; }

/* -------- KPI cards -------- */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #16162a 0%, #1e1e38 100%);
    border: 1px solid rgba(139, 92, 246, 0.2);
    padding: 18px 22px; border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
}
div[data-testid="stMetric"] label {
    color: #8b8bb5 !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #e2e8f0 !important; font-weight: 700 !important; font-size: 1.6rem !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    color: #a78bfa !important; font-size: 0.72rem !important;
}

/* -------- tabs -------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #1e1e38; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 10px 10px 0 0;
    padding: 12px 22px; color: #6b6b99; font-weight: 500;
    border: 1px solid transparent; transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #a78bfa; background: rgba(139, 92, 246, 0.05); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(139,92,246,0.12) 0%, transparent 100%) !important;
    color: #a78bfa !important; font-weight: 700 !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* -------- headings -------- */
h1 { color: #e2e8f0 !important; font-weight: 800 !important; }
h2 { color: #c4b5fd !important; font-weight: 700 !important; font-size: 1.5rem !important; }
h3 { color: #a5b4fc !important; font-weight: 600 !important; }

/* -------- dataframes -------- */
[data-testid="stDataFrame"] { border: 1px solid #2a2a4a; border-radius: 10px; overflow: hidden; }

/* -------- alerts -------- */
div[data-testid="stAlert"] { border-radius: 12px !important; }

/* -------- dividers -------- */
hr { border-color: #2a2a4a !important; margin: 16px 0 !important; }

/* -------- custom insight card -------- */
.insight-card {
    background: linear-gradient(135deg, #16162a, #1e1e38);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 14px; padding: 24px 28px; margin: 10px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.insight-card h4 { color: #a78bfa; margin: 0 0 8px 0; font-size: 1.1rem; }
.insight-card p { color: #9ca3af; margin: 0; line-height: 1.6; }
.insight-card .metric-highlight {
    color: #60a5fa; font-weight: 700; font-size: 1.3rem;
}

/* -------- rule cards -------- */
.rule-card {
    border-radius: 14px; padding: 28px 32px; margin: 12px 0;
    border-left: 4px solid; box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.rule-card.green { background: linear-gradient(135deg, #0a2e1a, #132e1e); border-left-color: #34d399; }
.rule-card.amber { background: linear-gradient(135deg, #2e1f0a, #2e2413); border-left-color: #fbbf24; }
.rule-card h4 { margin: 0 0 12px 0; font-size: 1.15rem; }
.rule-card.green h4 { color: #34d399; }
.rule-card.amber h4 { color: #fbbf24; }
.rule-card p, .rule-card li { color: #d1d5db; line-height: 1.7; font-size: 0.95rem; }
.rule-card ul { padding-left: 18px; }
.rule-card strong { color: #f3f4f6; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.06)"
FONT_COLOR = "#9ca3af"

FEAR_C  = "#ef4444"
GREED_C = "#22c55e"
NEUT_C  = "#3b82f6"
ACCENT  = "#a78bfa"
SENT_COLORS = {"Fear": FEAR_C, "Greed": GREED_C, "Neutral": NEUT_C}

def apply_theme(fig, height=None):
    """Apply the premium dark theme to any plotly figure."""
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=13),
        title_font=dict(size=16, color="#e2e8f0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af")),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    if height:
        fig.update_layout(height=height)
    return fig

# ═══════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Filters")
    st.markdown("---")
    selected_sentiments = st.multiselect(
        "Sentiment Regime",
        options=["Fear", "Neutral", "Greed"],
        default=["Fear", "Neutral", "Greed"])
    selected_segments = st.multiselect(
        "Leverage Segment",
        options=df["Seg_Leverage"].dropna().unique().tolist(),
        default=df["Seg_Leverage"].dropna().unique().tolist())
    st.markdown("---")
    st.caption("Built with Streamlit + Plotly")

mask = (df["Sentiment"].isin(selected_sentiments)) & \
       (df["Seg_Leverage"].isin(selected_segments))
fdf = df[mask].copy()

# ═══════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>Bitcoin Sentiment vs Trader Performance</h1>
    <p>Deep analysis of Hyperliquid trader behavior across Fear &amp; Greed regimes &mdash; 211K+ trades, 32 accounts, 4 ML models</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  KPI ROW
# ═══════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PnL", f"${fdf['Daily_PnL'].sum():,.0f}")
c2.metric("Avg Win Rate", f"{fdf['Win_Rate'].mean():.1%}")
c3.metric("Unique Traders", f"{fdf['Account'].nunique()}")
c4.metric("Total Trades", f"{fdf['Total_Trades'].sum():,.0f}")
c5.metric("Date Range", f"{fdf['date'].min().date()} to {fdf['date'].max().date()}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════
tab_a, tab_b1, tab_b2, tab_b3, tab_c, tab_bonus = st.tabs([
    "Data Overview",
    "Performance",
    "Behavior",
    "Segments",
    "Strategy Rules",
    "Model & Clusters",
])

# ──────── TAB A ────────
with tab_a:
    st.markdown("## Data Preparation Summary")
    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown("### Dataset Stats")
        st.markdown(f"""
        <div class="insight-card">
            <h4>Historical Trader Data</h4>
            <p>
                <span class="metric-highlight">{df.shape[0]:,}</span> daily records &bull;
                <span class="metric-highlight">{df['Account'].nunique()}</span> unique traders<br>
                {df.shape[1]} columns &bull;
                {df['date'].min().date()} to {df['date'].max().date()}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Sentiment Distribution")
        sent_counts = df.groupby("Sentiment")["date"].nunique().reset_index()
        sent_counts.columns = ["Sentiment", "Days"]
        fig = px.pie(sent_counts, names="Sentiment", values="Days",
                     color="Sentiment", color_discrete_map=SENT_COLORS, hole=0.55)
        fig.update_traces(textfont_size=14, textfont_color="white",
                          marker=dict(line=dict(color="#0f0f1a", width=2)))
        apply_theme(fig, height=320)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Sample Merged Data")
    st.dataframe(
        df[["date", "Account", "Daily_PnL", "Win_Rate", "Total_Trades",
            "Avg_Trade_Size_USD", "Long_Short_Ratio", "Leverage_Proxy",
            "Sentiment"]].head(15).style.format({
                "Daily_PnL": "${:,.2f}", "Win_Rate": "{:.2%}",
                "Avg_Trade_Size_USD": "${:,.2f}",
                "Long_Short_Ratio": "{:.2f}", "Leverage_Proxy": "{:.2f}"}),
        width="stretch", height=400)

# ──────── TAB B.1 ────────
with tab_b1:
    st.markdown("## Performance: Fear vs Greed")
    perf = fdf.groupby("Sentiment").agg(
        Mean_PnL=("Daily_PnL", "mean"), Median_PnL=("Daily_PnL", "median"),
        Std_PnL=("Daily_PnL", "std"), Mean_WR=("Win_Rate", "mean"),
        Count=("Daily_PnL", "count"),
    ).reindex(["Fear", "Neutral", "Greed"]).dropna()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.bar(perf.reset_index(), x="Sentiment", y="Mean_PnL",
                     color="Sentiment", color_discrete_map=SENT_COLORS,
                     title="Average Daily PnL by Sentiment")
        fig.update_traces(marker_line_width=0, opacity=0.9,
                          texttemplate="$%{y:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False)
        apply_theme(fig, 420)
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(perf.reset_index(), x="Sentiment", y="Mean_WR",
                     color="Sentiment", color_discrete_map=SENT_COLORS,
                     title="Average Win Rate by Sentiment")
        fig.update_traces(marker_line_width=0, opacity=0.9,
                          texttemplate="%{y:.1%}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        apply_theme(fig, 420)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### PnL Distribution")
    fig = go.Figure()
    for s in ["Fear", "Neutral", "Greed"]:
        vals = fdf.loc[fdf["Sentiment"] == s, "Daily_PnL"].clip(-50000, 50000)
        fig.add_trace(go.Histogram(x=vals, name=s, marker_color=SENT_COLORS[s],
                                    opacity=0.55, nbinsx=80))
    fig.update_layout(barmode="overlay", xaxis_title="Daily PnL ($)", yaxis_title="Frequency")
    apply_theme(fig, 380)
    st.plotly_chart(fig, width="stretch")

    if "Drawdown" in fdf.columns:
        st.markdown("### Drawdown by Sentiment")
        dd_acct = fdf.groupby(["Account", "Sentiment"])["Drawdown"].min().reset_index()
        fig = px.box(dd_acct, x="Sentiment", y="Drawdown",
                     color="Sentiment", color_discrete_map=SENT_COLORS,
                     category_orders={"Sentiment": ["Fear", "Neutral", "Greed"]})
        fig.update_layout(showlegend=False)
        apply_theme(fig, 380)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(perf.style.format({
        "Mean_PnL": "${:,.2f}", "Median_PnL": "${:,.2f}",
        "Std_PnL": "${:,.2f}", "Mean_WR": "{:.3f}", "Count": "{:,.0f}"}),
        width="stretch")

# ──────── TAB B.2 ────────
with tab_b2:
    st.markdown("## Behavioral Changes by Sentiment")
    beh = fdf.groupby("Sentiment").agg(
        Avg_Trades=("Total_Trades", "mean"), Avg_Size=("Avg_Trade_Size_USD", "mean"),
        Avg_LS=("Long_Short_Ratio", "mean"), Avg_Lev=("Leverage_Proxy", "mean"),
    ).reindex(["Fear", "Neutral", "Greed"]).dropna().reset_index()

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.1,
                        subplot_titles=["Trade Frequency", "Avg Trade Size (USD)",
                                        "Long/Short Ratio", "Leverage Proxy"])
    for col_name, ri, ci in [("Avg_Trades",1,1),("Avg_Size",1,2),("Avg_LS",2,1),("Avg_Lev",2,2)]:
        fig.add_trace(go.Bar(
            x=beh["Sentiment"], y=beh[col_name],
            marker=dict(color=[SENT_COLORS.get(s) for s in beh["Sentiment"]],
                        line=dict(width=0)),
            opacity=0.9, showlegend=False,
            text=beh[col_name].round(1), textposition="outside",
        ), row=ri, col=ci)
    apply_theme(fig, 600)
    for ann in fig.layout.annotations:
        ann.font.color = "#c4b5fd"
        ann.font.size = 14
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Long vs Short Trades")
    ls_data = fdf.groupby("Sentiment")[["Long_Trades", "Short_Trades"]].sum().reindex(
        ["Fear", "Neutral", "Greed"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ls_data["Sentiment"], y=ls_data["Long_Trades"],
                         name="Long", marker_color=GREED_C, opacity=0.85))
    fig.add_trace(go.Bar(x=ls_data["Sentiment"], y=ls_data["Short_Trades"],
                         name="Short", marker_color=FEAR_C, opacity=0.85))
    fig.update_layout(barmode="group")
    apply_theme(fig, 400)
    st.plotly_chart(fig, width="stretch")

# ──────── TAB B.3 ────────
with tab_b3:
    st.markdown("## Trader Segmentation")
    seg_choice = st.selectbox("Select Segment Type",
                               ["Seg_Leverage", "Seg_Frequency", "Seg_Consistency"],
                               format_func=lambda x: x.replace("Seg_", ""))

    seg_perf = fdf.groupby([seg_choice, "Sentiment"])["Daily_PnL"].mean().reset_index()
    fig = px.bar(seg_perf, x="Sentiment", y="Daily_PnL", color=seg_choice,
                 barmode="group", title=f"Avg Daily PnL by {seg_choice.replace('Seg_', '')}",
                 category_orders={"Sentiment": ["Fear", "Neutral", "Greed"]},
                 color_discrete_sequence=["#a78bfa", "#60a5fa", "#34d399"])
    fig.update_traces(opacity=0.9)
    apply_theme(fig, 450)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Win Rate Heatmap")
    wr_pivot = fdf.groupby([seg_choice, "Sentiment"])["Win_Rate"].mean().unstack()
    cols_order = [c for c in ["Fear", "Neutral", "Greed"] if c in wr_pivot.columns]
    wr_pivot = wr_pivot[cols_order]
    fig = px.imshow(wr_pivot.values, x=cols_order, y=wr_pivot.index.tolist(),
                    text_auto=".3f", color_continuous_scale=["#7f1d1d", "#854d0e", "#14532d"],
                    zmin=0.3, zmax=0.95,
                    labels=dict(x="Sentiment", y=seg_choice.replace("Seg_", ""), color="Win Rate"))
    apply_theme(fig, 350)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Segment Summary")
    seg_summary = fdf.groupby(seg_choice).agg(
        Traders=("Account", "nunique"), Avg_PnL=("Daily_PnL", "mean"),
        Avg_WR=("Win_Rate", "mean"), Avg_Size=("Avg_Trade_Size_USD", "mean"),
        Total_Trades=("Total_Trades", "sum"),
    ).round(2)
    st.dataframe(seg_summary.style.format({
        "Avg_PnL": "${:,.2f}", "Avg_WR": "{:.3f}",
        "Avg_Size": "${:,.2f}", "Total_Trades": "{:,.0f}"}), width="stretch")

# ──────── TAB C ────────
with tab_c:
    st.markdown("## Actionable Strategy Rules")
    st.markdown("")

    ll_fear = fdf.loc[(fdf["Seg_Leverage"] == "Low Leverage") &
                       (fdf["Sentiment"] == "Fear"), "Daily_PnL"].mean()
    ll_greed = fdf.loc[(fdf["Seg_Leverage"] == "Low Leverage") &
                        (fdf["Sentiment"] == "Greed"), "Daily_PnL"].mean()
    inf_fear_wr = fdf.loc[(fdf["Seg_Frequency"] == "Infrequent") &
                           (fdf["Sentiment"] == "Fear"), "Win_Rate"].mean()
    inf_greed_wr = fdf.loc[(fdf["Seg_Frequency"] == "Infrequent") &
                            (fdf["Sentiment"] == "Greed"), "Win_Rate"].mean()

    pct_diff = ((ll_fear - ll_greed) / (abs(ll_greed) + 1e-9)) * 100

    st.markdown(f"""
    <div class="rule-card green">
        <h4>RULE 1 &mdash; Lean Into Fear (Low-Leverage Traders)</h4>
        <p>During <strong>Fear periods</strong>, Low-Leverage traders should <strong>increase</strong> trade frequency by 10-20%.</p>
        <ul>
            <li><strong>Segment:</strong> Low Leverage (below-median avg trade size)</li>
            <li><strong>Evidence:</strong> Avg PnL during Fear = <strong>${ll_fear:,.2f}</strong> vs Greed = <strong>${ll_greed:,.2f}</strong> ({pct_diff:+.0f}%)</li>
            <li><strong>Trigger:</strong> Fear &amp; Greed Index &lt; 40</li>
            <li><strong>Action:</strong> Increase position frequency; these traders thrive in volatile, fear-driven conditions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rule-card amber">
        <h4>RULE 2 &mdash; Sit Out Fear (Infrequent Traders)</h4>
        <p>During <strong>Fear periods</strong>, Infrequent traders should <strong>reduce</strong> activity and wait for Neutral/Greed.</p>
        <ul>
            <li><strong>Segment:</strong> Infrequent traders (below-median total trade count)</li>
            <li><strong>Evidence:</strong> Win rate drops from <strong>{inf_greed_wr:.1%}</strong> (Greed) to <strong>{inf_fear_wr:.1%}</strong> (Fear) &mdash; likely panic-trading</li>
            <li><strong>Trigger:</strong> Fear &amp; Greed Index &lt; 45</li>
            <li><strong>Action:</strong> Do not enter new positions; use limit orders only if necessary</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ──────── TAB BONUS ────────
with tab_bonus:
    st.markdown("## Predictive Model & Clustering")
    tab_model, tab_cluster = st.tabs(["Predictive Model", "Clustering"])

    with tab_model:
        st.markdown("### Next-Day PnL Direction Prediction")

        mc_path = "model_comparison.csv"
        if os.path.exists(mc_path):
            mc = pd.read_csv(mc_path)
            best_idx = mc["Balanced_Accuracy"].idxmax()
            best_model = mc.loc[best_idx, "Model"]

            st.markdown(f"""
            <div class="insight-card">
                <h4>Best Model: {best_model}</h4>
                <p>Selected by highest Balanced Accuracy across 4 candidate models.
                Trained with SMOTE oversampling and per-model F1-threshold optimization.</p>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(4, gap="medium")
            model_icons = {"Logistic Regression": "LR", "Random Forest": "RF",
                           "Gradient Boosting": "GB", "XGBoost": "XGB"}
            for i, (_, row) in enumerate(mc.iterrows()):
                cols[i].metric(
                    row["Model"],
                    f"{row['Balanced_Accuracy']:.1%}",
                    f"F1: {row['Macro_F1']:.3f} | AUC: {row['ROC_AUC']:.3f}")

            st.dataframe(mc.style.format({
                "Threshold": "{:.2f}", "Accuracy": "{:.4f}",
                "Balanced_Accuracy": "{:.4f}", "Macro_F1": "{:.4f}",
                "ROC_AUC": "{:.4f}"}).highlight_max(
                    subset=["Balanced_Accuracy", "Macro_F1", "ROC_AUC"],
                    color="rgba(139, 92, 246, 0.2)"),
                width="stretch")

        chart_path = os.path.join("charts", "Bonus_model_results.png")
        if os.path.exists(chart_path):
            st.image(chart_path, width="stretch")

        st.markdown("""
        <div class="insight-card">
            <h4>Methodology</h4>
            <p>
                <strong>Target:</strong> Next-day PnL direction (Profit/Loss)<br>
                <strong>Features (34):</strong> F&G value, momentum, rolling PnL stats, EMA, streaks, lags, sentiment dummies, day-of-week<br>
                <strong>Imbalance:</strong> SMOTE oversampling on training set<br>
                <strong>Models:</strong> Logistic Regression, Random Forest (400 trees), Gradient Boosting (300), XGBoost (300)<br>
                <strong>Threshold:</strong> Macro-F1 sweep (0.30 &ndash; 0.70) per model<br>
                <strong>Validation:</strong> Chronological 80/20 split + 5-fold stratified CV
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab_cluster:
        st.markdown("### Trader Behavioral Archetypes (K-Means)")

        for chart_name in ["Bonus_clustering.png", "Bonus_cluster_profiles.png"]:
            p = os.path.join("charts", chart_name)
            if os.path.exists(p):
                st.image(p, width="stretch")

        if not clusters.empty:
            st.markdown("### Cluster Summary")
            cl_summary = clusters.groupby("Cluster").agg(
                Traders=("Account", "count"), Avg_PnL=("avg_pnl", "mean"),
                Avg_WR=("avg_wr", "mean"), Avg_Size=("avg_size", "mean"),
                Total_Trades=("total_trades", "mean"), Avg_LS=("avg_ls", "mean"),
            ).round(2)
            st.dataframe(cl_summary.style.format({
                "Avg_PnL": "${:,.2f}", "Avg_WR": "{:.3f}",
                "Avg_Size": "${:,.2f}", "Total_Trades": "{:,.0f}",
                "Avg_LS": "{:.2f}"}), width="stretch")

            if "PCA1" in clusters.columns:
                fig = px.scatter(clusters, x="PCA1", y="PCA2",
                                 color=clusters["Cluster"].astype(str),
                                 hover_data=["Account", "avg_pnl", "avg_wr", "total_trades"],
                                 title="Trader Clusters (PCA Projection)",
                                 color_discrete_sequence=["#a78bfa", "#60a5fa", "#34d399", "#f472b6"])
                fig.update_traces(marker=dict(size=12, line=dict(width=1, color="#1e1e38")))
                apply_theme(fig, 500)
                st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding: 10px 0 20px 0;">
    <span style="color:#6b6b99; font-size:0.85rem;">
        Built with Streamlit + Plotly &bull; Data: Hyperliquid trades + Bitcoin Fear & Greed Index
    </span>
</div>
""", unsafe_allow_html=True)
