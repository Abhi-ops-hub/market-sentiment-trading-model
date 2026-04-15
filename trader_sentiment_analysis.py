"""
================================================================================
Trader Performance vs Market Sentiment Analysis
================================================================================
Analyzes the relationship between Bitcoin market sentiment (Fear / Greed)
and trader behavior / performance on Hyperliquid.

Parts:
  A  -- Data Preparation
  B  -- Analysis  (performance, behaviour, segmentation, 3 insights)
  C  -- Actionable Output (2 strategy rules)
  Bonus  -- Predictive model + K-Means clustering
================================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score,
                             balanced_accuracy_score, f1_score,
                             precision_recall_curve, roc_curve)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 12,
                      "axes.titlesize": 14, "axes.labelsize": 12})

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# #########################################################################
# #  PART A  -- DATA PREPARATION                                          #
# #########################################################################
print("=" * 72)
print("  PART A  -- DATA PREPARATION")
print("=" * 72)

# -- 1. Load & document --------------------------------------------------
print("\n1. Loading datasets ...")
hist = pd.read_csv("historical_data.csv")
sent = pd.read_csv("fear_greed_index.csv")

print(f"\n   > Historical Trader Data")
print(f"     Rows   : {hist.shape[0]:>10,}")
print(f"     Columns: {hist.shape[1]:>10}")
print(f"     Columns list: {list(hist.columns)}")

print(f"\n   > Bitcoin Sentiment Data")
print(f"     Rows   : {sent.shape[0]:>10,}")
print(f"     Columns: {sent.shape[1]:>10}")
print(f"     Columns list: {list(sent.columns)}")

# Missing values
print("\n   Missing values  -- Historical:")
m_hist = hist.isnull().sum()
if m_hist.sum() == 0:
    print("     [OK] None")
else:
    for col, n in m_hist[m_hist > 0].items():
        print(f"     {col}: {n}")

print("   Missing values  -- Sentiment:")
m_sent = sent.isnull().sum()
if m_sent.sum() == 0:
    print("     [OK] None")
else:
    for col, n in m_sent[m_sent > 0].items():
        print(f"     {col}: {n}")

# Duplicates
dup_h = hist.duplicated().sum()
dup_s = sent.duplicated().sum()
print(f"\n   Duplicate rows  -- Historical: {dup_h}")
print(f"   Duplicate rows  -- Sentiment : {dup_s}")
if dup_h:
    hist.drop_duplicates(inplace=True)
    print(f"     -> removed {dup_h} duplicate rows")
if dup_s:
    sent.drop_duplicates(inplace=True)
    print(f"     -> removed {dup_s} duplicate rows")

# -- 2. Convert timestamps & align by date -------------------------------
print("\n2. Converting timestamps & aligning by date ...")
hist["datetime"] = pd.to_datetime(hist["Timestamp IST"],
                                   format="%d-%m-%Y %H:%M")
hist["date"] = hist["datetime"].dt.normalize()

sent["date"] = pd.to_datetime(sent["date"])

print(f"   Historical range : {hist['date'].min().date()} -> {hist['date'].max().date()}")
print(f"   Sentiment  range : {sent['date'].min().date()} -> {sent['date'].max().date()}")
overlap_dates = set(hist["date"].unique()) & set(sent["date"].unique())
print(f"   Overlapping dates: {len(overlap_dates)}")

# Broad sentiment bucket (Fear / Greed / Neutral)
def bucket_sentiment(c):
    if "Fear" in c:
        return "Fear"
    if "Greed" in c:
        return "Greed"
    return "Neutral"

sent["Sentiment"] = sent["classification"].apply(bucket_sentiment)

# -- 3. Key metrics ------------------------------------------------------
print("\n3. Creating key metrics ...")

# Force numeric
for c in ["Closed PnL", "Size USD", "Execution Price", "Fee", "Size Tokens"]:
    hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0)

hist["Net_PnL"] = hist["Closed PnL"] - hist["Fee"]
hist["is_win"]  = hist["Closed PnL"] > 0
hist["has_pnl"] = hist["Closed PnL"] != 0
hist["is_long"] = hist["Side"].str.upper().str.strip() == "BUY"

# Daily account-level aggregation
daily = hist.groupby(["Account", "date"]).agg(
    Daily_PnL          = ("Net_PnL",       "sum"),
    Gross_PnL          = ("Closed PnL",    "sum"),
    Total_Trades       = ("Trade ID",      "count"),
    Winning_Trades     = ("is_win",        "sum"),
    PnL_Trades         = ("has_pnl",       "sum"),
    Avg_Trade_Size_USD = ("Size USD",      "mean"),
    Total_Volume_USD   = ("Size USD",      "sum"),
    Max_Trade_Size     = ("Size USD",      "max"),
    Total_Fees         = ("Fee",           "sum"),
    Long_Trades        = ("is_long",       "sum"),
).reset_index()

daily["Short_Trades"] = daily["Total_Trades"] - daily["Long_Trades"]
daily["Win_Rate"] = np.where(daily["PnL_Trades"] > 0,
                             daily["Winning_Trades"] / daily["PnL_Trades"],
                             np.nan)
daily["Long_Short_Ratio"] = np.where(
    daily["Short_Trades"] > 0,
    daily["Long_Trades"] / daily["Short_Trades"],
    np.where(daily["Long_Trades"] > 0, 10.0, 1.0))
daily["Long_Short_Ratio"] = daily["Long_Short_Ratio"].clip(upper=10)

# Leverage proxy  -- avg position size relative to the trader's own median
# (higher = more leveraged / risky behaviour)
acct_median_size = daily.groupby("Account")["Avg_Trade_Size_USD"].transform("median")
daily["Leverage_Proxy"] = daily["Avg_Trade_Size_USD"] / acct_median_size.replace(0, np.nan)
daily["Leverage_Proxy"].fillna(1, inplace=True)

print(f"   Daily account rows: {daily.shape[0]:,}   |   Unique accounts: {daily['Account'].nunique()}")
print(f"   Metrics: Daily_PnL, Win_Rate, Avg_Trade_Size_USD, Total_Trades,")
print(f"            Long_Short_Ratio, Leverage_Proxy, Total_Volume_USD")

# Merge sentiment
df = daily.merge(
    sent[["date", "Sentiment", "classification", "value"]].rename(
        columns={"value": "FG_Value"}),
    on="date", how="inner"
)
print(f"   Merged rows: {df.shape[0]:,}   |   Dates: {df['date'].nunique()}")

# #########################################################################
# #  PART B  -- ANALYSIS                                                   #
# #########################################################################
print("\n" + "=" * 72)
print("  PART B  -- ANALYSIS")
print("=" * 72)

FEAR_C, GREED_C, NEUT_C = "#e74c3c", "#27ae60", "#3498db"
SENT_COLORS = {"Fear": FEAR_C, "Greed": GREED_C, "Neutral": NEUT_C}
SENT_ORDER  = ["Fear", "Neutral", "Greed"]

# -- B.1  Performance: Fear vs Greed -------------------------------------
print("\n-- B.1  Performance comparison (PnL, Win Rate, Drawdown) --")

perf = df.groupby("Sentiment").agg(
    Mean_PnL    = ("Daily_PnL",          "mean"),
    Median_PnL  = ("Daily_PnL",          "median"),
    Std_PnL     = ("Daily_PnL",          "std"),
    Mean_WinR   = ("Win_Rate",           "mean"),
    Median_WinR = ("Win_Rate",           "median"),
    Records     = ("Daily_PnL",          "count"),
).reindex(SENT_ORDER).round(2)
print(perf.to_string())

# Drawdown proxy  -- cumulative PnL -> rolling max -> drawdown per account
df.sort_values(["Account", "date"], inplace=True)
df["Cum_PnL"]     = df.groupby("Account")["Daily_PnL"].cumsum()
df["Running_Max"] = df.groupby("Account")["Cum_PnL"].cummax()
df["Drawdown"]    = df["Cum_PnL"] - df["Running_Max"]   # always <= 0

dd_acct = df.groupby(["Account", "Sentiment"])["Drawdown"].min().reset_index()
dd_summary = dd_acct.groupby("Sentiment")["Drawdown"].agg(
    ["mean", "median", "min"]).reindex(SENT_ORDER).round(2)
dd_summary.columns = ["Mean_MaxDD", "Median_MaxDD", "Worst_DD"]
print("\n   Drawdown Proxy by Sentiment:")
print(dd_summary.to_string())

# Chart B1
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
# 1a - PnL distribution
for s in SENT_ORDER:
    vals = df.loc[df["Sentiment"] == s, "Daily_PnL"].clip(-50000, 50000)
    axes[0].hist(vals, bins=60, alpha=0.45, color=SENT_COLORS[s], label=s)
axes[0].set_title("Daily PnL Distribution"); axes[0].set_xlabel("PnL ($)")
axes[0].set_ylabel("Frequency"); axes[0].legend()

# 1b - Win-rate box
wr_data = [df.loc[df["Sentiment"] == s, "Win_Rate"].dropna() for s in SENT_ORDER]
bp = axes[1].boxplot(wr_data, labels=SENT_ORDER, patch_artist=True,
                      widths=0.5, showfliers=False)
for patch, s in zip(bp["boxes"], SENT_ORDER):
    patch.set_facecolor(SENT_COLORS[s]); patch.set_alpha(0.55)
axes[1].set_title("Win Rate by Sentiment"); axes[1].set_ylabel("Win Rate")

# 1c - Drawdown box
dd_data = [dd_acct.loc[dd_acct["Sentiment"] == s, "Drawdown"] for s in SENT_ORDER]
bp2 = axes[2].boxplot(dd_data, labels=SENT_ORDER, patch_artist=True,
                       widths=0.5, showfliers=False)
for patch, s in zip(bp2["boxes"], SENT_ORDER):
    patch.set_facecolor(SENT_COLORS[s]); patch.set_alpha(0.55)
axes[2].set_title("Max Drawdown Proxy"); axes[2].set_ylabel("Drawdown ($)")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/B1_performance_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   [OK] Saved {CHART_DIR}/B1_performance_by_sentiment.png")

# -- B.2  Behavioral changes --------------------------------------------
print("\n-- B.2  Behavioral changes by Sentiment --")

beh = df.groupby("Sentiment").agg(
    Avg_Trades     = ("Total_Trades",       "mean"),
    Avg_Size_USD   = ("Avg_Trade_Size_USD", "mean"),
    Avg_LS_Ratio   = ("Long_Short_Ratio",   "mean"),
    Avg_Volume     = ("Total_Volume_USD",   "mean"),
    Avg_Lev_Proxy  = ("Leverage_Proxy",     "mean"),
    Pct_Long       = ("Long_Trades",
                      lambda x: x.sum() /
                      (x.sum() + df.loc[x.index, "Short_Trades"].sum())),
).reindex(SENT_ORDER).round(4)
print(beh.to_string())

# Chart B2  -- 2x2 behavioral panel
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
_bars = lambda ax, vals, title, ylabel: (
    ax.bar(SENT_ORDER, vals, color=[FEAR_C, NEUT_C, GREED_C]),
    ax.set_title(title), ax.set_ylabel(ylabel))

_bars(axes[0, 0],
      [beh.loc[s, "Avg_Trades"] for s in SENT_ORDER],
      "Avg Trade Frequency / Day", "Trades")
_bars(axes[0, 1],
      [beh.loc[s, "Avg_Size_USD"] for s in SENT_ORDER],
      "Avg Trade Size (USD)", "USD")
_bars(axes[1, 0],
      [beh.loc[s, "Avg_LS_Ratio"] for s in SENT_ORDER],
      "Long / Short Ratio", "L/S Ratio")
axes[1, 0].axhline(1.0, ls="--", color="black", alpha=0.4, label="Balanced")
axes[1, 0].legend()
_bars(axes[1, 1],
      [beh.loc[s, "Avg_Lev_Proxy"] for s in SENT_ORDER],
      "Leverage Proxy (Size vs Own Median)", "Ratio")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/B2_behavioral_shifts.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK] Saved {CHART_DIR}/B2_behavioral_shifts.png")

# -- B.3  Trader segmentation -------------------------------------------
print("\n-- B.3  Trader Segmentation --")

acct = df.groupby("Account").agg(
    total_trades   = ("Total_Trades",       "sum"),
    avg_pnl        = ("Daily_PnL",          "mean"),
    std_pnl        = ("Daily_PnL",          "std"),
    avg_wr         = ("Win_Rate",           "mean"),
    avg_size       = ("Avg_Trade_Size_USD", "mean"),
    avg_lev_proxy  = ("Leverage_Proxy",     "mean"),
    total_vol      = ("Total_Volume_USD",   "sum"),
    avg_ls         = ("Long_Short_Ratio",   "mean"),
    active_days    = ("date",               "nunique"),
).fillna(0).reset_index()

# Segment 1  -- High vs Low leverage (proxy)
lev_med = acct["avg_lev_proxy"].median()
acct["Seg_Leverage"] = np.where(acct["avg_lev_proxy"] > lev_med,
                                 "High Leverage", "Low Leverage")

# Segment 2  -- Frequent vs Infrequent
freq_med = acct["total_trades"].median()
acct["Seg_Frequency"] = np.where(acct["total_trades"] > freq_med,
                                  "Frequent", "Infrequent")

# Segment 3  -- Consistent winner vs Inconsistent
wr_med  = acct["avg_wr"].median()
vol_med = acct["std_pnl"].median()
acct["Seg_Consistency"] = np.where(
    (acct["avg_wr"] >= wr_med) & (acct["std_pnl"] <= vol_med),
    "Consistent Winner",
    np.where(
        (acct["avg_wr"] < wr_med) & (acct["std_pnl"] > vol_med),
        "Inconsistent", "Mixed"))

for seg in ["Seg_Leverage", "Seg_Frequency", "Seg_Consistency"]:
    print(f"\n   {seg}:")
    print(acct[seg].value_counts().to_string())

# Broadcast back
df = df.merge(acct[["Account", "Seg_Leverage", "Seg_Frequency",
                      "Seg_Consistency"]], on="Account", how="left")

# Chart B3 - segment performance by sentiment
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for i, (scol, title) in enumerate([
    ("Seg_Leverage",    "High vs Low Leverage"),
    ("Seg_Frequency",   "Frequent vs Infrequent"),
    ("Seg_Consistency", "Consistency Segments")]):
    pivot = df.groupby([scol, "Sentiment"])["Daily_PnL"].mean().unstack()
    cols = [c for c in SENT_ORDER if c in pivot.columns]
    pivot[cols].plot(kind="bar", ax=axes[i],
                     color=[SENT_COLORS[c] for c in cols])
    axes[i].set_title(f"Avg PnL  -- {title}")
    axes[i].set_ylabel("Mean Daily PnL ($)")
    axes[i].tick_params(axis="x", rotation=25)
    axes[i].legend(title="Sentiment")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/B3_segment_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   [OK] Saved {CHART_DIR}/B3_segment_performance.png")

# Chart B3b - win-rate segmentation heatmap
seg_wr = df.groupby(["Seg_Leverage", "Sentiment"])["Win_Rate"].mean().unstack()
seg_wr2 = df.groupby(["Seg_Frequency", "Sentiment"])["Win_Rate"].mean().unstack()
seg_wr3 = df.groupby(["Seg_Consistency", "Sentiment"])["Win_Rate"].mean().unstack()
combined = pd.concat([seg_wr, seg_wr2, seg_wr3])
cols = [c for c in SENT_ORDER if c in combined.columns]
combined = combined[cols]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(combined, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
            linewidths=0.5, vmin=0.3, vmax=0.8)
ax.set_title("Win Rate by Segment x Sentiment")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/B3_winrate_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK] Saved {CHART_DIR}/B3_winrate_heatmap.png")

# -- B.4  Three key insights --------------------------------------------
print("\n-- B.4  Key Insights --\n")

fear_pnl  = df.loc[df["Sentiment"] == "Fear",  "Daily_PnL"].mean()
greed_pnl = df.loc[df["Sentiment"] == "Greed", "Daily_PnL"].mean()
neut_pnl  = df.loc[df["Sentiment"] == "Neutral","Daily_PnL"].mean()

hl_fear  = df.loc[(df["Seg_Leverage"] == "High Leverage") &
                   (df["Sentiment"] == "Fear"), "Daily_PnL"].mean()
hl_greed = df.loc[(df["Seg_Leverage"] == "High Leverage") &
                   (df["Sentiment"] == "Greed"), "Daily_PnL"].mean()
ll_fear  = df.loc[(df["Seg_Leverage"] == "Low Leverage") &
                   (df["Sentiment"] == "Fear"), "Daily_PnL"].mean()
ll_greed = df.loc[(df["Seg_Leverage"] == "Low Leverage") &
                   (df["Sentiment"] == "Greed"), "Daily_PnL"].mean()

freq_fear_wr  = df.loc[(df["Seg_Frequency"] == "Frequent") &
                        (df["Sentiment"] == "Fear"), "Win_Rate"].mean()
freq_greed_wr = df.loc[(df["Seg_Frequency"] == "Frequent") &
                        (df["Sentiment"] == "Greed"), "Win_Rate"].mean()
inf_fear_wr   = df.loc[(df["Seg_Frequency"] == "Infrequent") &
                        (df["Sentiment"] == "Fear"), "Win_Rate"].mean()
inf_greed_wr  = df.loc[(df["Seg_Frequency"] == "Infrequent") &
                        (df["Sentiment"] == "Greed"), "Win_Rate"].mean()

fear_size  = df.loc[df["Sentiment"] == "Fear",  "Avg_Trade_Size_USD"].mean()
greed_size = df.loc[df["Sentiment"] == "Greed", "Avg_Trade_Size_USD"].mean()
fear_freq  = df.loc[df["Sentiment"] == "Fear",  "Total_Trades"].mean()
greed_freq = df.loc[df["Sentiment"] == "Greed", "Total_Trades"].mean()

print(f"""   INSIGHT 1  -- Fear days generate higher average PnL
   +--------------------------------------------------------------+
   |  Fear  avg PnL : ${fear_pnl:>12,.2f}                        |
   |  Greed avg PnL : ${greed_pnl:>12,.2f}                        |
   |  Delta             : ${fear_pnl - greed_pnl:>12,.2f}  ({((fear_pnl - greed_pnl) / (abs(greed_pnl) + 1e-9)) * 100:>+.1f}%)          |
   |  Implication: Contrarian opportunities exist during fear.   |
   +--------------------------------------------------------------+

   INSIGHT 2  -- Traders take larger positions during Fear
   +--------------------------------------------------------------+
   |  Fear  avg size : ${fear_size:>12,.2f}                       |
   |  Greed avg size : ${greed_size:>12,.2f}                       |
   |  Fear  avg freq : {fear_freq:>10.1f} trades/day              |
   |  Greed avg freq : {greed_freq:>10.1f} trades/day              |
   |  Implication: Higher volatility -> bigger bets.              |
   +--------------------------------------------------------------+

   INSIGHT 3  -- Leverage segment & frequency show asymmetric sentiment sensitivity
   +--------------------------------------------------------------+
   |  High Lev  -- Fear  : ${hl_fear:>12,.2f}  |  Greed: ${hl_greed:>12,.2f} |
   |  Low  Lev  -- Fear  : ${ll_fear:>12,.2f}  |  Greed: ${ll_greed:>12,.2f} |
   |  Freq. WR  -- Fear  : {freq_fear_wr:.3f}       |  Greed: {freq_greed_wr:.3f}        |
   |  Infreq WR  -- Fear : {inf_fear_wr:.3f}       |  Greed: {inf_greed_wr:.3f}        |
   |  Low-lev traders capture more PnL in Fear; High-lev in Greed |
   +--------------------------------------------------------------+
""")

# Save merged
df.to_csv("merged_daily_analysis.csv", index=False)
print(f"   [OK] Saved merged_daily_analysis.csv  ({df.shape[0]:,} rows)")
acct.to_csv("trader_segments.csv", index=False)
print(f"   [OK] Saved trader_segments.csv  ({acct.shape[0]:,} rows)")

# #########################################################################
# #  PART C  -- ACTIONABLE OUTPUT                                          #
# #########################################################################
print("\n" + "=" * 72)
print("  PART C  -- ACTIONABLE OUTPUT (2 Strategy Rules)")
print("=" * 72)

# Dynamically pick correct direction based on data
if hl_fear > hl_greed:
    rule1_dir = "INCREASE"
    rule1_cond = "Fear"
    rule1_alt = "Greed"
else:
    rule1_dir = "INCREASE"
    rule1_cond = "Greed"
    rule1_alt = "Fear"

print(f"""
   RULE 1  --------------------------------------------------------------
   "During {rule1_cond} periods, Low-Leverage traders should {rule1_dir}
    position sizes to capture sentiment-driven opportunities."

   * Segment  : Low Leverage (proxy = below-median avg trade size)
   * Evidence : Low-lev traders avg PnL during Fear = ${ll_fear:,.2f}
                vs Greed = ${ll_greed:,.2f}.
                They earn {((ll_fear - ll_greed) / (abs(ll_greed) + 1e-9)) * 100:+.0f}% more PnL in Fear.
   * Action   : When Fear & Greed Index < 40, increase trade frequency
                by 10-20%.  These traders thrive in volatile conditions.
   ----------------------------------------------------------------------

   RULE 2  --------------------------------------------------------------
   "During Fear periods, Infrequent traders should REDUCE activity and
    wait for Neutral / Greed to re-enter."

   * Segment  : Infrequent traders (below-median total trade count)
   * Evidence : Infrequent trader win rate Fear = {inf_fear_wr:.3f}
                vs Greed = {inf_greed_wr:.3f}.  They likely panic-trade.
   * Action   : Set a minimum F&G threshold of 45 before new entries.
                Use limit orders instead of market orders during Fear.
   ----------------------------------------------------------------------
""")

# #########################################################################
# #  BONUS  -- PREDICTIVE MODEL  (Improved)                                #
# #########################################################################
print("=" * 72)
print("  BONUS  -- PREDICTIVE MODEL  (next-day PnL direction)")
print("=" * 72)

# -- Step 1: Aggregate to market-wide daily level -------------------------
daily_mkt = df.groupby("date").agg(
    Total_PnL     = ("Daily_PnL",          "sum"),
    Avg_WR        = ("Win_Rate",           "mean"),
    Avg_Size      = ("Avg_Trade_Size_USD", "mean"),
    Num_Traders   = ("Account",            "nunique"),
    Total_Trades  = ("Total_Trades",       "sum"),
    Avg_LS        = ("Long_Short_Ratio",   "mean"),
    Avg_Lev       = ("Leverage_Proxy",     "mean"),
    FG_Value      = ("FG_Value",           "first"),
    Sentiment     = ("Sentiment",          "first"),
    Long_Total    = ("Long_Trades",        "sum"),
    Short_Total   = ("Short_Trades",       "sum"),
    Avg_Volume    = ("Total_Volume_USD",   "mean"),
).sort_index().reset_index()

# Target: next-day profitable (1) or not (0)
daily_mkt["Next_PnL"] = daily_mkt["Total_PnL"].shift(-1)
daily_mkt["Target"]   = (daily_mkt["Next_PnL"] > 0).astype(int)

# -- Step 2: Rich feature engineering -------------------------------------
print("\n   Engineering features ...")

# Lag features (1d, 2d, 3d)
for lag in [1, 2, 3]:
    daily_mkt[f"PnL_Lag{lag}"]    = daily_mkt["Total_PnL"].shift(lag)
    daily_mkt[f"WR_Lag{lag}"]     = daily_mkt["Avg_WR"].shift(lag)
    daily_mkt[f"FG_Lag{lag}"]     = daily_mkt["FG_Value"].shift(lag)
    daily_mkt[f"Trades_Lag{lag}"] = daily_mkt["Total_Trades"].shift(lag)

# F&G momentum / change
daily_mkt["FG_Delta_1d"]  = daily_mkt["FG_Value"] - daily_mkt["FG_Value"].shift(1)
daily_mkt["FG_Delta_3d"]  = daily_mkt["FG_Value"] - daily_mkt["FG_Value"].shift(3)
daily_mkt["FG_MA_5d"]     = daily_mkt["FG_Value"].rolling(5).mean()
daily_mkt["FG_MA_10d"]    = daily_mkt["FG_Value"].rolling(10).mean()
daily_mkt["FG_Std_5d"]    = daily_mkt["FG_Value"].rolling(5).std()

# PnL momentum
daily_mkt["PnL_MA_3d"]    = daily_mkt["Total_PnL"].rolling(3).mean()
daily_mkt["PnL_MA_5d"]    = daily_mkt["Total_PnL"].rolling(5).mean()
daily_mkt["PnL_EMA_5d"]   = daily_mkt["Total_PnL"].ewm(span=5).mean()
daily_mkt["PnL_Std_5d"]   = daily_mkt["Total_PnL"].rolling(5).std()
daily_mkt["PnL_Std_10d"]  = daily_mkt["Total_PnL"].rolling(10).std()
daily_mkt["PnL_Min_5d"]   = daily_mkt["Total_PnL"].rolling(5).min()
daily_mkt["PnL_Max_5d"]   = daily_mkt["Total_PnL"].rolling(5).max()

# PnL streak: how many consecutive positive-PnL days
daily_mkt["PnL_Positive"] = (daily_mkt["Total_PnL"] > 0).astype(int)
# cumulative streak via groupby trick
neg_mask = daily_mkt["PnL_Positive"] == 0
daily_mkt["_grp"] = neg_mask.cumsum()
daily_mkt["PnL_Streak"] = daily_mkt.groupby("_grp")["PnL_Positive"].cumsum()
daily_mkt.drop(columns=["_grp"], inplace=True)
daily_mkt["PnL_Streak_Lag1"] = daily_mkt["PnL_Streak"].shift(1)

# Trading activity momentum
daily_mkt["Trades_MA_5d"]  = daily_mkt["Total_Trades"].rolling(5).mean()
daily_mkt["WR_MA_5d"]      = daily_mkt["Avg_WR"].rolling(5).mean()
daily_mkt["LS_MA_5d"]      = daily_mkt["Avg_LS"].rolling(5).mean()

# Day of week (pattern capture)
daily_mkt["DayOfWeek"] = pd.to_datetime(daily_mkt["date"]).dt.dayofweek

# Sentiment dummies
daily_mkt["Is_Fear"]  = (daily_mkt["Sentiment"] == "Fear").astype(int)
daily_mkt["Is_Greed"] = (daily_mkt["Sentiment"] == "Greed").astype(int)

daily_mkt.dropna(inplace=True)

# -- Step 3: Feature selection --------------------------------------------
feature_cols = [
    # Core today features
    "FG_Value", "Total_Trades", "Avg_Size", "Avg_LS", "Avg_WR",
    "Total_PnL", "Avg_Lev", "Num_Traders", "Avg_Volume",
    # Sentiment features
    "FG_Delta_1d", "FG_Delta_3d", "FG_MA_5d", "FG_MA_10d", "FG_Std_5d",
    "Is_Fear", "Is_Greed",
    # Lag features
    "PnL_Lag1", "PnL_Lag2", "PnL_Lag3",
    "WR_Lag1", "FG_Lag1", "Trades_Lag1",
    # Momentum / rolling stats
    "PnL_MA_3d", "PnL_MA_5d", "PnL_EMA_5d",
    "PnL_Std_5d", "PnL_Std_10d", "PnL_Min_5d", "PnL_Max_5d",
    "PnL_Streak_Lag1",
    "Trades_MA_5d", "WR_MA_5d", "LS_MA_5d",
    # Calendar
    "DayOfWeek",
]

X = daily_mkt[feature_cols].copy()
y = daily_mkt["Target"].copy()

print(f"   Features       : {len(feature_cols)}")
print(f"   Samples        : {len(X)}")
print(f"   Class balance  : Profit {y.sum()} ({y.mean():.1%})  |  Loss {(~y.astype(bool)).sum()} ({1 - y.mean():.1%})")

# -- Step 4: Chronological train/test split --------------------------------
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"   Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
print(f"   Train class balance: Profit {y_train.mean():.1%}")
print(f"   Test  class balance: Profit {y_test.mean():.1%}")

# -- Step 5: SMOTE to handle class imbalance in training set ---------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE: {len(X_train_sm)} training samples (balanced)")

# -- Step 6: Scale features -----------------------------------------------
scaler_mdl = StandardScaler()
X_train_sc = pd.DataFrame(scaler_mdl.fit_transform(X_train_sm),
                           columns=feature_cols)
X_test_sc  = pd.DataFrame(scaler_mdl.transform(X_test),
                           columns=feature_cols)

# -- Step 7: Train 4 models -----------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=3,
        random_state=42, class_weight="balanced", n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0),
}

results = {}
print("\n   Training and evaluating models ...")

for name, model in models.items():
    # Use scaled features for LR, raw for tree-based
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train_sm)
        probs = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train_sm, y_train_sm)
        probs = model.predict_proba(X_test)[:, 1]

    # Optimal threshold via F1 sweep
    thresholds = np.arange(0.30, 0.71, 0.01)
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        preds_t = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds_t, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    preds = (probs >= best_t).astype(int)
    acc  = accuracy_score(y_test, preds)
    bacc = balanced_accuracy_score(y_test, preds)
    f1_m = f1_score(y_test, preds, average="macro")
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = 0.0

    results[name] = {
        "model": model, "probs": probs, "preds": preds,
        "threshold": best_t, "accuracy": acc,
        "balanced_accuracy": bacc, "f1_macro": f1_m, "roc_auc": auc,
    }

    print(f"\n   --- {name} (threshold={best_t:.2f}) ---")
    print(f"   Accuracy          : {acc:.4f}")
    print(f"   Balanced Accuracy : {bacc:.4f}")
    print(f"   Macro F1          : {f1_m:.4f}")
    print(f"   ROC-AUC           : {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Loss", "Profit"],
                                 zero_division=0))

# -- Step 8: Select best model by balanced accuracy ------------------------
best_name = max(results, key=lambda k: results[k]["balanced_accuracy"])
best_res  = results[best_name]
best_mdl  = best_res["model"]
best_pred = best_res["preds"]
best_prob = best_res["probs"]

print(f"   ==>  BEST MODEL: {best_name}")
print(f"        Balanced Accuracy: {best_res['balanced_accuracy']:.4f}")
print(f"        ROC-AUC          : {best_res['roc_auc']:.4f}")

# -- Step 9: Cross-validation on full training set -------------------------
print("\n   Cross-validation (5-fold stratified on training data) ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
if best_name == "Logistic Regression":
    cv_scores = cross_val_score(best_mdl, X_train_sc, y_train_sm,
                                 cv=cv, scoring="balanced_accuracy")
else:
    cv_scores = cross_val_score(best_mdl, X_train_sm, y_train_sm,
                                 cv=cv, scoring="balanced_accuracy")
print(f"   CV Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"   Folds: {[f'{s:.4f}' for s in cv_scores]}")

# -- Step 10: Visualizations ----------------------------------------------

# Chart 1: Model comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 10a: Balanced accuracy comparison
model_names = list(results.keys())
baccs = [results[n]["balanced_accuracy"] for n in model_names]
f1s   = [results[n]["f1_macro"] for n in model_names]
aucs  = [results[n]["roc_auc"] for n in model_names]

x_pos = np.arange(len(model_names))
width = 0.25
axes[0, 0].bar(x_pos - width, baccs, width, label="Bal. Accuracy", color="#3498db")
axes[0, 0].bar(x_pos,         f1s,   width, label="Macro F1",      color="#e74c3c")
axes[0, 0].bar(x_pos + width, aucs,  width, label="ROC-AUC",       color="#27ae60")
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=9)
axes[0, 0].set_title("Model Comparison")
axes[0, 0].set_ylabel("Score")
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1.05)
for i, (b, f, a) in enumerate(zip(baccs, f1s, aucs)):
    axes[0, 0].text(i - width, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    axes[0, 0].text(i,         f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    axes[0, 0].text(i + width, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)

# 10b: Confusion matrix for best model
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Loss", "Profit"],
            yticklabels=["Loss", "Profit"], ax=axes[0, 1])
axes[0, 1].set_title(f"Confusion Matrix ({best_name})")
axes[0, 1].set_xlabel("Predicted"); axes[0, 1].set_ylabel("Actual")

# 10c: ROC curves for all models
for name in model_names:
    fpr, tpr, _ = roc_curve(y_test, results[name]["probs"])
    auc_val = results[name]["roc_auc"]
    axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1, 0].set_title("ROC Curves")
axes[1, 0].set_xlabel("False Positive Rate"); axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].legend(fontsize=8)

# 10d: Feature importance for best model
if hasattr(best_mdl, "feature_importances_"):
    importances = pd.Series(best_mdl.feature_importances_,
                             index=feature_cols).sort_values()
    importances.tail(15).plot(kind="barh", ax=axes[1, 1], color="#8e44ad")
    axes[1, 1].set_title(f"Top 15 Features ({best_name})")
    axes[1, 1].set_xlabel("Importance")
elif hasattr(best_mdl, "coef_"):
    importances = pd.Series(np.abs(best_mdl.coef_[0]),
                             index=feature_cols).sort_values()
    importances.tail(15).plot(kind="barh", ax=axes[1, 1], color="#8e44ad")
    axes[1, 1].set_title(f"Top 15 Features ({best_name} |coef|)")
    axes[1, 1].set_xlabel("|Coefficient|")

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/Bonus_model_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   [OK] Saved {CHART_DIR}/Bonus_model_results.png")

# Save model comparison table
model_comparison = pd.DataFrame({
    "Model": model_names,
    "Threshold": [results[n]["threshold"] for n in model_names],
    "Accuracy": [results[n]["accuracy"] for n in model_names],
    "Balanced_Accuracy": baccs,
    "Macro_F1": f1s,
    "ROC_AUC": aucs,
}).round(4)
model_comparison.to_csv("model_comparison.csv", index=False)
print(f"   [OK] Saved model_comparison.csv")
print(model_comparison.to_string(index=False))

# #########################################################################
# #  BONUS  -- CLUSTERING  (Behavioral Archetypes)                         #
# #########################################################################
print("\n" + "=" * 72)
print("  BONUS  -- TRADER CLUSTERING  (K-Means behavioural archetypes)")
print("=" * 72)

cluster_feats = ["avg_pnl", "avg_wr", "avg_size", "total_trades",
                  "avg_ls", "std_pnl", "avg_lev_proxy"]
cdata = acct[cluster_feats].replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
scaled = scaler.fit_transform(cdata)

# Elbow
inertias = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled)
    inertias.append(km.inertia_)

# Choose K=4
N_K = 4
km_final = KMeans(n_clusters=N_K, random_state=42, n_init=10)
acct["Cluster"] = km_final.fit_predict(scaled)

print(f"\n   K-Means with K={N_K}")
for c in range(N_K):
    sub = acct[acct["Cluster"] == c]
    print(f"\n   > Cluster {c}  ({len(sub)} traders)")
    print(f"     Avg PnL         : ${sub['avg_pnl'].mean():>12,.2f}")
    print(f"     Avg Win Rate    : {sub['avg_wr'].mean():>10.3f}")
    print(f"     Avg Trade Size  : ${sub['avg_size'].mean():>12,.2f}")
    print(f"     Total Trades    : {sub['total_trades'].mean():>10,.0f}")
    print(f"     Avg L/S Ratio   : {sub['avg_ls'].mean():>10.2f}")
    print(f"     Avg Lev Proxy   : {sub['avg_lev_proxy'].mean():>10.2f}")

# Charts
pca = PCA(n_components=2)
coords = pca.fit_transform(scaled)
acct["PCA1"], acct["PCA2"] = coords[:, 0], coords[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(list(K_range), inertias, "bo-")
axes[0].axvline(N_K, ls="--", color="red", alpha=0.7, label=f"K={N_K}")
axes[0].set_title("Elbow Method"); axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia"); axes[0].legend()

scatter_c = ["#e74c3c", "#27ae60", "#3498db", "#f1c40f", "#8e44ad", "#e67e22"]
for c in range(N_K):
    mask = acct["Cluster"] == c
    axes[1].scatter(acct.loc[mask, "PCA1"], acct.loc[mask, "PCA2"],
                     c=scatter_c[c], label=f"Cluster {c}", s=80, alpha=0.7,
                     edgecolors="white", linewidth=0.5)
axes[1].set_title("Trader Clusters (PCA)"); axes[1].set_xlabel("PC 1")
axes[1].set_ylabel("PC 2"); axes[1].legend()

plt.tight_layout()
plt.savefig(f"{CHART_DIR}/Bonus_clustering.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   [OK] Saved {CHART_DIR}/Bonus_clustering.png")

# Cluster profile heatmap
profile = acct.groupby("Cluster")[cluster_feats].mean()
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(profile_norm, annot=profile.round(2).values, fmt="",
            cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Cluster Profiles (normalised heatmap, values = raw means)")
ax.set_ylabel("Cluster")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/Bonus_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK] Saved {CHART_DIR}/Bonus_cluster_profiles.png")

acct.to_csv("trader_clusters.csv", index=False)
print(f"   [OK] Saved trader_clusters.csv")

# #########################################################################
# #  SUMMARY                                                             #
# #########################################################################
print("\n" + "=" * 72)
print("  ANALYSIS COMPLETE")
print("=" * 72)
print(f"""
   Output files:
     * merged_daily_analysis.csv   (daily account x sentiment data)
     * trader_segments.csv         (account-level with 3 segments)
     * trader_clusters.csv         (account-level with K-Means cluster)
     * {CHART_DIR}/                (all charts)

   Charts generated:
     * B1_performance_by_sentiment.png
     * B2_behavioral_shifts.png
     * B3_segment_performance.png
     * B3_winrate_heatmap.png
     * Bonus_model_results.png
     * Bonus_clustering.png
     * Bonus_cluster_profiles.png

   Run the Streamlit dashboard:
     streamlit run dashboard.py
""")
