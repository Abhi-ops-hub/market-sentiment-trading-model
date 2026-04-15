# Executive Summary: Bitcoin Sentiment & Trader Performance Analysis

## Methodology

We analyzed **211,224 trades** from **32 Hyperliquid accounts** alongside the **Bitcoin Fear & Greed Index** (2,644 daily observations) to quantify how market sentiment affects trader behavior and profitability.

**Data Pipeline:** Both datasets were cleaned, deduplicated, and aligned on a daily level. Six key metrics were computed per account per day: Daily PnL, Win Rate, Average Trade Size, Leverage Proxy, Trade Count, and Long/Short Ratio.

**Segmentation:** Traders were grouped into three behavioral segments using median-based thresholds — High vs Low Leverage, Frequent vs Infrequent, and Consistent Winner vs Inconsistent.

**Predictive Modeling:** A next-day PnL direction classifier was built using 34 engineered features (sentiment momentum, rolling PnL statistics, EMA, winning streaks, calendar effects). Four models were trained — Logistic Regression, Random Forest, Gradient Boosting, and XGBoost — with SMOTE oversampling to address the 78/22 class imbalance, and per-model threshold optimization via macro-F1 sweep.

---

## Key Insights

1. **Fear days generate +24% higher PnL** — Average daily PnL during Fear ($5,038) significantly outperforms Greed ($4,067), indicating contrarian profit opportunities when the market is fearful.

2. **Traders increase risk exposure during Fear** — Position sizes rise 43% ($8,530 vs $5,955) and trade frequency jumps 37% (105 vs 77 trades/day) during Fear, suggesting that experienced traders actively capitalize on volatility.

3. **Leverage segments show asymmetric sensitivity** — Low-leverage traders earn +125% more PnL in Fear vs Greed ($5,443 vs $2,420), while high-leverage traders perform better in Greed (+64%). Infrequent traders see their win rate drop from 82.9% (Greed) to 75.0% (Fear), indicating panic-driven losses.

4. **XGBoost achieves 63.1% balanced accuracy** in predicting next-day PnL direction, with 86.3% cross-validated balanced accuracy. Top predictive features: PnL momentum (5d EMA), Fear & Greed value, and rolling volatility.

---

## Strategy Recommendations

| # | Rule | Segment | Trigger | Action |
|---|------|---------|---------|--------|
| 1 | **Increase activity during Fear** | Low-Leverage traders | F&G Index < 40 | Raise trade frequency by 10–20%; these traders consistently capture more PnL in volatile, fear-driven markets |
| 2 | **Reduce activity during Fear** | Infrequent traders | F&G Index < 45 | Sit out or use limit orders only; their win rate drops ~10% during Fear due to likely panic-trading behavior |

**Bottom line:** Sentiment is a actionable signal — but the optimal response depends on trader type. Skilled, low-leverage traders should lean *into* fear; infrequent traders should wait it out.
