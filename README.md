<div align="center">

# 📊 Bitcoin Sentiment vs Trader Performance

### _Analyzing the relationship between Fear & Greed and Hyperliquid trader behavior_

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Prediction-006600?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

**Can market sentiment predict trader profitability?**  
This project dives deep into **211,000+ trades** across **32 Hyperliquid accounts** to find out.

[Explore the Dashboard](#-interactive-dashboard) · [View Results](#-key-findings) · [Get Started](#-quick-start)

</div>

---

## 🔍 Overview

This project performs a comprehensive, end-to-end analysis of how **Bitcoin Fear & Greed sentiment** impacts real-world trader behavior and performance on the **Hyperliquid** decentralized exchange.

The analysis covers:
- 📈 **Performance comparison** across Fear, Neutral, and Greed regimes
- 🧠 **Behavioral pattern detection** — how traders change sizing, frequency, and leverage
- 👥 **Trader segmentation** into actionable cohorts (leverage, frequency, consistency)
- 🤖 **Predictive modeling** — can we predict tomorrow's PnL direction from today's signals?
- 🔬 **K-Means clustering** — identifying distinct trader archetypes

---

## 🏗️ Project Structure

```
Project_science/
│
├── trader_sentiment_analysis.py   # Core analysis engine (Parts A/B/C + Bonus)
├── dashboard.py                   # Interactive Streamlit dashboard
│
├── historical_data.csv            # Hyperliquid trade data (input)
├── fear_greed_index.csv           # Bitcoin Fear & Greed Index (input)
│
├── charts/                        # Auto-generated visualizations
│   ├── B1_performance_by_sentiment.png
│   ├── B2_behavioral_shifts.png
│   ├── B3_segment_performance.png
│   ├── B3_winrate_heatmap.png
│   ├── Bonus_model_results.png
│   ├── Bonus_clustering.png
│   └── Bonus_cluster_profiles.png
│
├── merged_daily_analysis.csv      # Auto-generated: merged dataset
├── trader_segments.csv            # Auto-generated: segmented traders
├── trader_clusters.csv            # Auto-generated: clustered traders
├── model_comparison.csv           # Auto-generated: ML model results
│
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** installed on your system
- **pip** (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/market-sentiment-trading-model.git
cd market-sentiment-trading-model
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn streamlit plotly
```

<details>
<summary>📦 Full dependency list</summary>

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and aggregation |
| `numpy` | Numerical operations |
| `matplotlib` | Static chart generation |
| `seaborn` | Statistical visualizations & heatmaps |
| `scikit-learn` | ML models, metrics, preprocessing |
| `xgboost` | Gradient boosted trees (best model) |
| `imbalanced-learn` | SMOTE oversampling for class imbalance |
| `streamlit` | Interactive web dashboard |
| `plotly` | Interactive charts in dashboard |

</details>

### 3. Add Your Data

Place these two CSV files in the project root:
- `historical_data.csv` — Hyperliquid trade history
- `fear_greed_index.csv` — Bitcoin Fear & Greed Index

### 4. Run the Analysis

```bash
python trader_sentiment_analysis.py
```

This will:
- Clean and merge both datasets
- Calculate daily metrics (PnL, Win Rate, Leverage, etc.)
- Segment traders into 3 behavioral cohorts
- Train 4 ML models with SMOTE and threshold tuning
- Cluster traders into 4 archetypes via K-Means
- Generate 7 charts in `charts/`
- Output CSV files for all results

### 5. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Open your browser at **http://localhost:8501** to explore the interactive dashboard.

---

## 📊 Interactive Dashboard

The Streamlit dashboard provides 6 interactive tabs:

| Tab | What it shows |
|-----|---------------|
| **Part A: Data Overview** | Dataset stats, sentiment distribution pie chart, sample data table |
| **B.1 Performance** | PnL & Win Rate by sentiment, PnL distributions, drawdown analysis |
| **B.2 Behavior** | Trade frequency, position sizing, Long/Short ratios by sentiment |
| **B.3 Segments** | Segment-level performance heatmaps with selectable segment type |
| **Part C: Strategy Rules** | 2 actionable, data-backed trading rules |
| **Bonus: Model & Clusters** | 4-model comparison, ROC curves, feature importance, PCA clusters |

---

## 🔑 Key Findings

### 1. Fear Days = Higher PnL
Traders earn **+24% more PnL** on Fear days ($5,038) vs Greed days ($4,067), suggesting contrarian opportunities.

### 2. Traders Go Bigger During Fear
Position sizes increase by **43%** and trade frequency jumps **37%** during Fear — volatility attracts action.

### 3. Leverage Segments React Differently
| Segment | Fear PnL | Greed PnL | Edge |
|---------|----------|-----------|------|
| Low Leverage | $5,443 | $2,420 | **Fear** (+125%) |
| High Leverage | $4,308 | $7,047 | **Greed** (+64%) |

### 4. Predictive Model (XGBoost)
| Metric | Score |
|--------|-------|
| Balanced Accuracy | **63.1%** |
| Macro F1 | **0.624** |
| Loss Class Recall | **38%** |
| CV Balanced Accuracy | **86.3% (±2.8%)** |

---

## 🎯 Actionable Strategy Rules

> **Rule 1:** During **Fear** (F&G < 40), Low-Leverage traders should **increase** trade frequency by 10–20%. They thrive in volatile conditions.

> **Rule 2:** During **Fear**, Infrequent traders should **sit out** and wait for F&G ≥ 45. Their win rate drops significantly — likely due to panic trading.

---

## 🧪 Methodology

### Data Pipeline
```
Raw CSVs → Clean & Deduplicate → Timestamp Alignment → Daily Aggregation
    → Metric Calculation → Sentiment Merge → Segmentation → Analysis
```

### ML Pipeline
```
Feature Engineering (34 features) → SMOTE Oversampling → Train 4 Models
    → F1-Threshold Sweep → Balanced Accuracy Selection → Cross-Validation
```

### Models Compared
- **Logistic Regression** — baseline linear model
- **Random Forest** (400 trees) — ensemble with class weights
- **Gradient Boosting** (300 trees) — sequential boosting
- **XGBoost** (300 trees) — best performer, with scale_pos_weight

### Features Engineered
- **Sentiment**: F&G value, 1d/3d delta, 5d/10d moving average, Fear/Greed dummies
- **PnL Momentum**: 3d/5d MA, 5d EMA, rolling std (5d/10d), min/max, winning streak
- **Trading Activity**: Lag features (1-3d), trade/WR/LS moving averages
- **Calendar**: Day of week

---

## 📁 Datasets

| Dataset | Records | Description |
|---------|---------|-------------|
| `historical_data.csv` | 211,224 | Hyperliquid trades: account, symbol, price, size, side, PnL, fee |
| `fear_greed_index.csv` | 2,644 | Daily Bitcoin F&G: value (0-100), classification (Fear/Greed) |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ for data-driven trading insights**

_If you found this useful, give it a ⭐!_

</div>
