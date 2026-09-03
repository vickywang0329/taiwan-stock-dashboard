# Taiwan Stock Swing Trading Decision System

A full-stack project spanning data ingestion, database design, technical indicator computation, a multi-dimensional decision engine, and an interactive dashboard — built to help identify Taiwan-listed stocks that pass a fundamentals health check **and** show strong technical/institutional momentum for swing trading.

**Live Demo**: [https://taiwan-stock-dashboard-jumf.streamlit.app/](https://taiwan-stock-dashboard-jumf.streamlit.app/)

---

## Motivation

Most retail stock-screening tools fall into one of two camps: pure technical screens (prone to chasing overextended moves, blind to fundamental risk) or pure fundamental analysis (no sense of timing). This project separates "fundamental health" and "technical/institutional timing" into two independent judgment dimensions, so a strong technical signal isn't diluted by an average fundamental score (or vice versa). It also adjusts its valuation methodology for industries where P/E ratios are known to be unreliable — notably cyclical sectors like memory chips and shipping, where earnings swing dramatically with commodity cycles.

---

## Core Features

### 1. Three-Tier Decision Signal

For every stock in the watchlist, the system evaluates two independent dimensions — a fundamentals gate check and a technical/institutional score — and outputs one of three signals:

| Signal | Condition |
|---|---|
| **Buy (BUY_NOW)** | Fundamentals pass the gate check, and Stock Score ≥ 85 |
| **Watch (WATCH)** | Fundamentals pass the gate check, but Stock Score < 85 |
| **Avoid (AVOID)** | Fundamentals fail the gate check (further split into "fundamental red flag" / "fails both") |

### 2. Fundamentals Gate (a pass/fail checklist, not a weighted score)

- **Valuation check**: whether P/E (non-cyclical stocks) or P/B (cyclical stocks) is significantly above the industry benchmark
- **EPS growth**: whether estimated full-year EPS beats last year's
- **Gross margin trend**: whether gross margin has declined materially versus the same period last year (measured as a *relative* decline percentage, not raw percentage points, to avoid unfairly penalizing high-margin industries)

Cyclical stocks (shipping, memory semiconductors, plastics/petrochemicals, steel) have earnings driven primarily by commodity cycles rather than company-specific execution, so their valuation check uses **price-to-book (P/B)** instead of P/E, and they're exempt from the margin-decline check.

### 3. Technical/Institutional Score (Stock Score, 5-factor weighted, out of 100)

| Factor | Weight |
|---|---|
| Trend (moving average alignment) | 25% |
| Momentum (RSI, MACD) | 20% |
| Relative strength vs. benchmark | 20% |
| Institutional net buying | 20% |
| Sector fund flow | 15% |

Most sub-scores use continuous sigmoid-based scoring rather than discrete thresholds, avoiding "cliff-edge" effects where a stock narrowly missing a cutoff gets zeroed out entirely.

### 4. EPS Estimation Engine

Taiwan-listed companies disclose standalone quarterly EPS, not cumulative figures. The system prioritizes a seasonally-adjusted extrapolation method (this year's H1 EPS × last year's full-year/H1 growth ratio) to estimate full-year EPS, with multiple layers of guardrails for cyclical stocks and abnormal year-ago base periods. When extrapolation isn't reliable, it falls back to trailing-twelve-months (TTM) actuals.

### 5. Sector Fund Flow Heatmap

Computes a fund-rotation score per industry from institutional net buying and price momentum, visualized as a treemap.

### 6. Bilingual Interface

Full Traditional Chinese / English toggle across the entire app.

---

## Project Structure

```
dashboard_project/
├── Swing_Trading_Decision_System.py   # Main entry point: the decision dashboard
├── watchlist.py                        # Single source of truth for the stock universe
├── i18n.py                             # Shared language-switching module
├── sector_flow.py                      # Sector fund-rotation scoring (shared across pages)
│
├── decision_engine/                    # Core decision engine
│   ├── __init__.py
│   ├── db.py                           # Database connection & queries
│   ├── pipeline.py                     # Orchestrates the full data flow into final decisions
│   ├── engine.py                       # Three-tier signal logic
│   ├── scoring.py                      # Technical/institutional score computation
│   ├── valuation.py                    # Valuation checks, EPS estimation, P/B computation
│   ├── zones.py                        # Breakout price computation
│   ├── risk_reward.py                  # Stop-loss / target price helpers
│   └── pattern.py                      # Technical pattern classification (breakout/pullback) —
│                                        # a development-history artifact, no longer called by
│                                        # the live decision flow
│
├── pages/                              # Streamlit multi-page app
│   ├── 1_Individual_Stock.py           # Single-stock view (candlestick, indicators, flows)
│   ├── 2_Sector_Heatmap.py             # Sector fund-rotation heatmap
│   └── 3_Observation_Pool.py           # Watchlist overview
│
├── fetch_taiwan50_to_raw.py            # Daily fetch: price, institutional flow, margin/short data
├── fetch_eps_to_raw.py                 # Fetch quarterly financials (EPS, gross profit, net income, equity)
├── transform_to_staging.py             # Raw-to-staging data cleaning
├── calc_technical_indicators.py        # Computes MA / RSI / MACD / KD / ATR
├── load_stock_info.py                  # Imports company names & industry classification
├── taiwan_stocks_categorized_2.csv     # Industry classification mapping for the watchlist
│
├── .github/workflows/daily_update.yml  # Daily automated update schedule (GitHub Actions)
├── .devcontainer/                      # Dev container config
├── requirements.txt
├── secrets.toml.example                # Streamlit secrets template
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / Dashboard | Streamlit |
| Database | PostgreSQL (hosted on Supabase) |
| Data source | [FinMind API](https://finmindtrade.com/) (price, institutional flow, financials) |
| Data processing | pandas, numpy, SQLAlchemy |
| Scheduling | GitHub Actions (daily update on trading days) |
| Deployment | Streamlit Community Cloud |

**Database layers**: `raw` (source data) → `staging` (cleaned/transformed) → `mart` (technical indicators)

---

## Getting Started (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure database credentials

Copy `secrets.toml.example` to `.streamlit/secrets.toml` and fill in your Supabase connection details:

```toml
db_user = "postgres.xxxxxxxxxxxx"
db_password = "your_password"
db_host = "aws-0-ap-northeast-1.pooler.supabase.com"
db_port = "5432"
db_name = "postgres"
```

The batch scripts (`fetch_*.py`, `transform_*.py`, `calc_*.py`) also need a `.env` file in the project root with the same values under uppercase environment variable names: `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `FINMIND_TOKEN`.

### 3. Initialize the database

Run in order:

```bash
python load_stock_info.py taiwan_stocks_categorized_2.csv
python fetch_taiwan50_to_raw.py
python fetch_eps_to_raw.py
python transform_to_staging.py
python calc_technical_indicators.py
```

### 4. Launch the app

```bash
streamlit run Swing_Trading_Decision_System.py
```

---

## Daily Automated Updates

`.github/workflows/daily_update.yml` runs the price and technical-indicator update scripts at a fixed time on every Taiwan trading day and writes the results back to Supabase. The Streamlit Cloud deployment picks up the latest data once its cache expires.

---

## Limitations

1. **No awareness of news/event-driven moves**: the system scores primarily on technical and institutional-flow factors, with fundamentals acting as a pass/fail gate. It has no visibility into news, policy changes, or geopolitical events — factors that often drive the sharpest short-term price swings.
2. **Valuation judgment carries subjectivity**: the fundamentals gate's valuation check (whether P/E or P/B is significantly above the industry benchmark) involves subjective calibration, and "overvalued" does not mean a stock won't keep rising — momentum and valuation are independent dimensions. The system deliberately checks both, which means it can miss pure momentum-driven rallies.
3. **EPS estimates carry uncertainty**: the EPS-growth check relies on an estimate of full-year EPS (via seasonal extrapolation), not an actual reported figure, so it can diverge from the real year-end results.
4. **Subjectivity in the margin-decline threshold**: the gross margin trend itself uses only actual historical figures (no forecasting), but the threshold for what counts as a "material" decline is a subjective judgment call, and may still exclude stocks with strong momentum that are simply experiencing short-term margin noise.
5. **A deliberate screening philosophy**: the system is designed to surface stocks that are both "fundamentally sound" and "technically/institutionally strong" — not to capture every stock that's rallying. Some stocks with strong price action but failing fundamentals will be excluded by design; this is an intentional trade-off, not a bug.
6. **Chasing risk and data-freshness constraints**: even with the fundamentals gate in place, the current methodology can still generate signals after a move is already underway, carrying chasing risk. Data is refreshed once daily, so signals can lag intraday price action.

**⚠️ All scoring weights, valuation thresholds, and risk parameters in this project are initial assumptions set during development based on trading heuristics — they have not been validated through historical backtesting. This project is for technical demonstration and learning purposes only and does not constitute investment advice.**
