# Taiwan Stock Swing Trading Decision System

A full-stack project spanning data ingestion, database design, technical indicator computation, a multi-dimensional decision engine, and an interactive dashboard — built to help identify Taiwan-listed stocks that pass a fundamentals health check **and** show strong technical/institutional momentum for swing trading.

**Live Demo**: [https://taiwan-stock-dashboard-jumf.streamlit.app/](https://taiwan-stock-dashboard-jumf.streamlit.app/)

---

## Motivation

Most retail stock-screening tools fall into one of two camps: pure technical screens (prone to chasing overextended moves, blind to fundamental risk) or pure fundamental analysis (no sense of timing). This project separates "fundamental health" and "technical/institutional timing" into two independent judgment dimensions, so a strong technical signal isn't diluted by an average fundamental score (or vice versa). It also adjusts its valuation methodology for industries where P/E ratios are known to be unreliable — notably cyclical sectors like memory chips and shipping, where earnings swing dramatically with commodity cycles — and applies risk filters (overextension, volume confirmation, market regime) so that a high score alone doesn't automatically become a buy signal.

---

## Core Features

### 1. Three-Tier Decision Signal

For every stock in the watchlist, the system evaluates a fundamentals gate check and a technical/institutional score, then applies a market-regime filter, and outputs one of three signals:

| Signal | Condition |
|---|---|
| **Buy (BUY_NOW)** | Fundamentals pass the gate check, Stock Score ≥ 85, **and** the market regime is not bearish |
| **Watch (WATCH)** | Fundamentals pass the gate check, but Stock Score < 85 — or the score qualifies but the market regime is bearish |
| **Avoid (AVOID)** | Fundamentals fail the gate check (further split into "fundamental red flag" / "fails both") |

### 2. Fundamentals Gate (a pass/fail checklist, not a weighted score)

- **Valuation check**: whether P/E (non-cyclical stocks) or P/B (cyclical stocks) is significantly above the industry benchmark
- **EPS growth**: whether estimated full-year EPS beats last year's
- **Gross margin trend**: whether gross margin has declined materially versus the same period last year (measured as a *relative* decline percentage, not raw percentage points, to avoid unfairly penalizing high-margin industries)

Cyclical stocks (shipping, memory semiconductors, plastics/petrochemicals, steel) have earnings driven primarily by commodity cycles rather than company-specific execution, so their valuation check uses **price-to-book (P/B)** instead of P/E, and they're exempt from the margin-decline check.

### 3. Technical/Institutional Score (Stock Score, out of 100)

Five factors are scored 0–100 and combined with fixed weights:

| Factor | Weight | How it's measured |
|---|---|---|
| Trend | 25% | Close vs. MA5 / MA10 / MA60 alignment, with each gap **normalized by ATR14** so stocks of different volatility are judged on the same scale |
| Momentum | 20% | RSI(14) and MACD histogram |
| Relative strength | 20% | True 20-trading-day return versus the benchmark (0050) |
| Institutional net buying | 20% | Share of net-buy days over the last 10 trading days, plus trend in net-buy amount |
| Sector fund flow | 15% | Industry rank in the 10-day institutional fund-rotation score |

Most sub-scores use continuous sigmoid-based scoring rather than discrete thresholds, avoiding "cliff-edge" effects where a stock narrowly missing a cutoff gets zeroed out entirely.

**Risk discounts** are then applied as multipliers on the weighted score (they do not add a sixth weighted factor):

| Discount | What it penalizes | Effect |
|---|---|---|
| **Extension** | Price stretched far above its 10-day average, measured as (close − MA10) ÷ ATR14 | No discount up to 2 ATR; scales down to ×0.70 at 4.5 ATR or more |
| **Trend age** | Number of consecutive days the close has stayed above MA10 | Mild: at most −5%, so long-running leaders are not locked out of a Buy |
| **Volume** | Distribution: up-day volume ÷ down-day volume over the last 20 days (an IBD/CANSLIM-style measure) | No discount at a ratio ≥ 1.0; scales down to ×0.88 at a ratio ≤ 0.6 |

### 4. Market Regime Filter

Individual bullish signals tend to be less reliable when the overall market is weak, so the system classifies the market using the benchmark ETF (0050) and its 20-day and 60-day moving averages:

| Regime | Definition | Effect |
|---|---|---|
| Bullish | Close ≥ MA60 | No change |
| Neutral / weakening | Close < MA60, but MA20 is still above MA60 | No change (shown as a caution) |
| Bearish | Close < MA60 **and** MA20 < MA60 | All Buy signals are downgraded to Watch, with the reason shown separately from "score too low" |

The current regime is displayed at the top of the dashboard.

### 5. EPS Estimation Engine

Taiwan-listed companies disclose standalone quarterly EPS, not cumulative figures. The system prioritizes a seasonally-adjusted extrapolation method (this year's H1 EPS × last year's full-year/H1 growth ratio) to estimate full-year EPS, with multiple layers of guardrails for cyclical stocks and abnormal year-ago base periods. When extrapolation isn't reliable, it falls back to trailing-twelve-months (TTM) actuals.

### 6. Sector Fund Flow Heatmap

Computes a fund-rotation score per industry from institutional net buying and price momentum, visualized as a treemap.

### 7. Bilingual Interface

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
│   ├── db.py                           # Database connection (shared pooled engine) & queries
│   ├── pipeline.py                     # Orchestrates the full data flow into final decisions
│   ├── engine.py                       # Three-tier signal logic and market-regime downgrade
│   ├── scoring.py                      # Stock Score, risk discounts, market regime
│   ├── valuation.py                    # Valuation checks, EPS estimation, P/B computation
│   ├── zones.py                        # Breakout price computation — development-history
│   │                                    # artifact, not called by the live decision flow
│   ├── risk_reward.py                  # Stop-loss / target price helpers — development-history
│   │                                    # artifact, not called by the live decision flow
│   └── pattern.py                      # Technical pattern classification (breakout/pullback) —
│                                        # development-history artifact, not called by the live flow
│
├── pages/                              # Streamlit multi-page app
│   ├── 1_Individual_Stock.py           # Single-stock view (candlestick, indicators, flows)
│   ├── 2_Sector_Heatmap.py             # Sector fund-rotation heatmap
│   └── 3_Observation_Pool.py           # Watchlist overview
│
├── fetch_taiwan50_to_raw.py            # Daily fetch: price, institutional flow, margin/short data
├── fetch_eps_to_raw.py                 # Fetch quarterly financials (EPS, gross profit, net income, equity)
├── transform_to_staging.py             # Raw-to-staging data cleaning
├── calc_technical_indicators.py        # Incrementally computes MA / RSI / MACD / KD / ATR
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

## Engineering Notes

A few design choices made for long-term stability on a free-tier hosted database:

- **Bounded queries**: queries are restricted to the date ranges they need, so runtime stays flat as history accumulates.
- **Incremental indicator computation**: only newly arrived rows are computed (with a warm-up buffer for the moving averages); results were verified to match a full recompute.
- **Data-aware caching**: the dashboard's computed results are cached against the database's latest data date, so new data triggers a recompute automatically without waiting for a manual refresh.
- **Graceful failure**: if the database is unreachable, visitors see a short maintenance notice instead of a stack trace.
- **Timing logs**: each stage of the decision pipeline logs its elapsed time, which makes bottlenecks easy to locate.

---

## Daily Automated Updates

`.github/workflows/daily_update.yml` runs the price and technical-indicator update scripts at a fixed time on every Taiwan trading day and writes the results back to Supabase. Once new data lands, the dashboard detects the newer data date and recomputes automatically (within about 10 minutes of a visit).

---

## Limitations

1. **No awareness of news/event-driven moves**: the system scores primarily on technical and institutional-flow factors, with fundamentals acting as a pass/fail gate. It has no visibility into news, policy changes, or geopolitical events — factors that often drive the sharpest short-term price swings.
2. **Valuation judgment carries subjectivity**: the fundamentals gate's valuation check (whether P/E or P/B is significantly above the industry benchmark) involves subjective calibration, and "overvalued" does not mean a stock won't keep rising — momentum and valuation are independent dimensions. The system deliberately checks both, which means it can miss pure momentum-driven rallies.
3. **EPS estimates carry uncertainty**: the EPS-growth check relies on an estimate of full-year EPS (via seasonal extrapolation), not an actual reported figure, so it can diverge from the real year-end results.
4. **Subjectivity in the margin-decline threshold**: the gross margin trend itself uses only actual historical figures (no forecasting), but the threshold for what counts as a "material" decline is a subjective judgment call, and may still exclude stocks with strong momentum that are simply experiencing short-term margin noise.
5. **A deliberate screening philosophy**: the system is designed to surface stocks that are both "fundamentally sound" and "technically/institutionally strong" — not to capture every stock that's rallying. Some stocks with strong price action but failing fundamentals will be excluded by design; this is an intentional trade-off, not a bug.
6. **Chasing risk and data-freshness constraints**: the extension and volume discounts reduce, but do not eliminate, the chance of a signal appearing after a move is already underway. Data is refreshed once daily, so signals can lag intraday price action.
7. **A strength screener, not a full trading system**: the system identifies strong, fundamentally sound stocks, but it does not provide entry prices, stop-losses, targets, position sizing, or exit rules, and the market-regime filter is a simple moving-average rule on a single benchmark. Risk management remains the user's responsibility.

**⚠️ This project is for technical demonstration and learning purposes only and does not constitute investment advice.**
