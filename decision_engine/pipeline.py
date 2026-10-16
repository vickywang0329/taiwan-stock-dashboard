"""
decision_engine/pipeline.py
------------------------------
把整個觀察池跑過一次 Decision Engine，回傳給 Streamlit 頁面用的結果。

⚠️ sector_flow_score 目前用簡化版（假設 sector_rank_pct 已經算好傳進來），
   實際串接時請改成呼叫 pages/sector_heatmap.py 裡既有的資金輪動排名邏輯，
   兩邊共用同一份排名結果，不要重算兩次。
"""
from __future__ import annotations
import time
import pandas as pd
import numpy as np

from watchlist import WATCHLIST  # 專案既有的觀察池清單（單一來源）
from . import db, scoring, engine, valuation
import sector_flow


_T0 = [0.0]


def _tick(label: str) -> None:
    """計時 log：印出每個步驟距離上一步花了幾秒（Streamlit Cloud 的 Manage app log 看得到）。"""
    now = time.perf_counter()
    print(f"[pipeline] {label}: {now - _T0[0]:.1f}s", flush=True)
    _T0[0] = now


def _n_day_return(close_series: pd.Series, n: int) -> float:
    """
    真正的 N 個交易日報酬：最新收盤 / N 個交易日前的收盤 - 1。
    歷史不足 N+1 筆時，退而使用能取得的最早一筆（避免新股/缺資料直接報錯）。
    """
    if len(close_series) < 2:
        return 0.0
    base = close_series.iloc[-(n + 1)] if len(close_series) > n else close_series.iloc[0]
    return float(close_series.iloc[-1] / base - 1)


def _compute_sub_scores(stock_hist: pd.DataFrame, indicators_row: pd.Series,
                         benchmark_return: float, sector_rank_pct: float) -> dict:
    """
    只回傳五項純技術籌碼子分數（trend/momentum/relative_strength/
    institutional_flow/sector_flow），供 scoring.compute_tech_score() 使用。
    ⚠️ 基本面（估值/EPS成長/毛利率趨勢）已改成獨立的守門檢查，不再放進
    這個字典裡加權——見 engine.check_fundamentals()。
    """
    close = stock_hist["close"].iloc[-1]
    stock_return = _n_day_return(stock_hist["close"], scoring.RS_LOOKBACK_DAYS)

    net_buy_days = (stock_hist["institutional_net"].tail(scoring.INSTITUTIONAL_LOOKBACK_DAYS) > 0).sum()
    total_days = min(scoring.INSTITUTIONAL_LOOKBACK_DAYS, len(stock_hist))
    net_trend_up = (
        stock_hist["institutional_net"].tail(5).mean()
        > stock_hist["institutional_net"].tail(10).head(5).mean()
        if len(stock_hist) >= 10 else False
    )

    return {
        "trend": scoring.trend_score(
            close, indicators_row["ma5"], indicators_row["ma10"], indicators_row["ma60"],
            atr14=indicators_row.get("atr14"),
        ),
        "momentum": scoring.momentum_score(
            indicators_row["rsi14"], indicators_row.get("macd_hist")
        ),
        "relative_strength": scoring.relative_strength_score(stock_return, benchmark_return),
        "institutional_flow": scoring.institutional_flow_score(
            int(net_buy_days), int(total_days), bool(net_trend_up)
        ),
        "sector_flow": scoring.sector_flow_score(sector_rank_pct),
    }


def run_decision_system(sector_rank_lookup: dict[str, float] | None = None) -> pd.DataFrame:
    """
    sector_rank_lookup: {sector_zh: rank_pct(0~1)}，由 sector_flow.py 的排名邏輯提供
    （跟 pages/sector_heatmap.py 共用同一套算法，數字不會對不上）。
    未傳入時，自動呼叫 sector_flow.get_sector_rank_lookup() 取得近10日的真實排名。
    """
    _T0[0] = time.perf_counter()
    print("[pipeline] start", flush=True)
    if sector_rank_lookup is None:
        sector_rank_lookup = sector_flow.get_sector_rank_lookup(window="10d")

    _tick("sector_rank_lookup")
    stock_ids = list(WATCHLIST)
    price_hist = db.load_price_history(stock_ids, lookback_days=90)
    _tick("load_price_history")
    indicators = db.load_latest_indicators(stock_ids).set_index("stock_id")
    _tick("load_latest_indicators")
    info = db.load_stock_info(stock_ids).set_index("stock_id")
    _tick("load_stock_info")
    eps_raw = db.load_eps_quarterly(stock_ids)
    _tick("load_eps_quarterly")
    ma10_hist = db.load_ma10_history(stock_ids, lookback_days=60)

    # ---- 效能優化：先依股票代碼分組一次，後面對每檔股票就是 O(1) 查表，
    #      不用每次都對整份 90天×163檔 的價格表重新做一次線性掃描。----
    _tick("load_ma10_history")
    price_hist_by_stock = {sid: g.sort_values("date") for sid, g in price_hist.groupby("stock_id")}
    eps_by_stock = {sid: g for sid, g in eps_raw.groupby("stock_id")} if not eps_raw.empty else {}
    ma10_hist_by_stock = (
        {sid: g.sort_values("date") for sid, g in ma10_hist.groupby("stock_id")}
        if not ma10_hist.empty else {}
    )

    benchmark_id = db.COLUMNS["benchmark_stock_id"]
    bm_hist = price_hist_by_stock.get(benchmark_id, pd.DataFrame())
    benchmark_return = (
        _n_day_return(bm_hist["close"], scoring.RS_LOOKBACK_DAYS) if len(bm_hist) > 1 else 0.0
    )

    # ---- 大盤環境（只算一次，所有個股共用）：用 0050 的收盤價判斷多頭／盤整／空頭 ----
    regime, regime_gap_pct = (
        scoring.market_regime(bm_hist["close"].reset_index(drop=True))
        if not bm_hist.empty else ("unknown", None)
    )
    print(f"[pipeline] market regime: {regime} (0050 vs MA60: {regime_gap_pct})", flush=True)

    # ---- 預先算好「全體股票」的本益比(P/E)跟股價淨值比(P/B)，才能建立
    #      同產業基準（缺一不可，同業基準需要看過整個觀察池才算得出來，
    #      不能逐股計算時各算各的）。景氣循環股用P/B、非景氣循環股用P/E，
    #      兩套都先算好，之後在主迴圈依股票所屬產業選用對應那一套。----
    pe_records = []
    pb_records = []
    estimate_lookup = {}
    is_loss_lookup = {}
    last_year_fy_eps_lookup = {}
    bvps_status_lookup = {}
    for stock_id in stock_ids:
        hist = price_hist_by_stock.get(stock_id)
        if hist is None or hist.empty or stock_id not in info.index:
            continue
        close = float(hist["close"].iloc[-1])
        stock_eps_df = eps_by_stock.get(stock_id, pd.DataFrame())

        estimate, method = valuation.estimate_annual_eps(stock_eps_df)
        estimate_lookup[stock_id] = estimate
        is_loss_lookup[stock_id] = (method == "loss")
        last_year_fy_eps_lookup[stock_id] = valuation.get_last_year_full_year_eps(stock_eps_df)
        pe = valuation.compute_pe(close, estimate)
        if pe is not None:
            pe_records.append({"stock_id": stock_id, "industry": info.loc[stock_id, "sector_zh"], "pe": pe})

        bvps, bvps_status = valuation.compute_book_value_per_share(stock_eps_df)
        bvps_status_lookup[stock_id] = bvps_status
        pb = valuation.compute_pb(close, bvps)
        if pb is not None:
            pb_records.append({"stock_id": stock_id, "industry": info.loc[stock_id, "sector_zh"], "pb": pb})

    _tick("precompute PE/PB")
    pe_df = pd.DataFrame(pe_records)
    industry_pe_benchmark = (
        valuation.compute_industry_pe_benchmark(pe_df) if not pe_df.empty else {}
    )
    pe_lookup = dict(zip(pe_df["stock_id"], pe_df["pe"])) if not pe_df.empty else {}

    pb_df = pd.DataFrame(pb_records)
    industry_pb_benchmark = (
        valuation.compute_industry_pb_benchmark(pb_df) if not pb_df.empty else {}
    )
    pb_lookup = dict(zip(pb_df["stock_id"], pb_df["pb"])) if not pb_df.empty else {}

    _tick("industry benchmarks")
    results = []
    for stock_id in stock_ids:
        hist = price_hist_by_stock.get(stock_id)
        if hist is None or len(hist) < 25 or stock_id not in indicators.index:
            continue  # 資料不足先跳過

        ind_row = indicators.loc[stock_id]
        name = info.loc[stock_id, "name_zh"] if stock_id in info.index else stock_id
        name_en = info.loc[stock_id, "name_en"] if stock_id in info.index else stock_id
        sector = info.loc[stock_id, "sector_zh"] if stock_id in info.index else None
        sector_rank_pct = (
            sector_rank_lookup.get(sector, 0.5) if sector_rank_lookup and sector else 0.5
        )
        is_cyclical = sector in valuation.CYCLICAL_INDUSTRIES if sector else False

        stock_estimate = estimate_lookup.get(stock_id)
        stock_is_loss = is_loss_lookup.get(stock_id, False)

        # 龍頭股用比較寬鬆的過虛門檻（見 valuation.LEADING_STOCKS_THRESHOLD 的說明），
        # 查不到就用全市場預設的 valuation.OVERVALUATION_THRESHOLD
        custom_threshold = valuation.LEADING_STOCKS_THRESHOLD.get(stock_id, valuation.OVERVALUATION_THRESHOLD)

        if is_cyclical:
            # 景氣循環股：valuation 改用 P/B，不用 P/E（見使用者確認的設計）
            stock_pb = pb_lookup.get(stock_id)
            industry_avg_pb = industry_pb_benchmark.get(sector)
            has_neg_equity = bvps_status_lookup.get(stock_id) == "negative_equity"
            overvalued = valuation.is_overvalued(
                stock_pb, industry_avg_pb, is_loss=has_neg_equity, threshold_multiple=custom_threshold,
            )
            # 合理價格用P/B×每股淨值反推，跟非循環股用P/E×EPS的概念一致，
            # 但循環股的每股淨值需要重新算一次（前面預備階段沒有保留bvps本身，只留了狀態）
            stock_eps_df_for_bvps = eps_by_stock.get(stock_id, pd.DataFrame())
            bvps, _ = valuation.compute_book_value_per_share(stock_eps_df_for_bvps)
            fair_value = (
                bvps * industry_avg_pb
                if bvps is not None and bvps > 0 and industry_avg_pb is not None
                else None
            )
        else:
            industry_avg_pe = industry_pe_benchmark.get(sector)
            stock_pe = pe_lookup.get(stock_id)
            overvalued = valuation.is_overvalued(
                stock_pe, industry_avg_pe, is_loss=stock_is_loss, threshold_multiple=custom_threshold,
            )
            # 合理價格 = 估算全年EPS × 產業同業本益比基準
            # ⚠️ 要求 EPS 估算值必須是正值——虧損公司用本益比×EPS算「合理價格」
            # 沒有意義（會算出負的價格），比照 compute_pe() 同樣的邏輯處理
            fair_value = (
                stock_estimate * industry_avg_pe
                if stock_estimate is not None and stock_estimate > 0 and industry_avg_pe is not None
                else None
            )

        growing = valuation.eps_growing(stock_estimate, last_year_fy_eps_lookup.get(stock_id))
        stock_financials_df = eps_raw[eps_raw["stock_id"] == stock_id]
        margin_declining = valuation.is_margin_severely_declining(stock_financials_df)

        sub_scores = _compute_sub_scores(hist, ind_row, benchmark_return, sector_rank_pct)

        close = hist["close"].iloc[-1]

        # ---- 動能新鮮度：算出收盤價連續站上MA10幾天，換算成分數折扣係數 ----
        stock_ma10_hist = ma10_hist_by_stock.get(stock_id)
        if stock_ma10_hist is not None and not stock_ma10_hist.empty:
            merged = hist[["date", "close"]].merge(
                stock_ma10_hist[["date", "ma10"]], on="date", how="inner",
            ).sort_values("date")
            days_count = scoring.days_above_ma(merged["close"], merged["ma10"])
        else:
            days_count = None
        vol_mult = scoring.volume_multiplier(
            scoring.updown_volume_ratio(hist["close"], hist["volume"])
            if "volume" in hist.columns else None
        )
        freshness_multiplier = scoring.combined_freshness_multiplier(
            days_count, float(close), ind_row.get("ma10"), ind_row.get("atr14"),
        )

        decision = engine.decide(
            stock_id=stock_id,
            name=name,
            close=float(close),
            sub_scores=sub_scores,
            is_overvalued=overvalued,
            eps_growing=growing,
            margin_severely_declining=margin_declining,
            is_cyclical=is_cyclical,
            fair_value_estimate=fair_value,
            freshness_multiplier=freshness_multiplier,
            volume_multiplier=vol_mult,
            market_regime=regime,
            market_gap_pct=regime_gap_pct,
        )
        decision.name_en = name_en
        results.append(decision)

    _tick("per-stock decisions")
    return results
