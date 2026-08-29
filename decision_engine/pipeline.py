"""
decision_engine/pipeline.py
------------------------------
把整個觀察池跑過一次 Decision Engine，回傳給 Streamlit 頁面用的結果。

⚠️ sector_flow_score 目前用簡化版（假設 sector_rank_pct 已經算好傳進來），
   實際串接時請改成呼叫 pages/sector_heatmap.py 裡既有的資金輪動排名邏輯，
   兩邊共用同一份排名結果，不要重算兩次。
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from watchlist import WATCHLIST  # 專案既有的觀察池清單（單一來源）
from . import db, scoring, engine, valuation
import sector_flow


def _compute_sub_scores(stock_hist: pd.DataFrame, indicators_row: pd.Series,
                         benchmark_return: float, sector_rank_pct: float,
                         val_score: float, margin_score: float) -> dict:
    close = stock_hist["close"].iloc[-1]
    stock_return = close / stock_hist["close"].iloc[0] - 1

    net_buy_days = (stock_hist["institutional_net"].tail(scoring.INSTITUTIONAL_LOOKBACK_DAYS) > 0).sum()
    total_days = min(scoring.INSTITUTIONAL_LOOKBACK_DAYS, len(stock_hist))
    net_trend_up = (
        stock_hist["institutional_net"].tail(5).mean()
        > stock_hist["institutional_net"].tail(10).head(5).mean()
        if len(stock_hist) >= 10 else False
    )

    return {
        "trend": scoring.trend_score(
            close, indicators_row["ma5"], indicators_row["ma20"], indicators_row["ma60"]
        ),
        "momentum": scoring.momentum_score(
            indicators_row["rsi14"], indicators_row.get("macd_hist")
        ),
        "relative_strength": scoring.relative_strength_score(stock_return, benchmark_return),
        "institutional_flow": scoring.institutional_flow_score(
            int(net_buy_days), int(total_days), bool(net_trend_up)
        ),
        "sector_flow": scoring.sector_flow_score(sector_rank_pct),
        "valuation": val_score,
        "margin_trend": margin_score,
    }


def run_decision_system(sector_rank_lookup: dict[str, float] | None = None) -> pd.DataFrame:
    """
    sector_rank_lookup: {sector_zh: rank_pct(0~1)}，由 sector_flow.py 的排名邏輯提供
    （跟 pages/sector_heatmap.py 共用同一套算法，數字不會對不上）。
    未傳入時，自動呼叫 sector_flow.get_sector_rank_lookup() 取得近10日的真實排名。
    """
    if sector_rank_lookup is None:
        sector_rank_lookup = sector_flow.get_sector_rank_lookup(window="10d")

    stock_ids = list(WATCHLIST)
    price_hist = db.load_price_history(stock_ids, lookback_days=90)
    indicators = db.load_latest_indicators(stock_ids).set_index("stock_id")
    info = db.load_stock_info(stock_ids).set_index("stock_id")
    eps_raw = db.load_eps_quarterly(stock_ids)

    # ---- 效能優化：先依股票代碼分組一次，後面對每檔股票就是 O(1) 查表，
    #      不用每次都對整份 90天×163檔 的價格表重新做一次線性掃描。
    #      這個分組動作，原本在下面的迴圈裡對每檔股票各做了2次（前置計算
    #      一次、正式判斷一次），163檔×2=326次線性掃描，改成這樣只掃描一次。----
    price_hist_by_stock = {sid: g.sort_values("date") for sid, g in price_hist.groupby("stock_id")}
    eps_by_stock = {sid: g for sid, g in eps_raw.groupby("stock_id")} if not eps_raw.empty else {}

    benchmark_id = db.COLUMNS["benchmark_stock_id"]
    bm_hist = price_hist_by_stock.get(benchmark_id, pd.DataFrame())
    benchmark_return = (
        bm_hist["close"].iloc[-1] / bm_hist["close"].iloc[0] - 1 if len(bm_hist) > 1 else 0.0
    )

    # ---- 預先算好「全體股票」的本益比，才能建立同產業基準（缺一不可，
    #      同業基準需要看過整個觀察池才算得出來，不能逐股計算時各算各的）----
    pe_records = []
    estimate_lookup = {}
    last_year_fy_eps_lookup = {}
    for stock_id in stock_ids:
        hist = price_hist_by_stock.get(stock_id)
        if hist is None or hist.empty or stock_id not in info.index:
            continue
        close = float(hist["close"].iloc[-1])
        stock_eps_df = eps_by_stock.get(stock_id, pd.DataFrame())
        estimate, _method = valuation.estimate_annual_eps(stock_eps_df)
        estimate_lookup[stock_id] = estimate
        last_year_fy_eps_lookup[stock_id] = valuation.get_last_year_full_year_eps(stock_eps_df)
        pe = valuation.compute_pe(close, estimate)
        if pe is not None:
            pe_records.append({"stock_id": stock_id, "industry": info.loc[stock_id, "sector_zh"], "pe": pe})

    pe_df = pd.DataFrame(pe_records)
    industry_pe_benchmark = (
        valuation.compute_industry_pe_benchmark(pe_df) if not pe_df.empty else {}
    )
    pe_lookup = dict(zip(pe_df["stock_id"], pe_df["pe"])) if not pe_df.empty else {}

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

        stock_pe = pe_lookup.get(stock_id)
        industry_avg_pe = industry_pe_benchmark.get(sector)
        val_score = valuation.valuation_score(stock_pe, industry_avg_pe)
        overvalued = valuation.is_overvalued(stock_pe, industry_avg_pe)

        # 合理價格 = 估算全年EPS × 產業同業本益比基準
        stock_estimate = estimate_lookup.get(stock_id)
        fair_value = (
            stock_estimate * industry_avg_pe
            if stock_estimate is not None and industry_avg_pe is not None
            else None
        )

        growing = valuation.eps_growing(stock_estimate, last_year_fy_eps_lookup.get(stock_id))
        stock_financials_df = eps_raw[eps_raw["stock_id"] == stock_id]
        margin_score = valuation.margin_trend_score(stock_financials_df)

        sub_scores = _compute_sub_scores(hist, ind_row, benchmark_return, sector_rank_pct, val_score, margin_score)

        close = hist["close"].iloc[-1]
        high_20d = hist["high"].iloc[-21:-1]  # 不含當日
        low_20d = hist["low"].iloc[-21:]      # 含當日，用於停損
        prior_swing_high = float(hist["high"].tail(60).max())
        institutional_ok = hist["institutional_net"].tail(3).sum() > 0
        # 假突破：曾經站上近20日高點但收盤又跌破，簡化判斷，之後可再細修
        is_false_breakout = bool(
            (hist["close"].iloc[-2] >= hist["high"].iloc[-22:-2].max())
            and (close < hist["close"].iloc[-2])
        ) if len(hist) >= 22 else False

        decision = engine.decide(
            stock_id=stock_id,
            name=name,
            close=float(close),
            high_20d=high_20d,
            low_20d=low_20d,
            prior_swing_high=prior_swing_high,
            atr14=float(ind_row["atr14"]) if pd.notna(ind_row["atr14"]) else np.nan,
            ma60=float(ind_row["ma60"]),
            sub_scores=sub_scores,
            institutional_ok=bool(institutional_ok),
            is_false_breakout=is_false_breakout,
            is_overvalued=overvalued,
            fair_value_estimate=fair_value,
            eps_growing=growing,
        )
        decision.name_en = name_en
        results.append(decision)

    return results
