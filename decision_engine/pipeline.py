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
from . import db, scoring, engine


def _compute_sub_scores(stock_hist: pd.DataFrame, indicators_row: pd.Series,
                         benchmark_return: float, sector_rank_pct: float) -> dict:
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
    }


def run_decision_system(sector_rank_lookup: dict[str, float] | None = None) -> pd.DataFrame:
    """
    sector_rank_lookup: {sector_zh: rank_pct(0~1)}，由 sector_heatmap.py 的排名邏輯提供。
    若未提供則全部視為中位數（0.5），僅供沒接上熱力圖排名時先跑得動。
    """
    stock_ids = list(WATCHLIST)
    price_hist = db.load_price_history(stock_ids, lookback_days=90)
    indicators = db.load_latest_indicators(stock_ids).set_index("stock_id")
    info = db.load_stock_info(stock_ids).set_index("stock_id")

    benchmark_id = db.COLUMNS["benchmark_stock_id"]
    bm_hist = price_hist[price_hist["stock_id"] == benchmark_id].sort_values("date")
    benchmark_return = (
        bm_hist["close"].iloc[-1] / bm_hist["close"].iloc[0] - 1 if len(bm_hist) > 1 else 0.0
    )

    results = []
    for stock_id in stock_ids:
        hist = price_hist[price_hist["stock_id"] == stock_id].sort_values("date")
        if len(hist) < 25 or stock_id not in indicators.index:
            continue  # 資料不足先跳過

        ind_row = indicators.loc[stock_id]
        name = info.loc[stock_id, "name_zh"] if stock_id in info.index else stock_id
        sector = info.loc[stock_id, "sector_zh"] if stock_id in info.index else None
        sector_rank_pct = (
            sector_rank_lookup.get(sector, 0.5) if sector_rank_lookup and sector else 0.5
        )

        sub_scores = _compute_sub_scores(hist, ind_row, benchmark_return, sector_rank_pct)

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
        )
        results.append(decision)

    return results
