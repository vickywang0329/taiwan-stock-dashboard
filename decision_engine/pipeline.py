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
                         benchmark_return: float, sector_rank_pct: float) -> dict:
    """
    只回傳五項純技術籌碼子分數（trend/momentum/relative_strength/
    institutional_flow/sector_flow），供 scoring.compute_tech_score() 使用。
    ⚠️ 基本面（估值/EPS成長/毛利率趨勢）已改成獨立的守門檢查，不再放進
    這個字典裡加權——見 engine.check_fundamentals()。
    """
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

        if is_cyclical:
            stock_pb = pb_lookup.get(stock_id)
            industry_avg_pb = industry_pb_benchmark.get(sector)
            has_neg_equity = bvps_status_lookup.get(stock_id) == "negative_equity"
            custom_threshold = valuation.LEADING_STOCKS_THRESHOLD.get(stock_id, valuation.OVERVALUATION_THRESHOLD)
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
            custom_threshold = valuation.LEADING_STOCKS_THRESHOLD.get(stock_id, valuation.OVERVALUATION_THRESHOLD)
            overvalued = valuation.is_overvalued(
                stock_pe, industry_avg_pe, is_loss=stock_is_loss, threshold_multiple=custom_threshold,
            )

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
        )
        decision.name_en = name_en
        results.append(decision)

    return results
