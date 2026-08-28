"""
sector_flow.py
----------------
產業資金輪動分數計算——這是「唯一算法版本」，
pages/sector_heatmap.py 跟 decision_engine 都應該從這裡呼叫，
不要各自重算一次，避免兩邊數字對不上。

公式沿用 pages/sector_heatmap.py 原本定案的邏輯：
法人資金強度佔80% + 價格動能佔20%，各自先做0-100正規化再加權。

⚠️ 這支模組不依賴 Streamlit（沒有 @st.cache 裝飾器），
   這樣 decision_engine 也能在非 Streamlit 環境下呼叫。
   pages/sector_heatmap.py 呼叫這裡的函式時，自己在外層包一層
   @st.cache_data 就好，快取行為不會受影響。
"""
from __future__ import annotations
import pandas as pd
from sqlalchemy import text

from decision_engine.db import get_engine


def load_sector_data() -> pd.DataFrame:
    """
    抓每檔股票：最新收盤價、最新成交值、近3/10/20/60日的三大法人合計買賣超股數之和、
    近3/10/20/60日的價格報酬率，並關聯產業分類（中文＋英文皆保留）
    """
    engine = get_engine()
    query = text("""
        WITH ranked AS (
            SELECT
                m.date, m.stock_id, m.close, m.trading_value,
                m.institutional_total_net,
                LAG(m.close, 3) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_3d_ago,
                LAG(m.close, 10) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_10d_ago,
                LAG(m.close, 20) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_20d_ago,
                LAG(m.close, 60) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_60d_ago,
                SUM(m.institutional_total_net) OVER (
                    PARTITION BY m.stock_id ORDER BY m.date
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS inst_net_3d_shares,
                SUM(m.institutional_total_net) OVER (
                    PARTITION BY m.stock_id ORDER BY m.date
                    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
                ) AS inst_net_10d_shares,
                ROW_NUMBER() OVER (PARTITION BY m.stock_id ORDER BY m.date DESC) AS rn
            FROM staging.daily_master m
        )
        SELECT r.date, r.stock_id, r.close, r.trading_value,
               r.close_3d_ago, r.close_10d_ago, r.close_20d_ago, r.close_60d_ago,
               r.inst_net_3d_shares, r.inst_net_10d_shares,
               s.industry AS industry_zh, s.industry_en, s.name_zh, s.name_en
        FROM ranked r
        LEFT JOIN raw.stock_info s ON s.stock_id = r.stock_id
        WHERE r.rn = 1
    """)
    df = pd.read_sql(query, engine)

    df["momentum_3d"] = (df["close"] - df["close_3d_ago"]) / df["close_3d_ago"] * 100
    df["inst_strength_3d"] = df["inst_net_3d_shares"] * df["close"]
    df["momentum_10d"] = (df["close"] - df["close_10d_ago"]) / df["close_10d_ago"] * 100
    df["inst_strength_10d"] = df["inst_net_10d_shares"] * df["close"]
    df["momentum_20d"] = (df["close"] - df["close_20d_ago"]) / df["close_20d_ago"] * 100
    df["momentum_60d"] = (df["close"] - df["close_60d_ago"]) / df["close_60d_ago"] * 100

    df = df.dropna(subset=["industry_zh"])
    df = df[df["industry_zh"].str.len() > 0]
    return df


def minmax_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo) * 100


def compute_sector_scores(
    df: pd.DataFrame, inst_col: str, momentum_col: str, group_col: str = "industry_zh"
) -> pd.DataFrame:
    """
    group_col：用哪個欄位分組。
    - pages/sector_heatmap.py 顯示用，傳入語言選定後的 "industry" 欄位（中/英文皆可）
    - decision_engine 背景運算用，固定傳入 "industry_zh"（不受語言切換影響）
    無論傳哪個欄位，底層的正規化與 80/20 加權公式完全相同，數字不會對不上。
    """
    valid = df.dropna(subset=[inst_col, momentum_col])
    sector_df = valid.groupby(group_col).agg(
        inst_strength=(inst_col, "sum"),
        momentum=(momentum_col, "mean"),
        stock_count=("stock_id", "count"),
    ).reset_index()

    sector_df["norm_inst"] = minmax_normalize(sector_df["inst_strength"])
    sector_df["norm_momentum"] = minmax_normalize(sector_df["momentum"])
    sector_df["rotation_score"] = sector_df["norm_inst"] * 0.8 + sector_df["norm_momentum"] * 0.2
    sector_df = sector_df.sort_values("rotation_score", ascending=False).reset_index(drop=True)
    return sector_df


def get_sector_rank_lookup(window: str = "10d") -> dict[str, float]:
    """
    給 decision_engine 用：回傳 {產業中文名: rank_pct}，
    rank_pct 是 0~1 的百分位（0＝資金輪動分數最強，1＝最弱），
    跟 scoring.sector_flow_score() 的定義一致（該函式內部會做 100*(1-rank_pct)）。

    window："3d" 或 "10d"，對應熱力圖頁面的近3日／近10日資金輪動分數。
    """
    df = load_sector_data()
    if df.empty:
        return {}

    inst_col, momentum_col = ("inst_strength_3d", "momentum_3d") if window == "3d" \
        else ("inst_strength_10d", "momentum_10d")

    scores = compute_sector_scores(df, inst_col, momentum_col, group_col="industry_zh")
    if scores.empty:
        return {}

    scores = scores.sort_values("rotation_score", ascending=False).reset_index(drop=True)
    n = len(scores)
    scores["rank_pct"] = scores.index / max(n - 1, 1)

    return dict(zip(scores["industry_zh"], scores["rank_pct"]))
