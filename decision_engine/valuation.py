"""
decision_engine/valuation.py
----------------------------
估值檢查：估算全年 EPS、算本益比、跟同產業基準比較，判斷是否「估值過虛」。

⚠️ 這裡的門檻（1.5倍同業平均）是初始假設，之後要用回測校準。

計算邏輯（已與使用者確認定案）：
1. 全年 EPS 估算，優先用「季節調整外推法」：
   全年估算 = 今年H1累計EPS × (去年全年累計EPS ÷ 去年H1累計EPS)
   （財報空窗期，今年H1資料還沒公布時）備援改用 TTM（近四季合計）
2. 本益比 = 現價 ÷ 估算全年EPS
3. 同產業基準 = 同產業其他股票本益比，排除虧損股（PE<=0）與極端值後的平均
4. 本益比 > 同業基準 × 1.5 倍 → 估值過虛（valuation 子分數不通過）
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _quarter_of(d: pd.Timestamp) -> int:
    """財報季底月份對應到第幾季：3月=Q1, 6月=Q2, 9月=Q3, 12月=Q4"""
    return {3: 1, 6: 2, 9: 3, 12: 4}.get(d.month, 0)


def decumulate_quarterly_eps(stock_eps_df: pd.DataFrame) -> pd.DataFrame:
    """
    把台股財報慣例的「季累計EPS」，還原成「單季EPS」。
    stock_eps_df 需含 date（季底日）、eps_cumulative，且是單一股票的資料。
    Q1 本身就是單季（年度重新累計的起點）；Q2/Q3/Q4 要減去上一季的累計值，
    遇到跨年度（從去年Q4銜接到今年Q1）不能相減，Q1 一律直接採用累計值本身。
    """
    df = stock_eps_df.sort_values("date").reset_index(drop=True).copy()
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].apply(_quarter_of)

    standalone = []
    prev_row = None
    for _, row in df.iterrows():
        if row["quarter"] == 1 or prev_row is None or prev_row["year"] != row["year"]:
            standalone.append(row["eps_cumulative"])
        else:
            standalone.append(row["eps_cumulative"] - prev_row["eps_cumulative"])
        prev_row = row

    df["eps_standalone"] = standalone
    return df


def estimate_annual_eps(stock_eps_df: pd.DataFrame, today: pd.Timestamp | None = None) -> tuple[float | None, str]:
    """
    估算全年 EPS。回傳 (估算值, 使用的方法)，方法為 "extrapolation" 或 "ttm"，
    資料不足時回傳 (None, "insufficient_data")。
    """
    if today is None:
        today = pd.Timestamp.today()

    if stock_eps_df.empty:
        return None, "insufficient_data"

    df = stock_eps_df.sort_values("date").reset_index(drop=True).copy()
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].apply(_quarter_of)

    this_year, last_year = today.year, today.year - 1

    h1_this = df[(df["year"] == this_year) & (df["quarter"] == 2)]
    h1_last = df[(df["year"] == last_year) & (df["quarter"] == 2)]
    fy_last = df[(df["year"] == last_year) & (df["quarter"] == 4)]

    # ---- 優先：季節調整外推法 ----
    if not h1_this.empty and not h1_last.empty and not fy_last.empty:
        h1_this_val = float(h1_this.iloc[0]["eps_cumulative"])
        h1_last_val = float(h1_last.iloc[0]["eps_cumulative"])
        fy_last_val = float(fy_last.iloc[0]["eps_cumulative"])
        if h1_last_val != 0:
            estimate = h1_this_val * (fy_last_val / h1_last_val)
            return estimate, "extrapolation"

    # ---- 備援：TTM（近四季合計，財報空窗期或去年H1為0時使用）----
    decumulated = decumulate_quarterly_eps(df)
    recent_quarters = decumulated.tail(4)
    if len(recent_quarters) == 4:
        ttm = float(recent_quarters["eps_standalone"].sum())
        return ttm, "ttm"

    return None, "insufficient_data"


def compute_pe(current_price: float, estimated_annual_eps: float | None) -> float | None:
    """本益比 = 現價 ÷ 估算全年EPS。EPS <= 0（虧損或估算失敗）視為無法計算，回傳 None。"""
    if estimated_annual_eps is None or estimated_annual_eps <= 0:
        return None
    return current_price / estimated_annual_eps


def get_last_year_full_year_eps(stock_eps_df: pd.DataFrame, today: pd.Timestamp | None = None) -> float | None:
    """取得「去年全年（Q4累計）EPS」，供判斷今年估算EPS是否較去年成長使用。"""
    if today is None:
        today = pd.Timestamp.today()
    if stock_eps_df.empty:
        return None

    df = stock_eps_df.copy()
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].apply(_quarter_of)

    last_year = today.year - 1
    fy_last = df[(df["year"] == last_year) & (df["quarter"] == 4)]
    if fy_last.empty:
        return None
    return float(fy_last.iloc[0]["eps_cumulative"])


def eps_growing(estimated_annual_eps: float | None, last_year_full_year_eps: float | None) -> bool:
    """
    今年估算全年EPS 是否較去年全年EPS 成長。
    缺資料時預設為 True（不因資料缺漏而卡關，跟 valuation 的中性原則一致）。
    """
    if estimated_annual_eps is None or last_year_full_year_eps is None:
        return True
    return estimated_annual_eps > last_year_full_year_eps


def compute_industry_pe_benchmark(pe_by_industry: pd.DataFrame) -> dict[str, float]:
    """
    pe_by_industry 需含 industry、pe 兩欄（每檔股票一列）。
    排除 PE<=0（虧損股）與極端值（IQR法：超過 Q3+1.5*IQR）後，計算每個產業的平均本益比。
    回傳 {產業: 平均本益比}。
    """
    result = {}
    for industry, group in pe_by_industry.groupby("industry"):
        valid = group[group["pe"] > 0]["pe"]
        if len(valid) < 2:
            continue
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        cleaned = valid[valid <= upper_bound]
        if cleaned.empty:
            continue
        result[industry] = float(cleaned.mean())
    return result


def get_gross_margin_at(stock_financials_df: pd.DataFrame, year: int, quarter: int) -> float | None:
    """取得指定年度、季度（累計）的毛利率。營收用 毛利+營業成本 反推，避免猜測FinMind營收欄位名稱。"""
    if stock_financials_df.empty:
        return None
    df = stock_financials_df.copy()
    df["fyear"] = df["date"].dt.year
    df["fquarter"] = df["date"].apply(_quarter_of)
    row = df[(df["fyear"] == year) & (df["fquarter"] == quarter)]
    if row.empty:
        return None

    gp = row.iloc[0].get("gross_profit")
    cogs = row.iloc[0].get("cost_of_goods_sold")
    if gp is None or cogs is None or pd.isna(gp) or pd.isna(cogs):
        return None

    revenue = gp + cogs  # 會計恆等式：營收 = 毛利 + 營業成本
    if revenue == 0:
        return None
    return gp / revenue


def _find_latest_available_quarter(stock_financials_df: pd.DataFrame, today: pd.Timestamp) -> tuple[int, int] | None:
    """找出「今年」最新一筆有毛利/成本資料的季度，回傳 (year, quarter)，找不到回傳 None。"""
    if stock_financials_df.empty:
        return None
    df = stock_financials_df.dropna(subset=["gross_profit", "cost_of_goods_sold"])
    if df.empty:
        return None
    df = df.copy()
    df["fyear"] = df["date"].dt.year
    df["fquarter"] = df["date"].apply(_quarter_of)
    this_year_rows = df[df["fyear"] == today.year]
    if this_year_rows.empty:
        return None
    latest = this_year_rows.sort_values("date").iloc[-1]
    return int(latest["fyear"]), int(latest["fquarter"])


MARGIN_SENSITIVITY = 2.0  # ⚠️ 初始假設，之後用回測校準


def margin_trend_score(stock_financials_df: pd.DataFrame, today: pd.Timestamp | None = None) -> float:
    """
    毛利率趨勢分數（漸進式 0-100，不是硬指標）：
    拿「今年最新一期累計毛利率」跟「去年同一期累計毛利率」比較（同期比同期），
    用 S 型函數把變化幅度（百分點）映射成分數：
    - 毛利率持平（變化=0） → 50分（中性）
    - 毛利率上升越多 → 分數越接近100
    - 毛利率下降越多 → 分數越接近0
    缺資料時回傳中性 50 分。
    """
    if today is None:
        today = pd.Timestamp.today()

    latest_period = _find_latest_available_quarter(stock_financials_df, today)
    if latest_period is None:
        return 50.0
    year, quarter = latest_period

    margin_this = get_gross_margin_at(stock_financials_df, year, quarter)
    margin_last = get_gross_margin_at(stock_financials_df, year - 1, quarter)
    if margin_this is None or margin_last is None:
        return 50.0

    change_pct_points = (margin_this - margin_last) * 100
    score = 100 / (1 + np.exp(-change_pct_points / MARGIN_SENSITIVITY))
    return float(np.clip(score, 0, 100))


OVERVALUATION_THRESHOLD = 1.5  # ⚠️ 初始假設，之後用回測校準


def valuation_score(pe: float | None, industry_avg_pe: float | None, threshold_multiple: float = OVERVALUATION_THRESHOLD) -> float:
    """
    估值子分數（門檻式，不是漸進分數）：
    - 缺資料（PE 或同業基準算不出來）→ 給滿分（不因資料缺漏而扣分，保持中性）
    - 本益比 <= 同業基準 × threshold_multiple → 100（通過，估值合理）
    - 本益比 > 同業基準 × threshold_multiple → 0（不通過，估值過虛）
    """
    if pe is None or industry_avg_pe is None or industry_avg_pe <= 0:
        return 100.0
    return 100.0 if pe <= industry_avg_pe * threshold_multiple else 0.0


def is_overvalued(pe: float | None, industry_avg_pe: float | None, threshold_multiple: float = OVERVALUATION_THRESHOLD) -> bool:
    """
    布林版本，供 Decision Engine 判斷是否要強制排除 BUY_NOW（即使其他各項條件都達標）。
    缺資料時視為「不算過虛」（不強制排除），跟 valuation_score 的中性原則一致。
    """
    if pe is None or industry_avg_pe is None or industry_avg_pe <= 0:
        return False
    return pe > industry_avg_pe * threshold_multiple
