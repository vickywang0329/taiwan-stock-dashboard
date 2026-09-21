"""
decision_engine/db.py
----------------------
資料庫連線與欄位對照設定。

已對照專案真實 schema 修正 COLUMNS 字典（staging.daily_master /
mart.technical_indicators / raw.stock_info）。
"""
from __future__ import annotations
import os
from contextlib import contextmanager
from decimal import Decimal
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# 連線設定：沿用專案既有的 .env / st.secrets 模式
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str | None = None) -> str | None:
    """本機跑批次腳本用 .env，Streamlit Cloud 用 st.secrets。"""
    try:
        import streamlit as st  # noqa: WPS433
        if key.lower() in st.secrets:
            return st.secrets[key.lower()]
    except Exception:
        pass
    return os.environ.get(key, default)


def get_engine():
    user = _get_secret("DB_USER")
    password = _get_secret("DB_PASSWORD")
    host = _get_secret("DB_HOST", "aws-0-ap-northeast-1.pooler.supabase.com")
    port = _get_secret("DB_PORT", "5432")
    dbname = _get_secret("DB_NAME", "postgres")
    safe_password = quote_plus(password) if password else password
    url = f"postgresql+psycopg2://{user}:{safe_password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


@contextmanager
def get_conn():
    engine = get_engine()
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 欄位對照表 —— 已對照真實 schema 修正，之後 schema 若再變動只改這裡即可
# ---------------------------------------------------------------------------
COLUMNS = {
    "daily_master": {
        "table": "staging.daily_master",
        "stock_id": "stock_id",
        "date": "date",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        # 三大法人合計買賣超淨額，staging 層已經算好
        "institutional_net": "institutional_total_net",
        "foreign_net": "foreign_net",
        "trust_net": "investment_trust_net",
        "dealer_net": "dealer_net",
    },
    "technical_indicators": {
        "table": "mart.technical_indicators",
        "stock_id": "stock_id",
        "date": "date",
        "ma5": "ma5",
        "ma10": "ma10",
        "ma20": "ma20",
        "ma60": "ma60",
        "rsi14": "rsi_14",
        "macd": "macd",
        "macd_signal": "macd_signal",
        # 注意：資料庫裡沒有 macd_hist 欄位，改在 Python 端用 macd - macd_signal 算出
        "kd_k": "kd_k",
        "kd_d": "kd_d",
        "atr14": "atr14",
    },
    "stock_info": {
        "table": "raw.stock_info",
        "stock_id": "stock_id",
        "name_zh": "name_zh",
        "name_en": "name_en",
        "sector_zh": "industry",
        "sector_en": "industry_en",
    },
    "benchmark_stock_id": "0050",  # 相對強度計算的比較基準
}


def _query_df(sql: str, params: dict) -> pd.DataFrame:
    """
    用 SQLAlchemy 原生的 conn.execute() 執行查詢後手動組成 DataFrame，
    不透過 pd.read_sql()——某些 pandas / SQLAlchemy 版本組合下，
    pd.read_sql() 對 text() 物件的判斷有相容性問題，即使正確用 text()
    包裝，仍可能誤走到不支援具名參數的路徑，導致 ":ids" 這種語法
    直接被送進資料庫報錯。改用這個函式可以完全繞開該問題。

    ⚠️ 副作用：手動組 DataFrame 沒有 pd.read_sql() 自動把資料庫的
    NUMERIC 型別轉成 float 的機制，PostgreSQL 的數字欄位在這裡會被
    psycopg2 讀成 Python 原生的 decimal.Decimal，跟 float 混合運算
    會直接報錯（TypeError），所以這裡補一段：只要欄位裡出現過
    Decimal，就把整欄位轉成 float。
    """
    with get_conn() as conn:
        result = conn.execute(text(sql), params)
        rows = result.fetchall()
        columns = list(result.keys())
    df = pd.DataFrame(rows, columns=columns)

    # 效能優化：同一個資料庫欄位的型別必然一致，只需要檢查每欄「第一個非空值」
    # 是不是 Decimal，不用逐一掃描整欄——資料量大時（例如163檔股票×90天）
    # 這個差異很明顯（O(1) vs O(n) per 欄位）
    for col in df.columns:
        sample = df[col].dropna()
        if not sample.empty and isinstance(sample.iloc[0], Decimal):
            df[col] = df[col].astype(float)

    return df


def load_price_history(stock_ids: list[str], lookback_days: int = 90) -> pd.DataFrame:
    """
    撈近 N 個交易日的價格 + 三大法人淨額（用於算分數與技術指標）。

    ⚠️ 2026/09 修正記錄：原本 SQL 完全沒有日期篩選，撈出「全部歷史資料」
    後才在 Python 端用 .tail(lookback_days) 裁切——隨著 staging.daily_master
    每天自動累積新資料，這個查詢的執行時間會跟著資料庫總筆數持續增長，
    直到某天超過 Supabase 的 statement_timeout 被強制取消（跟 sector_flow.py
    同一輪修正發現的問題，這裡是同樣的反面模式，而且這個函式是主頁面
    每次載入都會呼叫的核心查詢，比 sector_flow.py 更急迫）。改成在SQL層
    就用日期範圍篩選，不管資料庫累積多少年的歷史，查詢處理的範圍都固定，
    執行時間維持穩定。

    緩衝天數：台股一年約240個交易日，換算日曆天數約為交易天數的1.52倍
    （365÷240），這裡用2.2倍再加30天當保守緩衝，確保篩選範圍內確實包含
    至少 lookback_days 個交易日，不會因為篩選過窄導致 tail() 抓不滿。
    """
    c = COLUMNS["daily_master"]
    calendar_days_buffer = int(lookback_days * 2.2) + 30
    sql = f"""
        select {c['stock_id']} as stock_id, {c['date']} as date,
               {c['close']} as close, {c['high']} as high, {c['low']} as low,
               {c['volume']} as volume,
               {c['institutional_net']} as institutional_net
        from {c['table']}
        where {c['stock_id']} = any(:ids)
          and {c['date']} >= (select max({c['date']}) from {c['table']}) - make_interval(days => :buffer_days)
        order by {c['stock_id']}, {c['date']}
    """
    df = _query_df(sql, {"ids": stock_ids, "buffer_days": calendar_days_buffer})
    # 只保留每檔股票最近 lookback_days 筆
    df = (
        df.sort_values(["stock_id", "date"])
        .groupby("stock_id", group_keys=False)
        .tail(lookback_days)
        .reset_index(drop=True)
    )
    return df


def load_latest_indicators(stock_ids: list[str]) -> pd.DataFrame:
    """
    撈每檔股票最新一筆技術指標，並在 Python 端補算 macd_hist（資料庫沒有這欄）。

    ⚠️ 2026/09 修正記錄：DISTINCT ON + ORDER BY date DESC 這種寫法，效能
    高度依賴資料庫有沒有 (stock_id, date) 的索引——沒有索引時，PostgreSQL
    必須先掃描、排序整張表才能篩出每檔股票最新一筆，這張表隨每天自動
    寫入持續增長，查詢會跟著越來越慢（觀察到實際案例：QueryCanceled）。
    主要修正是資料庫加索引（CREATE INDEX ON mart.technical_indicators
    (stock_id, date DESC)），這裡額外加上日期篩選當第二道防線——即使
    索引不存在或因故失效，查詢範圍也有上限，不會隨資料庫增長而擴大。
    30天緩衝遠超過「找最新一筆」實際需要的範圍（正常情況下每天都有新
    資料，1天內就找得到），純粹是保險，避免偶發性缺資料的股票被誤篩掉。
    """
    c = COLUMNS["technical_indicators"]
    sql = f"""
        select distinct on ({c['stock_id']})
            {c['stock_id']} as stock_id, {c['date']} as date,
            {c['ma5']} as ma5, {c['ma10']} as ma10, {c['ma20']} as ma20, {c['ma60']} as ma60,
            {c['rsi14']} as rsi14, {c['macd']} as macd,
            {c['macd_signal']} as macd_signal,
            {c['kd_k']} as kd_k, {c['kd_d']} as kd_d, {c['atr14']} as atr14
        from {c['table']}
        where {c['stock_id']} = any(:ids)
          and {c['date']} >= (select max({c['date']}) from {c['table']}) - INTERVAL '30 days'
        order by {c['stock_id']}, {c['date']} desc
    """
    df = _query_df(sql, {"ids": stock_ids})
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def load_ma10_history(stock_ids: list[str], lookback_days: int = 60) -> pd.DataFrame:
    """
    撈近 N 個交易日的 MA10 歷史序列（不是只有最新一筆），供計算「連續站上
    MA10 幾天」使用——判斷動能新鮮度，需要看過去一段期間的均線走勢，
    load_latest_indicators() 只回傳最新一筆，滿足不了這個需求。

    ⚠️ 2026/09 修正記錄：同 load_price_history()，加上日期範圍篩選，
    避免隨資料庫歷史資料增加、查詢時間跟著無限增長。
    """
    c = COLUMNS["technical_indicators"]
    calendar_days_buffer = int(lookback_days * 2.2) + 30
    sql = f"""
        select {c['stock_id']} as stock_id, {c['date']} as date, {c['ma10']} as ma10
        from {c['table']}
        where {c['stock_id']} = any(:ids)
          and {c['date']} >= (select max({c['date']}) from {c['table']}) - make_interval(days => :buffer_days)
        order by {c['stock_id']}, {c['date']}
    """
    df = _query_df(sql, {"ids": stock_ids, "buffer_days": calendar_days_buffer})
    df = (
        df.sort_values(["stock_id", "date"])
        .groupby("stock_id", group_keys=False)
        .tail(lookback_days)
        .reset_index(drop=True)
    )
    return df


def load_stock_info(stock_ids: list[str]) -> pd.DataFrame:
    c = COLUMNS["stock_info"]
    sql = f"""
        select {c['stock_id']} as stock_id, {c['name_zh']} as name_zh,
               {c['name_en']} as name_en,
               {c['sector_zh']} as sector_zh
        from {c['table']}
        where {c['stock_id']} = any(:ids)
    """
    return _query_df(sql, {"ids": stock_ids})


def load_eps_quarterly(stock_ids: list[str]) -> pd.DataFrame:
    """撈取觀察池股票的季度累計財報資料（EPS、毛利、營業成本、淨利、股東權益），
    供估值(P/E或P/B)與毛利率趨勢判斷使用。"""
    sql = """
        select stock_id, date, eps_cumulative, gross_profit, cost_of_goods_sold,
               net_income, equity
        from raw.eps_quarterly
        where stock_id = any(:ids)
        order by stock_id, date
    """
    df = _query_df(sql, {"ids": stock_ids})
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_latest_data_date():
    """
    查詢 staging.daily_master 裡實際最新一筆資料的日期，供頁面顯示
    「資料更新至」使用。⚠️ 這裡刻意跟 pages/1_Individual_Stock.py 的
    load_latest_data_date() 查同一張表、用同一種邏輯（MAX(date)），
    確保各頁面顯示的日期一致，不會出現「主頁顯示今天、其他頁面顯示
    昨天」這種誤導使用者的落差——這正是之前修正過的問題：主頁原本
    用 pd.Timestamp.today()（今天的日曆日期）硬顯示，沒有真的查詢
    資料庫，跟資料實際新舊程度脫節。
    """
    c = COLUMNS["daily_master"]
    sql = f"select max({c['date']}) as latest_date from {c['table']}"
    with get_conn() as conn:
        result = conn.execute(text(sql)).scalar()
    return result
