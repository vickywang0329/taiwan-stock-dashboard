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


def load_price_history(stock_ids: list[str], lookback_days: int = 90) -> pd.DataFrame:
    """撈近 N 個交易日的價格 + 三大法人淨額（用於算分數與技術指標）。"""
    c = COLUMNS["daily_master"]
    sql = f"""
        select {c['stock_id']} as stock_id, {c['date']} as date,
               {c['close']} as close, {c['high']} as high, {c['low']} as low,
               {c['volume']} as volume,
               {c['institutional_net']} as institutional_net
        from {c['table']}
        where {c['stock_id']} = any(:ids)
        order by {c['stock_id']}, {c['date']}
    """
    with get_conn() as conn:
        df = pd.read_sql(text(sql), conn, params={"ids": stock_ids})
    # 只保留每檔股票最近 lookback_days 筆
    df = (
        df.sort_values(["stock_id", "date"])
        .groupby("stock_id", group_keys=False)
        .tail(lookback_days)
        .reset_index(drop=True)
    )
    return df


def load_latest_indicators(stock_ids: list[str]) -> pd.DataFrame:
    """撈每檔股票最新一筆技術指標，並在 Python 端補算 macd_hist（資料庫沒有這欄）。"""
    c = COLUMNS["technical_indicators"]
    sql = f"""
        select distinct on ({c['stock_id']})
            {c['stock_id']} as stock_id, {c['date']} as date,
            {c['ma5']} as ma5, {c['ma20']} as ma20, {c['ma60']} as ma60,
            {c['rsi14']} as rsi14, {c['macd']} as macd,
            {c['macd_signal']} as macd_signal,
            {c['kd_k']} as kd_k, {c['kd_d']} as kd_d, {c['atr14']} as atr14
        from {c['table']}
        where {c['stock_id']} = any(:ids)
        order by {c['stock_id']}, {c['date']} desc
    """
    with get_conn() as conn:
        df = pd.read_sql(text(sql), conn, params={"ids": stock_ids})
    df["macd_hist"] = df["macd"] - df["macd_signal"]
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
    with get_conn() as conn:
        return pd.read_sql(text(sql), conn, params={"ids": stock_ids})
