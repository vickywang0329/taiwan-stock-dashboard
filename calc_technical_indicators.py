"""
mart layer 技術指標計算腳本
從 staging.daily_master 讀取資料，計算均線、RSI、MACD、KD，寫入 mart.technical_indicators
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd
import numpy as np

from watchlist import WATCHLIST

# ---- 資料庫連線設定，請依你的實際狀況修改 ----
# ---- 資料庫連線設定：從環境變數讀取，不寫死在程式碼裡 ----
# 本機測試時，把 .env.example 複製成 .env 並填入真實值（.env 已在 .gitignore 中，不會被上傳）
load_dotenv()

DB_CONFIG = {
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ.get("DB_PORT", "5432"),
    "database": os.environ.get("DB_NAME", "postgres"),
}


def get_engine():
    safe_password = quote_plus(DB_CONFIG["password"])
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{safe_password}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def calc_kd(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 9, k_smooth: int = 3, d_smooth: int = 3):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    # KD 用類似 RSV 的平滑移動平均逼近（實務上常用 SMA 近似）
    k = rsv.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ATR（Average True Range，平均真實區間）
    True Range 取三者最大值：當日高低差 / |當日高-昨收| / |當日低-昨收|
    再取近 period 天的簡單移動平均
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def transform(engine, stock_id: str) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT date, stock_id, high, low, close FROM staging.daily_master "
             "WHERE stock_id = :sid ORDER BY date"),
        engine, params={"sid": stock_id},
    )

    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()
    df["rsi_14"] = calc_rsi(df["close"], period=14)
    df["macd"], df["macd_signal"] = calc_macd(df["close"])
    df["kd_k"], df["kd_d"] = calc_kd(df["high"], df["low"], df["close"])
    df["atr14"] = calc_atr(df["high"], df["low"], df["close"], period=14)

    result = df[["date", "stock_id", "ma5", "ma20", "ma60", "rsi_14",
                 "macd", "macd_signal", "kd_k", "kd_d", "atr14"]]
    # 前面天數不足以算出指標的列會是 NaN，轉成 None 讓資料庫存成 NULL
    result = result.replace({np.nan: None})
    return result


def load_to_mart(engine, df: pd.DataFrame, stock_id: str):
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM mart.technical_indicators WHERE stock_id = :sid"),
            {"sid": stock_id},
        )
    df.to_sql("technical_indicators", engine, schema="mart", if_exists="append", index=False)


def main():
    engine = get_engine()
    total = len(WATCHLIST)
    for i, stock_id in enumerate(WATCHLIST, start=1):
        try:
            result = transform(engine, stock_id)
            if result.empty:
                print(f"[{i}/{total}] {stock_id} staging layer 沒有資料，跳過")
                continue
            load_to_mart(engine, result, stock_id)
            print(f"[{i}/{total}] {stock_id} 技術指標計算完成，共 {len(result)} 筆")
        except Exception as e:
            print(f"[{i}/{total}] {stock_id} 失敗：{e}")

    print("\n全部股票技術指標計算完成！")


if __name__ == "__main__":
    main()
