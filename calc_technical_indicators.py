"""
mart layer 技術指標計算腳本
從 staging.daily_master 讀取資料，計算均線、RSI、MACD、KD，寫入 mart.technical_indicators

⚠️ 2026/09 效能優化記錄：原本每天對每檔股票「整段歷史全刪全寫」，隨著
資料庫累積天數增加，每晚的讀取量（SELECT 全部歷史）跟寫入量（DELETE+
INSERT 全部歷史）都會跟著無限增長——跟 db.py／sector_flow.py 那次修正
發現的同一種問題，只是這次出現在「寫入」這一側，不是使用者互動時的
「讀取」這一側，急迫性較低，但同樣值得趁早修正，不要等它變慢到造成
問題才處理。

改成增量式處理，在不改變計算結果（跟原本全量重算的數字完全一致）的
前提下大幅降低每次執行的讀寫量：

- 只有「這檔股票在 mart.technical_indicators 裡完全沒有資料」（新加入
  觀察池、或第一次執行整套系統）時，才做一次完整的全歷史計算——這是
  無法避免的一次性成本，之後同一檔股票就不會再發生。
- 已經有資料的股票，只抓「最近 RECOMPUTE_LOOKBACK_DAYS 個交易日」的
  原始價格（不是全部歷史）用來重新計算指標。這個緩衝天數（250個交易日，
  約一年）遠超過 MA60 需要的60天，也足夠讓 MACD 用到的 EMA 充分收斂，
  確保在這個窗口內算出來的指標數值，跟拿全部歷史去算的結果完全一致。
- 只刪除／補寫「最近 OVERWRITE_RECENT_DAYS 天 + 任何新增的交易日」這一
  小段，不去動更早以前、理論上不會再變動的歷史資料。

  OVERWRITE_RECENT_DAYS 這個緩衝的用意：防範資料來源（FinMind）事後對
  「已公布過」的近期資料做小幅修正（例如更新最終確認的成交量），讓這種
  修正也能被涵蓋到，不會因為改成增量式而漏掉。

⚠️ 取捨：如果資料來源對「比 OVERWRITE_RECENT_DAYS 天更早」的歷史資料做
回溯修正，增量模式不會自動偵測到、重新計算——這種情況理論上極少見
（股價成交資料公布後幾乎不會再事後修改，比較可能發生在財報數字，但
財報不是這支腳本處理的範圍）。如果真的需要保證絕對正確，可以加上
--full 參數強制對全部股票做一次完整重算：
    python calc_technical_indicators.py --full
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd
import numpy as np

from watchlist import WATCHLIST

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

# 台股一年約240個交易日，換算日曆天數約為交易天數的1.52倍（365÷240），
# 下面兩個常數都用這個比例換算查詢用的日曆天緩衝，跟 db.py 那次修正
# 用的是同一套換算邏輯，保持一致。
RECOMPUTE_LOOKBACK_DAYS = 250  # 交易日緩衝，足夠讓MA60、MACD的EMA算得準確
OVERWRITE_RECENT_DAYS = 10     # 即使已經算過，也重新覆蓋最近這幾天，防範資料來源事後修正


def _safe_error(e) -> str:
    """
    印出錯誤訊息前，先把密碼從字串裡過濾掉——避免資料庫連線失敗時，
    底層套件的錯誤訊息不小心把完整連線字串（含密碼）一起印出來，
    寫進 GitHub Actions 的執行日誌裡（尤其是 repo 設成 Public 之後，
    日誌任何人都能看）。
    """
    msg = str(e)
    password = DB_CONFIG.get("password")
    if password:
        msg = msg.replace(password, "***")
    return msg


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


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """給定一段（date, stock_id, high, low, close）序列，算出全部技術指標。
    不管是全量或增量呼叫，都共用這同一份計算邏輯，確保兩種模式算出來的
    數字完全一致，不會因為改成增量式而產生差異。"""
    df = df.sort_values("date").reset_index(drop=True)
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()
    df["rsi_14"] = calc_rsi(df["close"], period=14)
    df["macd"], df["macd_signal"] = calc_macd(df["close"])
    df["kd_k"], df["kd_d"] = calc_kd(df["high"], df["low"], df["close"])
    df["atr14"] = calc_atr(df["high"], df["low"], df["close"], period=14)

    result = df[["date", "stock_id", "ma5", "ma10", "ma20", "ma60", "rsi_14",
                 "macd", "macd_signal", "kd_k", "kd_d", "atr14"]].copy()
    return result


def get_last_computed_date(engine, stock_id: str):
    """查這檔股票在 mart.technical_indicators 裡，已經算過的最新日期。
    完全沒有資料（例如新加入觀察池的股票）回傳 None，代表要做完整計算。"""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(date) FROM mart.technical_indicators WHERE stock_id = :sid"),
            {"sid": stock_id},
        ).scalar()
    return result


def transform_full(engine, stock_id: str) -> pd.DataFrame:
    """全歷史計算——只在「這檔股票完全沒有算過」時使用一次。"""
    df = pd.read_sql(
        text("SELECT date, stock_id, high, low, close FROM staging.daily_master "
             "WHERE stock_id = :sid ORDER BY date"),
        engine, params={"sid": stock_id},
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    result = _compute_indicators(df)
    result = result.replace({np.nan: None})
    return result


def transform_incremental(engine, stock_id: str, last_computed_date) -> pd.DataFrame | None:
    """
    增量計算：只抓「足夠回推計算所需」的視窗，不是全部歷史。
    回傳的 DataFrame 只包含「需要重新寫入」的那一小段（最近
    OVERWRITE_RECENT_DAYS 天 + 任何比 last_computed_date 更新的交易日），
    不含更早、不需要重新寫入的歷史列（那些只是拿來當計算緩衝，算完就丟）。

    如果 staging 層完全沒有比上次算過的日期更新的資料（代表還沒有新的
    一天可算），回傳 None，呼叫端據此跳過寫入動作，不做任何無意義的
    刪除/重寫。
    """
    buffer_calendar_days = int(RECOMPUTE_LOOKBACK_DAYS * 1.52) + 30
    df = pd.read_sql(
        text("""
            SELECT date, stock_id, high, low, close FROM staging.daily_master
            WHERE stock_id = :sid
              AND date >= (:last_date)::date - make_interval(days => :buffer_days)
            ORDER BY date
        """),
        engine, params={
            "sid": stock_id, "last_date": last_computed_date,
            "buffer_days": buffer_calendar_days,
        },
    )
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])
    last_computed_ts = pd.Timestamp(last_computed_date)

    if df["date"].max() <= last_computed_ts:
        return None  # 沒有新的交易日可算，跳過，不做無意義的刪除/重寫

    result = _compute_indicators(df)

    # 只保留「需要覆蓋或新增」的那一段：OVERWRITE_RECENT_DAYS 天前
    # （以 last_computed_date 為準）之後的資料，更早的緩衝資料只是拿來
    # 讓 rolling window / EMA 算得準確，不需要真的寫回資料庫。
    write_from_calendar_days = int(OVERWRITE_RECENT_DAYS * 1.52) + 5
    write_from = last_computed_ts - pd.Timedelta(days=write_from_calendar_days)
    result = result[result["date"] >= write_from].reset_index(drop=True)
    result = result.replace({np.nan: None})
    return result


def load_to_mart(engine, df: pd.DataFrame, stock_id: str, from_date=None):
    """
    寫入 mart.technical_indicators。
    from_date 給定時：只刪除該股票「>= from_date」的既有資料，只補寫這
    一段（增量模式，刪除/寫入範圍固定，不隨累積歷史增加而變大）。
    from_date 為 None 時：刪除該股票的全部既有資料再整段寫入（全量模式，
    只在首次計算、或使用者主動要求 --full 強制重算時使用）。
    """
    with engine.begin() as conn:
        if from_date is not None:
            conn.execute(
                text("DELETE FROM mart.technical_indicators WHERE stock_id = :sid AND date >= :from_date"),
                {"sid": stock_id, "from_date": from_date},
            )
        else:
            conn.execute(
                text("DELETE FROM mart.technical_indicators WHERE stock_id = :sid"),
                {"sid": stock_id},
            )
    df.to_sql("technical_indicators", engine, schema="mart", if_exists="append", index=False)


def main():
    force_full = "--full" in sys.argv
    engine = get_engine()
    total = len(WATCHLIST)
    for i, stock_id in enumerate(WATCHLIST, start=1):
        try:
            last_date = None if force_full else get_last_computed_date(engine, stock_id)

            if last_date is None:
                result = transform_full(engine, stock_id)
                if result.empty:
                    print(f"[{i}/{total}] {stock_id} staging layer 沒有資料，跳過")
                    continue
                load_to_mart(engine, result, stock_id, from_date=None)
                mode = "強制全量重算" if force_full else "首次完整計算"
                print(f"[{i}/{total}] {stock_id} {mode}完成，共 {len(result)} 筆")
            else:
                result = transform_incremental(engine, stock_id, last_date)
                if result is None:
                    print(f"[{i}/{total}] {stock_id} 沒有新交易日，跳過")
                    continue
                load_to_mart(engine, result, stock_id, from_date=result["date"].min())
                print(f"[{i}/{total}] {stock_id} 增量計算完成，更新 {len(result)} 筆（更早的歷史資料保留不動）")
        except Exception as e:
            print(f"[{i}/{total}] {stock_id} 失敗：{_safe_error(e)}")

    print("\n全部股票技術指標計算完成！")
    if not force_full:
        print("提示：如需強制對所有股票做一次完整重算，執行 python calc_technical_indicators.py --full")


if __name__ == "__main__":
    main()
