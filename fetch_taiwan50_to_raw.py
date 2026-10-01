"""
批次抓取觀察池股票的股價、三大法人、融資融券資料，寫入 PostgreSQL 的 raw schema

增量更新邏輯：
- 以「股價表」裡這檔股票最後一天為基準，往前回溯 LOOKBACK_DAYS 天開始重抓，
  抓到今天為止。回溯的目的是讓三大法人、融資融券這類較晚公布或事後修正的資料，
  在之後幾天的執行中自動補齊。
- 資料庫裡還沒有這檔股票的股價資料時，從 DEFAULT_START_DATE 開始完整回補。
- 不再要求三張表都要有資料：有些標的（例如槓桿／反向 ETF）本來就沒有融資融券
  或法人資料，舊版會把它們誤判成「新股票」，導致每天都從頭全量重抓。
"""

from FinMind.data import DataLoader
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd
import time
from datetime import date, timedelta

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

# ---- FinMind token（選填，填了請求上限會從 300/hr 提升到 600/hr）----
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

# 全新股票（資料庫裡還沒有任何股價資料）第一次回補的起始日
DEFAULT_START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()  # 每次執行都自動抓到今天

# 從股價最後一天往前回溯幾天重抓，涵蓋法人／融資融券的延遲公布與修正
LOOKBACK_DAYS = 7

RAW_TABLES = ["stock_price", "institutional_investors", "margin_short_sale"]


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


def get_start_date(engine, stock_id: str) -> str:
    """以股價表的最後一天為基準，往前回溯 LOOKBACK_DAYS 天作為起始日。
    股價表完全沒有這檔股票時，回傳 DEFAULT_START_DATE 做完整回補。"""
    with engine.connect() as conn:
        last_date = conn.execute(
            text("SELECT MAX(date) FROM raw.stock_price WHERE stock_id = :sid"),
            {"sid": stock_id},
        ).scalar()

    if last_date is None:
        return DEFAULT_START_DATE

    start = last_date - timedelta(days=LOOKBACK_DAYS)
    return max(start.isoformat(), DEFAULT_START_DATE)


def fetch_one_stock(dl: DataLoader, stock_id: str, start_date: str):
    price_df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=END_DATE)
    inst_df = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date, end_date=END_DATE)
    margin_df = dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_date, end_date=END_DATE)
    return price_df, inst_df, margin_df


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Trading_Volume": "trading_volume",
        "Trading_money": "trading_money",
        "Trading_turnover": "trading_turnover",
    })
    cols = ["date", "stock_id", "open", "max", "min", "close",
            "trading_volume", "trading_money", "spread", "trading_turnover"]
    return df[cols]


def clean_margin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "MarginPurchaseTodayBalance": "margin_purchase_today_balance",
        "MarginPurchaseYesterdayBalance": "margin_purchase_yesterday_balance",
        "ShortSaleTodayBalance": "short_sale_today_balance",
        "ShortSaleYesterdayBalance": "short_sale_yesterday_balance",
    })
    cols = ["date", "stock_id", "margin_purchase_today_balance",
            "margin_purchase_yesterday_balance", "short_sale_today_balance",
            "short_sale_yesterday_balance"]
    return df[cols]


def write_raw(engine, stock_id: str, start_date: str, price_df, inst_df, margin_df):
    """刪除本次重抓的區間（start_date 之後），再寫入新資料。
    刪除與寫入放在同一個 transaction：任何一步失敗都會整個 rollback，
    不會出現「舊資料已刪、新資料沒寫進去」的情況。"""
    # FinMind 查無資料時回傳的是沒有欄位的空表，先處理掉，避免 clean 函式 KeyError
    frames = {
        "stock_price": clean_price(price_df) if not price_df.empty else None,
        "institutional_investors": inst_df if not inst_df.empty else None,
        "margin_short_sale": clean_margin(margin_df) if not margin_df.empty else None,
    }

    with engine.begin() as conn:
        for tbl in RAW_TABLES:
            conn.execute(
                text(f"DELETE FROM raw.{tbl} WHERE stock_id = :sid AND date >= :start"),
                {"sid": stock_id, "start": start_date},
            )
        for tbl, df in frames.items():
            if df is not None:
                # method="multi" + chunksize：多筆合併成一條 INSERT 批次寫入
                df.to_sql(tbl, conn, schema="raw", if_exists="append",
                          index=False, method="multi", chunksize=1000)

    return {tbl: (0 if df is None else len(df)) for tbl, df in frames.items()}


def main():
    engine = get_engine()
    dl = DataLoader()
    if FINMIND_TOKEN:
        dl.login_by_token(api_token=FINMIND_TOKEN)
    else:
        print("提醒：未設定 FINMIND_TOKEN，FinMind 請求上限為 300 次／小時")

    total = len(WATCHLIST)
    failed = []
    for i, stock_id in enumerate(WATCHLIST, start=1):
        try:
            start_date = get_start_date(engine, stock_id)
            price_df, inst_df, margin_df = fetch_one_stock(dl, stock_id, start_date)
            if price_df.empty:
                print(f"[{i}/{total}] {stock_id} 沒有新資料（{start_date} 之後尚無交易日），跳過")
                continue

            counts = write_raw(engine, stock_id, start_date, price_df, inst_df, margin_df)
            print(f"[{i}/{total}] {stock_id} 寫入 {start_date} ~ {END_DATE}："
                  f"股價 {counts['stock_price']}、法人 {counts['institutional_investors']}、"
                  f"融資融券 {counts['margin_short_sale']} 筆")
        except Exception as e:
            failed.append(stock_id)
            print(f"[{i}/{total}] {stock_id} 失敗：{_safe_error(e)}")
        time.sleep(0.3)  # 避免過快觸發 API 限制

    print(f"\n完成：{total - len(failed)}/{total} 檔成功")
    if failed:
        print(f"失敗清單：{', '.join(failed)}")
        sys.exit(1)  # 讓 GitHub Actions 這一步標示為失敗，問題才不會被忽略


if __name__ == "__main__":
    main()
