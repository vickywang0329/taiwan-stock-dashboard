"""
批次抓取觀察池股票的股價、三大法人、融資融券資料，寫入 PostgreSQL 的 raw schema
採「增量更新」邏輯：每檔股票自動從資料庫裡「目前已有的最後一天」的隔天開始抓，
抓到今天為止；資料庫裡還沒有的新股票，則從 DEFAULT_START_DATE 開始做完整回補。
"""

from FinMind.data import DataLoader
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd
import time
from datetime import date, timedelta

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

# ---- FinMind token（選填，填了請求上限會從 300/hr 提升到 600/hr）----
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

# 全新股票（資料庫裡還沒有任何資料）第一次回補的起始日
DEFAULT_START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()  # 每次執行都自動抓到今天


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
    """分別查股價、三大法人、融資融券三張表，各自的資料存到哪一天。
    取三者中「最早」的下一天當作這次抓取的起始日——
    這樣即使某張表曾經因故漏抓（例如上次執行中途失敗），也會被自動補上，
    不會因為只看股價表就誤判「這檔股票已經是最新」。
    資料庫裡完全沒有這檔股票時，回傳 DEFAULT_START_DATE 做完整回補。"""
    tables = ["stock_price", "institutional_investors", "margin_short_sale"]
    last_dates = []
    with engine.connect() as conn:
        for tbl in tables:
            result = conn.execute(
                text(f"SELECT MAX(date) FROM raw.{tbl} WHERE stock_id = :sid"),
                {"sid": stock_id},
            ).scalar()
            last_dates.append(result)

    if any(d is None for d in last_dates):
        # 只要有任何一張表完全沒有這檔股票的資料，就視為全新股票，完整回補
        return DEFAULT_START_DATE

    earliest = min(last_dates)
    return (earliest + timedelta(days=1)).isoformat()


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
    """只刪除本次要重新抓取的區間（start_date 之後），保留更早的歷史資料，
    再把新抓到的資料附加進去——這樣重複執行也不會出錯（idempotent）。"""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM raw.stock_price WHERE stock_id = :sid AND date >= :start"),
            {"sid": stock_id, "start": start_date},
        )
        conn.execute(
            text("DELETE FROM raw.institutional_investors WHERE stock_id = :sid AND date >= :start"),
            {"sid": stock_id, "start": start_date},
        )
        conn.execute(
            text("DELETE FROM raw.margin_short_sale WHERE stock_id = :sid AND date >= :start"),
            {"sid": stock_id, "start": start_date},
        )

    clean_price(price_df).to_sql("stock_price", engine, schema="raw", if_exists="append", index=False)
    inst_df.to_sql("institutional_investors", engine, schema="raw", if_exists="append", index=False)
    clean_margin(margin_df).to_sql("margin_short_sale", engine, schema="raw", if_exists="append", index=False)


def main():
    engine = get_engine()
    dl = DataLoader()
    if FINMIND_TOKEN:
        dl.login_by_token(api_token=FINMIND_TOKEN)

    total = len(WATCHLIST)
    for i, stock_id in enumerate(WATCHLIST, start=1):
        try:
            start_date = get_start_date(engine, stock_id)
            if start_date > END_DATE:
                print(f"[{i}/{total}] {stock_id} 已是最新，跳過")
                continue

            price_df, inst_df, margin_df = fetch_one_stock(dl, stock_id, start_date)
            if price_df.empty:
                print(f"[{i}/{total}] {stock_id} 沒有新資料（{start_date} 之後尚無交易日），跳過")
                continue

            write_raw(engine, stock_id, start_date, price_df, inst_df, margin_df)
            print(f"[{i}/{total}] {stock_id} 新增 {len(price_df)} 筆（{start_date} ~ {END_DATE}）")
        except Exception as e:
            print(f"[{i}/{total}] {stock_id} 失敗：{_safe_error(e)}")
        time.sleep(0.3)  # 避免過快觸發 API 限制

    print("\n全部股票更新完成！")


if __name__ == "__main__":
    main()
