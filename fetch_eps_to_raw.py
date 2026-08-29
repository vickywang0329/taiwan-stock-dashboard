"""
抓取觀察池股票近 2.5 年的季度累計 EPS（台股財報慣例：每季揭露的是累計數，
Q2 = 上半年累計，Q4 = 全年累計），寫入 raw.eps_quarterly，
供估值檢查（本益比）使用。

執行方式：python fetch_eps_to_raw.py
"""
from FinMind.data import DataLoader
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
import pandas as pd
import time
from datetime import date, timedelta

from watchlist import WATCHLIST

load_dotenv()

DB_CONFIG = {
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ.get("DB_PORT", "5432"),
    "database": os.environ.get("DB_NAME", "postgres"),
}
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

# 抓近 2.5 年的季報，足夠算「今年H1/去年H1/去年全年」跟 TTM 備援兩種算法
START_DATE = (date.today() - timedelta(days=int(365 * 2.5))).isoformat()
END_DATE = date.today().isoformat()


def get_engine():
    safe_password = quote_plus(DB_CONFIG["password"])
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{safe_password}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


def fetch_eps(dl: DataLoader, stock_id: str) -> pd.DataFrame:
    df = dl.taiwan_stock_financial_statement(
        stock_id=stock_id, start_date=START_DATE, end_date=END_DATE
    )
    if df.empty:
        return df
    df = df[df["type"] == "EPS"][["date", "stock_id", "value"]].rename(
        columns={"value": "eps_cumulative"}
    )
    return df


def write_raw(engine, stock_id: str, df: pd.DataFrame):
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM raw.eps_quarterly WHERE stock_id = :sid"),
            {"sid": stock_id},
        )
    df.to_sql("eps_quarterly", engine, schema="raw", if_exists="append", index=False)


def main():
    engine = get_engine()
    dl = DataLoader()
    if FINMIND_TOKEN:
        dl.login_by_token(api_token=FINMIND_TOKEN)

    total = len(WATCHLIST)
    for i, stock_id in enumerate(WATCHLIST, start=1):
        try:
            df = fetch_eps(dl, stock_id)
            if df.empty:
                print(f"[{i}/{total}] {stock_id} 沒有EPS資料（可能是ETF），跳過")
                continue
            write_raw(engine, stock_id, df)
            print(f"[{i}/{total}] {stock_id} 寫入 {len(df)} 筆季度EPS資料")
        except Exception as e:
            print(f"[{i}/{total}] {stock_id} 失敗：{e}")
        time.sleep(0.3)

    print("\n全部股票EPS資料更新完成！")


if __name__ == "__main__":
    main()
