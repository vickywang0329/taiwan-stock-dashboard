"""
抓取觀察池股票近 2.5 年的季度累計財報數字，寫入 raw.eps_quarterly：
- eps_cumulative：季度累計EPS（台股慣例，Q2=上半年累計，Q4=全年累計），供本益比估值使用
- gross_profit / cost_of_goods_sold：季度累計毛利／營業成本，供毛利率趨勢判斷使用
  （這三項來自同一次 FinMind API 呼叫，不需要額外多打 API，運算成本幾乎不變）

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


def fetch_financials(dl: DataLoader, stock_id: str) -> pd.DataFrame:
    """
    一次呼叫拿到整份財報，篩出 EPS、毛利、營業成本三項（同一次API，不多打），
    從長格式(date,stock_id,type,value) pivot 成寬格式，一列代表一季。
    """
    df = dl.taiwan_stock_financial_statement(
        stock_id=stock_id, start_date=START_DATE, end_date=END_DATE
    )
    if df.empty:
        return df

    df = df[df["type"].isin(["EPS", "GrossProfit", "CostOfGoodsSold"])]
    if df.empty:
        return pd.DataFrame()

    pivoted = df.pivot_table(
        index=["date", "stock_id"], columns="type", values="value", aggfunc="first"
    ).reset_index()
    pivoted = pivoted.rename(columns={
        "EPS": "eps_cumulative",
        "GrossProfit": "gross_profit",
        "CostOfGoodsSold": "cost_of_goods_sold",
    })

    # 有些公司可能缺特定項目（例如金融股沒有毛利概念），缺的欄位補 None，
    # 確保寫進資料庫的欄位齊全，不會因為某公司缺某項而整支腳本出錯
    for col in ["eps_cumulative", "gross_profit", "cost_of_goods_sold"]:
        if col not in pivoted.columns:
            pivoted[col] = None

    return pivoted[["date", "stock_id", "eps_cumulative", "gross_profit", "cost_of_goods_sold"]]


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
            df = fetch_financials(dl, stock_id)
            if df.empty:
                print(f"[{i}/{total}] {stock_id} 沒有財報資料（可能是ETF），跳過")
                continue
            write_raw(engine, stock_id, df)
            print(f"[{i}/{total}] {stock_id} 寫入 {len(df)} 筆季度財報資料")
        except Exception as e:
            print(f"[{i}/{total}] {stock_id} 失敗：{_safe_error(e)}")
        time.sleep(0.3)

    print("\n全部股票財報資料更新完成！")


if __name__ == "__main__":
    main()
