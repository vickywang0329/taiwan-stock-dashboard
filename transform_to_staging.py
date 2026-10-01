"""
staging layer 轉換腳本
從 raw schema 讀取三張原始表，清洗轉換後寫入 staging.daily_master
（不再讀本地 CSV，改成直接對資料庫讀寫）

增量更新：預設只處理最近 LOOKBACK_DAYS 天的資料，
避免每天把全部歷史資料刪掉重寫，大幅降低資料庫 Disk IO。
需要全量重建時（例如第一次執行、補資料），設定環境變數 FULL_REFRESH=1。
"""

import os
from datetime import date, timedelta

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd

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

# 每次回溯處理的天數。保留一些緩衝，讓 FinMind 延遲更新、資料修正，
# 或某天 Action 失敗時，隔天都能自動補上。
LOOKBACK_DAYS = 14


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


def get_since_date() -> date:
    """決定這次要處理的起始日期：全量重建或只回溯最近幾天。"""
    if os.environ.get("FULL_REFRESH") == "1":
        print("FULL_REFRESH=1：全量重建所有歷史資料")
        return date(2000, 1, 1)
    since = date.today() - timedelta(days=LOOKBACK_DAYS)
    print(f"增量更新：處理 {since} 之後的資料")
    return since


def transform(engine, stock_id: str, since: date) -> pd.DataFrame:
    params = {"sid": stock_id, "since": since}

    # ---- 1. 從 raw 讀股價 ----
    price_df = pd.read_sql(
        text("SELECT * FROM raw.stock_price "
             "WHERE stock_id = :sid AND date >= :since"),
        engine, params=params,
    )
    price_df = price_df.rename(columns={
        "max": "high",
        "min": "low",
        "trading_volume": "volume",
        "trading_money": "trading_value",
    })
    price_df = price_df[["date", "stock_id", "open", "high", "low", "close",
                          "volume", "trading_value"]]

    # ---- 2. 從 raw 讀三大法人，long → wide 並計算淨買賣超 ----
    inst_df = pd.read_sql(
        text("SELECT * FROM raw.institutional_investors "
             "WHERE stock_id = :sid AND date >= :since"),
        engine, params=params,
    )
    if inst_df.empty:
        # ETF 等標的可能沒有三大法人資料，補一個空表避免後面出錯
        inst_clean = pd.DataFrame(columns=[
            "date", "foreign_net", "investment_trust_net", "dealer_net", "institutional_total_net"
        ])
    else:
        inst_df["net"] = inst_df["buy"] - inst_df["sell"]
        inst_wide = inst_df.pivot_table(
            index="date", columns="name", values="net", aggfunc="sum"
        ).reset_index()

        dealer_cols = [c for c in ["Dealer_self", "Dealer_Hedging", "Foreign_Dealer_Self"]
                       if c in inst_wide.columns]
        inst_wide["dealer_net"] = inst_wide[dealer_cols].sum(axis=1) if dealer_cols else 0

        for col in ["Foreign_Investor", "Investment_Trust"]:
            if col not in inst_wide.columns:
                inst_wide[col] = 0

        inst_clean = inst_wide[["date", "Foreign_Investor", "Investment_Trust", "dealer_net"]].rename(
            columns={"Foreign_Investor": "foreign_net", "Investment_Trust": "investment_trust_net"}
        )
        inst_clean["institutional_total_net"] = (
            inst_clean["foreign_net"] + inst_clean["investment_trust_net"] + inst_clean["dealer_net"]
        )

    # ---- 3. 從 raw 讀融資融券，計算每日增減 ----
    margin_df = pd.read_sql(
        text("SELECT * FROM raw.margin_short_sale "
             "WHERE stock_id = :sid AND date >= :since"),
        engine, params=params,
    )
    if margin_df.empty:
        margin_clean = pd.DataFrame(columns=[
            "date", "margin_balance", "margin_balance_change", "short_balance", "short_balance_change"
        ])
    else:
        margin_df["margin_balance_change"] = (
            margin_df["margin_purchase_today_balance"] - margin_df["margin_purchase_yesterday_balance"]
        )
        margin_df["short_balance_change"] = (
            margin_df["short_sale_today_balance"] - margin_df["short_sale_yesterday_balance"]
        )
        margin_clean = margin_df[[
            "date", "margin_purchase_today_balance", "margin_balance_change",
            "short_sale_today_balance", "short_balance_change",
        ]].rename(columns={
            "margin_purchase_today_balance": "margin_balance",
            "short_sale_today_balance": "short_balance",
        })

    # ---- 4. 合併 ----
    master = price_df.merge(inst_clean, on="date", how="left")
    master = master.merge(margin_clean, on="date", how="left")
    master = master.sort_values("date").reset_index(drop=True)
    return master


def load_to_staging(engine, df: pd.DataFrame, stock_id: str, since: date):
    # 刪除與寫入放在同一個 transaction：
    # 寫入途中失敗會整個 rollback，不會出現「舊資料已刪、新資料沒寫進去」的情況
    with engine.begin() as conn:
        # 只刪除這次要重寫的日期範圍，不動更早的歷史資料
        conn.execute(
            text("DELETE FROM staging.daily_master "
                 "WHERE stock_id = :sid AND date >= :since"),
            {"sid": stock_id, "since": since},
        )
        # method="multi" + chunksize：多筆合併成一條 INSERT 批次寫入，
        # 減少 GitHub Actions（美國）與資料庫（東京）之間的來回次數
        df.to_sql(
            "daily_master", conn, schema="staging",
            if_exists="append", index=False,
            method="multi", chunksize=1000,
        )


def main():
    engine = get_engine()
    since = get_since_date()
    total = len(WATCHLIST)
    for i, stock_id in enumerate(WATCHLIST, start=1):
        try:
            master = transform(engine, stock_id, since)
            if master.empty:
                print(f"[{i}/{total}] {stock_id} raw layer 沒有資料，跳過")
                continue
            load_to_staging(engine, master, stock_id, since)
            print(f"[{i}/{total}] {stock_id} 轉換完成並寫入，共 {len(master)} 筆")
        except Exception as e:
            print(f"[{i}/{total}] {stock_id} 失敗：{_safe_error(e)}")

    print("\n全部股票 staging 轉換完成！")


if __name__ == "__main__":
    main()
