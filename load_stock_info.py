"""
把公司中英文名稱 + 中英文產業分類清單匯入 raw.stock_info
（改用自訂分類取代 FinMind 原始分類，準確度更高、且支援雙語顯示）

執行前請先在 pgAdmin 的 Query Tool 執行一次以下 SQL：

    CREATE TABLE IF NOT EXISTS raw.stock_info (
        stock_id TEXT PRIMARY KEY,
        name_zh TEXT,
        name_en TEXT,
        industry TEXT,
        industry_en TEXT,
        loaded_at TIMESTAMP DEFAULT now()
    );
    ALTER TABLE raw.stock_info ADD COLUMN IF NOT EXISTS industry_en TEXT;

執行方式：python load_stock_info.py taiwan_stocks_categorized_2.csv
"""

import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd

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


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "taiwan_stocks_categorized_2.csv"
    df = pd.read_csv(csv_path, dtype={"股票代號": str}, encoding="utf-8-sig")
    df = df.rename(columns={
        "股票代號": "stock_id",
        "中文名稱": "name_zh",
        "官方英文名稱": "name_en",
        "產業分類": "industry",
        "產業分類(英文)": "industry_en",
    })
    df = df[["stock_id", "name_zh", "name_en", "industry", "industry_en"]]

    engine = get_engine()

    # 確保欄位存在（若之前的表沒有 industry_en，這裡順手補上）
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE raw.stock_info ADD COLUMN IF NOT EXISTS industry_en TEXT"))

    with engine.begin() as conn:
        # 每次重新匯入前先清空，避免舊資料殘留或重複
        conn.execute(text("DELETE FROM raw.stock_info"))

    df.to_sql("stock_info", engine, schema="raw", if_exists="append", index=False)
    print(f"已匯入 {len(df)} 筆公司名稱與產業分類資料到 raw.stock_info")


if __name__ == "__main__":
    main()
