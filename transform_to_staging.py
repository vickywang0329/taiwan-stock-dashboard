"""
staging layer 轉換腳本（SQL 版本）
從 raw schema 讀取三張原始表，清洗轉換後寫入 staging.daily_master。

整段轉換用一條 INSERT ... SELECT 在資料庫內完成：
資料不需要傳到執行這支程式的機器（例如 GitHub Actions）上，
也不需要逐支股票來回查詢，通常幾秒鐘就能跑完。

增量更新：預設只處理最近 LOOKBACK_DAYS 天的資料。
需要全量重建時（例如第一次執行、補資料、watchlist 新增股票），
設定環境變數 FULL_REFRESH=1。
"""

import os
import time
from datetime import date, timedelta

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

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

# 自營商相關的法人類別，加總成 dealer_net
DEALER_NAMES = ("Dealer_self", "Dealer_Hedging", "Foreign_Dealer_Self")


DELETE_SQL = text("""
    DELETE FROM staging.daily_master
    WHERE stock_id = ANY(:ids)
      AND date >= :since
""")

# 三張 raw 表在資料庫內直接轉換、合併，寫入 staging.daily_master
#   - 股價：欄位改名（max→high、min→low 等）
#   - 三大法人：原本 pandas 的 pivot_table（long → wide），
#     改用 SUM(...) FILTER (WHERE name = ...) 的條件式聚合
#   - 融資融券：今日餘額 - 昨日餘額 = 每日增減
INSERT_SQL = text("""
    WITH inst AS (
        SELECT
            date,
            stock_id,
            COALESCE(SUM(buy - sell) FILTER (WHERE name = 'Foreign_Investor'), 0)
                AS foreign_net,
            COALESCE(SUM(buy - sell) FILTER (WHERE name = 'Investment_Trust'), 0)
                AS investment_trust_net,
            COALESCE(SUM(buy - sell) FILTER (WHERE name = ANY(:dealer_names)), 0)
                AS dealer_net
        FROM raw.institutional_investors
        WHERE stock_id = ANY(:ids)
          AND date >= :since
        GROUP BY date, stock_id
    )
    INSERT INTO staging.daily_master (
        date, stock_id, open, high, low, close, volume, trading_value,
        foreign_net, investment_trust_net, dealer_net, institutional_total_net,
        margin_balance, margin_balance_change, short_balance, short_balance_change
    )
    SELECT
        p.date,
        p.stock_id,
        p.open,
        p.max                AS high,
        p.min                AS low,
        p.close,
        p.trading_volume     AS volume,
        p.trading_money      AS trading_value,
        i.foreign_net,
        i.investment_trust_net,
        i.dealer_net,
        i.foreign_net + i.investment_trust_net + i.dealer_net
                             AS institutional_total_net,
        m.margin_purchase_today_balance
                             AS margin_balance,
        m.margin_purchase_today_balance - m.margin_purchase_yesterday_balance
                             AS margin_balance_change,
        m.short_sale_today_balance
                             AS short_balance,
        m.short_sale_today_balance - m.short_sale_yesterday_balance
                             AS short_balance_change
    FROM raw.stock_price p
    LEFT JOIN inst i
        ON i.date = p.date AND i.stock_id = p.stock_id
    LEFT JOIN raw.margin_short_sale m
        ON m.date = p.date AND m.stock_id = p.stock_id
    WHERE p.stock_id = ANY(:ids)
      AND p.date >= :since
""")


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


def main():
    engine = get_engine()
    since = get_since_date()
    params = {
        "ids": list(WATCHLIST),
        "since": since,
        "dealer_names": list(DEALER_NAMES),
    }

    start = time.time()
    try:
        # DELETE 與 INSERT 在同一個 transaction：
        # 任一步失敗會整個 rollback，staging 不會出現資料缺口
        with engine.begin() as conn:
            deleted = conn.execute(DELETE_SQL, params).rowcount
            inserted = conn.execute(INSERT_SQL, params).rowcount
    except Exception as e:
        print(f"staging 轉換失敗：{_safe_error(e)}")
        raise SystemExit(1)  # 讓 GitHub Actions 這一步標示為失敗

    elapsed = time.time() - start
    print(f"刪除舊資料 {deleted} 筆，寫入新資料 {inserted} 筆"
          f"（{len(WATCHLIST)} 檔股票，耗時 {elapsed:.1f} 秒）")
    print("\n全部股票 staging 轉換完成！")


if __name__ == "__main__":
    main()
