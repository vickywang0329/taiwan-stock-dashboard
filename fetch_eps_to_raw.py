"""
抓取觀察池股票近 4 年的季度累計財報數字，寫入 raw.eps_quarterly：
- eps_cumulative：季度累計EPS（台股慣例，Q2=上半年累計，Q4=全年累計），供本益比估值使用
- gross_profit / cost_of_goods_sold：季度累計毛利／營業成本，供毛利率趨勢判斷使用
- net_income：歸屬母公司淨利（損益表）；equity：歸屬母公司權益（資產負債表），
  供景氣循環股的 P/B 估值使用（每檔股票 2 次 FinMind API 呼叫：損益表 + 資產負債表）

⚠️ 從 2.5 年延長到 4 年，是為了讓景氣循環股（航運/記憶體/塑化石化/鋼鐵）的
毛利率、波動度等計算，能涵蓋更長的景氣週期——但要老實說：4年仍然可能無法
涵蓋完整的一個循環週期（例如萬海航運真正的暴衝高點在2021Q4~2022Q2，
距今已超過4年），這是一個部分改善，不是完整解法。

執行方式：python fetch_eps_to_raw.py            # 全部更新
診斷單一股票：python fetch_eps_to_raw.py --check 2603
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

# 抓近 4 年的季報（原本2.5年，這次延長），供景氣循環股的計算涵蓋更長週期
START_DATE = (date.today() - timedelta(days=int(365 * 4))).isoformat()
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


INCOME_TYPES = ["EPS", "GrossProfit", "CostOfGoodsSold", "IncomeAfterTaxes",
                "EquityAttributableToOwnersOfParent"]
# 資產負債表的「歸屬母公司業主權益」；找不到時退而求其次用「權益總額」
BS_EQUITY_TYPES = ["EquityAttributableToOwnersOfParent", "Equity"]
ALL_COLS = ["eps_cumulative", "gross_profit", "cost_of_goods_sold", "net_income", "equity"]


def build_wide(income: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """
    把損益表（EPS、毛利、營業成本、淨利）與資產負債表（股東權益）合併成一季一列。

    ⚠️ 關鍵修正：FinMind 的 EquityAttributableToOwnersOfParent 這個名稱，
    在「損益表」裡是「歸屬母公司業主的本期淨利」，在「資產負債表」裡才是「歸屬母公司業主權益」，
    兩者同名但意義不同。舊版只呼叫損益表，所以 equity 欄位從來沒有真正的股東權益。
    這裡損益表的同名欄位改當作 net_income（歸屬母公司淨利，與 EPS 的計算基礎一致），
    equity 一律取自資產負債表。
    """
    if income is None or income.empty:
        return pd.DataFrame()

    inc = income[income["type"].isin(INCOME_TYPES)]
    if inc.empty:
        return pd.DataFrame()
    wide = inc.pivot_table(index=["date", "stock_id"], columns="type",
                           values="value", aggfunc="first").reset_index()
    for t in INCOME_TYPES:
        if t not in wide.columns:
            wide[t] = None
    # 淨利：優先用歸屬母公司淨利（和 EPS 同基礎），沒有才用稅後淨利
    wide["net_income"] = wide["EquityAttributableToOwnersOfParent"].where(
        wide["EquityAttributableToOwnersOfParent"].notna(), wide["IncomeAfterTaxes"])
    wide = wide.rename(columns={"EPS": "eps_cumulative", "GrossProfit": "gross_profit",
                                "CostOfGoodsSold": "cost_of_goods_sold"})
    wide = wide[["date", "stock_id", "eps_cumulative", "gross_profit",
                 "cost_of_goods_sold", "net_income"]]

    eq = pd.DataFrame(columns=["date", "stock_id", "equity"])
    if bs is not None and not bs.empty:
        picked = None
        for t in BS_EQUITY_TYPES:           # 依優先順序取第一個有資料的名稱
            sub = bs[bs["type"] == t]
            if not sub.empty:
                picked = sub
                break
        if picked is not None:
            eq = (picked.groupby(["date", "stock_id"], as_index=False)["value"].first()
                  .rename(columns={"value": "equity"}))
    wide = wide.merge(eq, on=["date", "stock_id"], how="left")
    return wide[["date", "stock_id"] + ALL_COLS]


def fetch_financials(dl: DataLoader, stock_id: str) -> pd.DataFrame:
    """損益表一次呼叫 + 資產負債表一次呼叫（每檔共 2 次 API）。"""
    income = dl.taiwan_stock_financial_statement(
        stock_id=stock_id, start_date=START_DATE, end_date=END_DATE)
    if income.empty:
        return income
    time.sleep(0.3)
    bs = dl.taiwan_stock_balance_sheet(
        stock_id=stock_id, start_date=START_DATE, end_date=END_DATE)
    return build_wide(income, bs)


def check_stock(dl: DataLoader, stock_id: str):
    """診斷：不寫資料庫，只印出 FinMind 實際回傳的欄位名稱與合併結果。"""
    income = dl.taiwan_stock_financial_statement(
        stock_id=stock_id, start_date=START_DATE, end_date=END_DATE)
    bs = dl.taiwan_stock_balance_sheet(
        stock_id=stock_id, start_date=START_DATE, end_date=END_DATE)
    print(f"損益表 {len(income)} 筆；資產負債表 {len(bs)} 筆")
    for name, d in (("損益表", income), ("資產負債表", bs)):
        if d.empty:
            continue
        hits = sorted(t for t in d["type"].unique()
                      if any(k in t.lower() for k in ("equity", "income", "profit")))
        print(f"{name} 含 equity/income/profit 的 type：{hits}")
    print(build_wide(income, bs).tail(6).to_string())


def write_raw(engine, stock_id: str, df: pd.DataFrame):
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM raw.eps_quarterly WHERE stock_id = :sid"),
            {"sid": stock_id},
        )
    df.to_sql("eps_quarterly", engine, schema="raw", if_exists="append", index=False)


def main():
    import sys
    dl = DataLoader()
    if FINMIND_TOKEN:
        dl.login_by_token(api_token=FINMIND_TOKEN)
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":   # python fetch_eps_to_raw.py --check 2603
        check_stock(dl, sys.argv[2])
        return
    engine = get_engine()

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
