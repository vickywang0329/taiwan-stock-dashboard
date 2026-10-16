"""
公司行動（股票分割）清單——單一來源。

FinMind 的 raw 股價是「未還原」的原始成交價：股票分割（例如 1 拆 4）當天，
價格會一夜之間掉到 1/4，但實際上股東財富沒變。如果不還原，技術指標
（MA、ATR、報酬率）、相對強度、大盤環境判斷都會被這一天的「假崩跌」汙染，
持續約 60 個交易日。

做法：raw 層保持原始資料不動；在 transform_to_staging.py 轉成 staging 時，
把「生效日之前」的價格除以分割比例，成交量與法人買賣超股數、融資融券餘額
乘以分割比例，讓整段歷史在同一個單位下連續。

新增分割時，只要在下面加一行：
    ("股票代號", "分割後第一個交易日", 比例)
比例＝分割前價格 ÷ 分割後價格（1 拆 4 填 4）。下一次每日更新時，
這些股票會自動整段歷史重算（staging 與 mart 都是），不需要手動全量重建。

怎麼知道有新的分割？transform_to_staging.py 每次執行後，會在日誌印出
「單日漲跌幅超過限制」的股票；看到單日跌幅 50% 以上就很可能是分割，
用分割前收盤 ÷ 分割後收盤，取最接近的整數（並用同產業股票當天的漲跌幅
確認），填進來即可。
"""

# (股票代號, 生效日＝分割後第一個交易日, 分割比例)
SPLITS = [
    ("0050", "2025-06-18", 4),    # 元大台灣50 1 拆 4
    ("2327", "2025-08-25", 4),    # 國巨 1 拆 4
    ("6669", "2026-09-02", 3),    # 緯穎 1 拆 3
    ("00631L", "2026-03-31", 22), # 元大台灣50正2（用 0050 當天 -2.1% × 約2倍槓桿驗證，比例 22 而非 23）
]

SPLIT_STOCK_IDS = frozenset(s for s, _, _ in SPLITS)


# ---------------------------------------------------------------------------
# EPS 分割調整
#
# raw.eps_quarterly 的 EPS 是各季「當時公布」的原始數字，FinMind 不會在分割後回頭
# 重算歷史季度。但股價（staging）已經還原成分割後的單位，兩邊單位不一致，
# 本益比、每股淨值就會算錯（例如緯穎分割 1 拆 3 之後，EPS 還是舊單位，
# 本益比被低估 3 倍，同產業平均本益比也被拉低）。
#
# 規則：某一季的財報「法定公布期限」早於分割生效日 → 該季 EPS 是分割前單位，
# 除以分割比例；期限晚於生效日 → 已經是分割後單位，不動。
# 法定期限：Q1 5/15、Q2 8/14、Q3 11/14、Q4 隔年 3/31。
# ---------------------------------------------------------------------------
import pandas as pd

_DEADLINE_MD = {3: (5, 15, 0), 6: (8, 14, 0), 9: (11, 14, 0), 12: (3, 31, 1)}


def report_deadline(quarter_end) -> pd.Timestamp:
    """財報季底日期 → 法定公布期限。"""
    q = pd.Timestamp(quarter_end)
    m, d, add_year = _DEADLINE_MD.get(q.month, (q.month, 28, 0))
    return pd.Timestamp(year=q.year + add_year, month=m, day=d)


def adjust_eps_for_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    df 需含 stock_id、date（季底日期）、eps_cumulative 三欄。
    回傳調整後的新 DataFrame（不改動傳入的物件）。
    """
    if df.empty or not SPLITS:
        return df
    out = df.copy()
    deadlines = out["date"].apply(report_deadline)
    for sid, eff_date, ratio in SPLITS:
        mask = (out["stock_id"] == sid) & (deadlines < pd.Timestamp(eff_date))
        out.loc[mask, "eps_cumulative"] = out.loc[mask, "eps_cumulative"] / ratio
    return out
