"""
paper_trading.py
================
紙上交易（前瞻測試）：每天記錄決策系統產生的 BUY_NOW，模擬下單，並追蹤勝率與報酬率。

這「不是回測」：訊號是系統當天實際產生、當天寫入資料庫，之後不能修改；進出場價格是之後才發生的
真實價格。用途是檢驗回測的規則在樣本外是否還有效。

固定規則（與 v7 回測一致，開始之後不要改，否則結果又變成事後調整）：
  • 訊號：decision_engine 的 BUY_NOW（分數≥85、基本面通過、大盤多頭）
  • 進場：訊號日收盤後確認，「下一個交易日開盤價」進場
  • 成本：來回 0.585%（手續費＋證交稅，不含滑價）
  • 同一檔股票、同一出場規則，持有期間不重複進場
  • 四種出場「並行」追蹤（同一批訊號各自獨立模擬）：
      H10 / H15 / H20 = 進場日起持有 10 / 15 / 20 個交易日，第 N 日收盤出場
      MA20            = 收盤跌破 MA20（還原價）的下一個交易日開盤出場，最長 60 個交易日
  • 報酬含現金股利與配股（用除權息事件還原）；staging 的價格只還原「corporate_actions.SPLITS」
    清單裡的分割，所以這裡不重複還原那幾筆，其餘事件都在這裡處理。

指令：
  python paper_trading.py run      # 每天更新（GitHub Actions 在資料更新完之後執行）
  python paper_trading.py report   # 印出目前的勝率／報酬率／覆蓋率

資料表（schema = paper）：
  signals          訊號紀錄，只新增、不修改
  trades           每筆（訊號日×股票×出場規則）的模擬交易狀態與結果
  daily_marks      持有中部位的每日報酬
  run_log          每天執行紀錄（大盤狀態、訊號數、被大盤擋掉的數量）
  corporate_events 除權息／減資／分割事件快取
  meta             起始日等
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

# ------------------------------------------------------------------ 固定規則
RULES = ("H10", "H15", "H20", "MA20")
COST = 0.00585            # 來回成本
MA_WINDOW = 20
MA_MAX_HOLD = 60          # MA20 出場最長持有（自訊號日起算的交易日數）
BENCHMARK = "0050"
PRICE_TABLE = "staging.daily_master"
RECOMPUTE_DAYS_AFTER_EXIT = 7   # 出場後 7 個日曆天內仍會重算（容許遲到的除權息資料）
PRICE_LOOKBACK_DAYS = 150       # 載入價格往前多看幾天，讓 MA20 有足夠歷史
NO_ENTRY_CANCEL_ROWS = 5        # 訊號後連續幾個交易日都沒有進場價就取消
MIN_TRADES_FOR_READING = 30     # 少於這個筆數，報表標註「樣本不足」


def rule_version() -> str:
    """寫進每筆訊號；規則改了，報表會警告同時出現多個版本。"""
    try:
        from decision_engine import engine
        thr = engine.THRESHOLDS["tech_score_buy_now"]
    except Exception:
        thr = "?"
    return f"v7|score>={thr}|gate|bull-only|exits={','.join(RULES)}|cost={COST:.3%}"


# ====================================================================
# 一、純計算邏輯（不碰資料庫，可單獨測試）
# ====================================================================
def _nz(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    a[~np.isfinite(a)] = np.nan
    a[a <= 0] = np.nan
    return a


class Prices:
    """單一股票的價格序列與還原資料。px 需含 date, open, close。"""

    def __init__(self, px: pd.DataFrame, ev: pd.DataFrame | None):
        px = px.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        self.dates = pd.to_datetime(px["date"]).values.astype("datetime64[ns]")
        self.open = _nz(px["open"].values)
        self.close = _nz(px["close"].values)
        self.n = len(px)
        self.ev = ev if ev is not None and len(ev) else pd.DataFrame(columns=["ex_date", "f", "in_s"])
        # S 序列：只還原「改變股數」的事件（分割、減資、配股），MA20 與出場觸發用
        f_s = np.ones(self.n)
        for d, f in zip(self.ev.loc[self.ev["in_s"].astype(bool), "ex_date"],
                        self.ev.loc[self.ev["in_s"].astype(bool), "f"]):
            f_s[self.dates < np.datetime64(pd.Timestamp(d))] *= float(f)
        self.close_s = self.close * f_s
        self.ma20 = pd.Series(self.close_s).rolling(MA_WINDOW, min_periods=MA_WINDOW).mean().values
        self._pos = {d: i for i, d in enumerate(self.dates)}

    def idx(self, d) -> int | None:
        return self._pos.get(np.datetime64(pd.Timestamp(d), "ns"))

    def date(self, i: int) -> pd.Timestamp:
        return pd.Timestamp(self.dates[i])

    def mult(self, d_from, d_to) -> float:
        """含股利的還原乘數 = 對 (d_from, d_to] 內每個事件乘上 1/f（f = 參考價 ÷ 除權息前收盤）。"""
        if self.ev.empty:
            return 1.0
        ex = pd.to_datetime(self.ev["ex_date"])
        m = self.ev[(ex > pd.Timestamp(d_from)) & (ex <= pd.Timestamp(d_to))]
        return float(np.prod(1.0 / m["f"].astype(float).values)) if len(m) else 1.0

    def ex_factor_on(self, d) -> float:
        """某一天剛好是除權息日時的 1/f 乘積（沒有事件則為 1）。"""
        if self.ev.empty:
            return 1.0
        m = self.ev[pd.to_datetime(self.ev["ex_date"]) == pd.Timestamp(d)]
        return float(np.prod(1.0 / m["f"].astype(float).values)) if len(m) else 1.0


def _bm_return(bm: Prices | None, d_entry, d_exit, exit_kind: str) -> float:
    """0050 在同一個窗口的報酬（毛，含股利）。exit_kind: 'open' 或 'close'。"""
    if bm is None:
        return np.nan
    ie, ix = bm.idx(d_entry), bm.idx(d_exit)
    if ie is None or ix is None:
        return np.nan
    p0 = bm.open[ie]
    p1 = bm.open[ix] if exit_kind == "open" else bm.close[ix]
    if not (np.isfinite(p0) and np.isfinite(p1)):
        return np.nan
    return float(p1 * bm.mult(d_entry, d_exit) / p0 - 1)


def simulate_trade(rule: str, signal_date, P: Prices, bm: Prices | None = None) -> dict:
    """
    單筆模擬交易，每次都從訊號日重新算（無狀態），因此補跑、重跑、歷史價格被重算都不會出錯。

    回傳 dict：status ∈ {'pending_entry','open','exit_pending','closed','cancelled'}
    """
    out = dict(status="pending_entry", entry_date=None, entry_price=None, exit_date=None, exit_price=None,
               exit_reason=None, hold_days=None, gross_return=None, net_return=None, bm_return=None,
               trigger_date=None, last_price_date=None, last_price=None, unrealized_gross=None, marks=[])
    i = P.idx(signal_date)
    if i is None:
        return out                                  # 訊號日的價格還沒進資料庫
    n = P.n
    e = i + 1
    if e >= n:
        return out                                  # 還沒到進場日
    entry = P.open[e]
    if not np.isfinite(entry):
        if n - 1 - e >= NO_ENTRY_CANCEL_ROWS:
            out.update(status="cancelled", exit_reason="no_entry_price")
        return out
    out.update(status="open", entry_date=P.date(e), entry_price=float(entry))

    x, kind, reason, trigger = None, None, None, None
    if rule.startswith("H"):
        k = int(rule[1:])
        if i + k <= n - 1:
            x, kind, reason = i + k, "close", f"hold_{k}d"
            if not np.isfinite(P.close[x]):
                out.update(status="cancelled", exit_reason="no_exit_price")
                return out
    elif rule == "MA20":
        last_j = min(n - 1, i + MA_MAX_HOLD)
        for j in range(e, last_j + 1):
            c, m = P.close_s[j], P.ma20[j]
            if np.isfinite(c) and np.isfinite(m) and c < m:
                trigger = j
                if j + 1 <= n - 1 and np.isfinite(P.open[j + 1]):
                    x, kind, reason = j + 1, "open", "below_ma20"
                elif j + 1 <= n - 1:                        # 隔天沒有開盤價：退而用觸發日收盤（與回測一致）
                    x, kind, reason = j, "close", "below_ma20_fallback"
                break
        if x is None and trigger is None and i + MA_MAX_HOLD <= n - 1:
            x, kind, reason = i + MA_MAX_HOLD, "close", "max_hold_60d"
        if x is None and trigger is not None:
            out.update(trigger_date=P.date(trigger))        # 已觸發，等下一個交易日開盤
    else:
        raise ValueError(f"未知的出場規則 {rule}")

    last = n - 1 if x is None else x
    # ---- 每日報酬（含股利還原）----
    marks, prev_close = [], None
    for t in range(e, last + 1):
        if t == e:
            end = P.close[t]
            base = entry
        else:
            at_open_exit = (x is not None and t == x and kind == "open")
            end = P.open[t] if at_open_exit else P.close[t]
            base = prev_close
            if np.isfinite(end):
                end = end * P.ex_factor_on(P.date(t))
        if np.isfinite(end) and base is not None and np.isfinite(base):
            marks.append((P.date(t), float(end / base - 1), COST if (x is not None and t == x) else 0.0))
        if np.isfinite(P.close[t]):
            prev_close = P.close[t]
    out["marks"] = marks

    d_e = P.date(e)
    if x is not None:
        exit_px = P.open[x] if kind == "open" else P.close[x]
        gross = float(exit_px * P.mult(d_e, P.date(x)) / entry - 1)
        out.update(status="closed", exit_date=P.date(x), exit_price=float(exit_px), exit_reason=reason,
                   hold_days=int(x - e + 1), gross_return=gross, net_return=gross - COST,
                   bm_return=_bm_return(bm, d_e, P.date(x), kind))
    else:
        lp = P.close[last]
        if np.isfinite(lp):
            out.update(last_price_date=P.date(last), last_price=float(lp),
                       unrealized_gross=float(lp * P.mult(d_e, P.date(last)) / entry - 1),
                       bm_return=_bm_return(bm, d_e, P.date(last), "close"))
        out["status"] = "exit_pending" if trigger is not None else "open"
    return out


# ====================================================================
# 二、除權息事件（FinMind）
# ====================================================================
def _split_in_staging(stock_id: str, ex_date) -> bool:
    """staging 是否已經用 corporate_actions.SPLITS 還原過這個分割（同一檔、日期相差 5 天內）。"""
    try:
        import corporate_actions as ca
    except Exception:
        return False
    for sid, eff, _ in ca.SPLITS:
        if sid == stock_id and abs((pd.Timestamp(eff) - pd.Timestamp(ex_date)).days) <= 5:
            return True
    return False


def fetch_events_finmind(start: str, end: str, stock_ids: list[str]) -> pd.DataFrame:
    """
    抓 [start, end] 的除權息／減資／分割事件，轉成 (stock_id, ex_date, kind, f, in_s)。
    f = 參考價 ÷ 除權息前收盤（<1 代表價格下調）。先試整個市場一次抓，不行才逐檔抓。
    """
    from FinMind.data import DataLoader
    dl = DataLoader()
    token = os.environ.get("FINMIND_TOKEN", "")
    if token:
        dl.login_by_token(api_token=token)
    ids = set(stock_ids)

    def get(dataset: str) -> pd.DataFrame:
        try:
            df = dl.get_data(dataset=dataset, start_date=start, end_date=end)
            return df if df is not None else pd.DataFrame()
        except Exception:
            frames = []
            for sid in ids:
                d = dl.get_data(dataset=dataset, data_id=sid, start_date=start, end_date=end)
                if d is not None and len(d):
                    frames.append(d)
            return pd.concat(frames) if frames else pd.DataFrame()

    rows = []
    dv = get("TaiwanStockDividendResult")
    for r in dv.itertuples():
        if r.stock_id in ids and r.before_price and r.before_price > 0 and r.reference_price and r.reference_price > 0:
            in_s = "權" in str(r.stock_or_cache_dividend)
            rows.append((r.stock_id, pd.Timestamp(r.date), "div", r.reference_price / r.before_price, in_s))
    cr = get("TaiwanStockCapitalReductionReferencePrice")
    for r in cr.itertuples():
        if r.stock_id in ids and r.ClosingPriceonTheLastTradingDay > 0 and r.PostReductionReferencePrice > 0:
            rows.append((r.stock_id, pd.Timestamp(r.date), "capred",
                         r.PostReductionReferencePrice / r.ClosingPriceonTheLastTradingDay, True))
    sp = get("TaiwanStockSplitPrice")
    seen = set()
    for r in sp.itertuples():
        if r.stock_id in ids and r.before_price > 0 and not _split_in_staging(r.stock_id, r.date):
            rows.append((r.stock_id, pd.Timestamp(r.date), "split", r.after_price / r.before_price, True))
            seen.add((r.stock_id, pd.Timestamp(r.date)))
    pv = get("TaiwanStockParValueChange")
    for r in pv.itertuples():
        if r.stock_id in ids and r.before_close > 0 and (r.stock_id, pd.Timestamp(r.date)) not in seen \
                and not _split_in_staging(r.stock_id, r.date):
            rows.append((r.stock_id, pd.Timestamp(r.date), "split", r.after_ref_close / r.before_close, True))
    return pd.DataFrame(rows, columns=["stock_id", "ex_date", "kind", "f", "in_s"])


# ====================================================================
# 三、資料庫
# ====================================================================
DDL = [
    "CREATE SCHEMA IF NOT EXISTS paper",
    """CREATE TABLE IF NOT EXISTS paper.signals (
        signal_date DATE NOT NULL, stock_id TEXT NOT NULL, stock_name TEXT, stock_score NUMERIC,
        market_regime TEXT, market_gap_pct NUMERIC, signal_close NUMERIC, rule_version TEXT NOT NULL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (signal_date, stock_id))""",
    """CREATE TABLE IF NOT EXISTS paper.trades (
        signal_date DATE NOT NULL, stock_id TEXT NOT NULL, exit_rule TEXT NOT NULL, status TEXT NOT NULL,
        entry_date DATE, entry_price NUMERIC, exit_date DATE, exit_price NUMERIC, exit_reason TEXT,
        hold_days INTEGER, gross_return NUMERIC, net_return NUMERIC, bm_return NUMERIC,
        trigger_date DATE, last_price_date DATE, last_price NUMERIC, unrealized_gross NUMERIC,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (signal_date, stock_id, exit_rule))""",
    """CREATE TABLE IF NOT EXISTS paper.daily_marks (
        mark_date DATE NOT NULL, signal_date DATE NOT NULL, stock_id TEXT NOT NULL, exit_rule TEXT NOT NULL,
        day_return NUMERIC, cost NUMERIC, PRIMARY KEY (mark_date, signal_date, stock_id, exit_rule))""",
    """CREATE TABLE IF NOT EXISTS paper.run_log (
        data_date DATE PRIMARY KEY, run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, market_regime TEXT,
        market_gap_pct NUMERIC, n_buy_now INTEGER, n_blocked_by_regime INTEGER, n_trades_created INTEGER,
        events_ok INTEGER, rule_version TEXT)""",
    """CREATE TABLE IF NOT EXISTS paper.corporate_events (
        stock_id TEXT NOT NULL, ex_date DATE NOT NULL, kind TEXT NOT NULL, f NUMERIC NOT NULL, in_s BOOLEAN,
        PRIMARY KEY (stock_id, ex_date, kind))""",
    "CREATE TABLE IF NOT EXISTS paper.meta (key TEXT PRIMARY KEY, value TEXT)",
]
ACTIVE = ("pending_entry", "open", "exit_pending")


def _iso(d) -> str | None:
    return None if d is None or (isinstance(d, float) and np.isnan(d)) else pd.Timestamp(d).date().isoformat()


def _f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def _df(conn, sql: str, **params) -> pd.DataFrame:
    stmt = text(sql)
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            stmt = stmt.bindparams(bindparam(k, expanding=True))
    return pd.read_sql(stmt, conn, params=params)


def ensure_schema(conn) -> None:
    for stmt in DDL:
        if stmt.startswith("CREATE SCHEMA") and conn.dialect.name == "sqlite":
            continue
        conn.execute(text(stmt))


def _commit(conn) -> None:
    try:
        conn.commit()
    except Exception:
        pass


def latest_data_date(conn) -> pd.Timestamp:
    v = conn.execute(text(f"SELECT MAX(date) FROM {PRICE_TABLE}")).scalar()
    return pd.Timestamp(v)


def load_prices(conn, stock_ids: list[str], since: pd.Timestamp) -> dict[str, pd.DataFrame]:
    df = _df(conn, f"SELECT stock_id, date, open, close FROM {PRICE_TABLE} "
                   f"WHERE stock_id IN :ids AND date >= :d0 ORDER BY stock_id, date",
             ids=list(stock_ids), d0=_iso(since))
    df["date"] = pd.to_datetime(df["date"])
    return {sid: g[["date", "open", "close"]] for sid, g in df.groupby("stock_id")}


def load_events(conn, stock_ids: list[str]) -> dict[str, pd.DataFrame]:
    df = _df(conn, "SELECT stock_id, ex_date, kind, f, in_s FROM paper.corporate_events WHERE stock_id IN :ids",
             ids=list(stock_ids))
    if df.empty:
        return {}
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["in_s"] = df["in_s"].astype(bool)
    return {sid: g[["ex_date", "f", "in_s"]] for sid, g in df.groupby("stock_id")}


def save_events(conn, ev: pd.DataFrame) -> None:
    for r in ev.itertuples():
        conn.execute(text(
            "INSERT INTO paper.corporate_events (stock_id, ex_date, kind, f, in_s) VALUES (:s,:d,:k,:f,:i) "
            "ON CONFLICT (stock_id, ex_date, kind) DO UPDATE SET f=excluded.f, in_s=excluded.in_s"),
            dict(s=r.stock_id, d=_iso(r.ex_date), k=r.kind, f=float(r.f), i=bool(r.in_s)))


def record_signals_and_create_trades(conn, decisions, data_date: pd.Timestamp) -> tuple[int, int, int]:
    """寫入今天的 BUY_NOW（只新增、不修改），並為每個出場規則建立 pending 交易。回傳 (買進數, 被大盤擋掉數, 新增交易數)。"""
    buys = [d for d in decisions if d.signal == "BUY_NOW"]
    blocked = [d for d in decisions if d.signal == "WATCH"
               and any(str(c).startswith("market_regime_") for c in (d.missing_conditions or []))]
    regime = getattr(decisions[0], "market_regime", "unknown") if decisions else "unknown"
    if buys and regime != "bull":
        print(f"[paper] ⚠️ 大盤是 {regime} 卻出現 BUY_NOW，與規則不符，今天不記錄訊號。", flush=True)
        return len(buys), len(blocked), 0
    rv, created = rule_version(), 0
    for d in buys:
        conn.execute(text(
            "INSERT INTO paper.signals (signal_date, stock_id, stock_name, stock_score, market_regime, "
            "market_gap_pct, signal_close, rule_version) VALUES (:d,:s,:n,:sc,:r,:g,:c,:v) "
            "ON CONFLICT (signal_date, stock_id) DO NOTHING"),
            dict(d=_iso(data_date), s=d.stock_id, n=d.name, sc=_f(d.stock_score), r=d.market_regime,
                 g=_f(d.market_gap_pct), c=_f(d.current_price), v=rv))
        for rule in RULES:
            # 同一檔、同一規則：持有中或今天才剛出場就不重複進場（與回測的 d <= 前次出場日 一致）
            busy = conn.execute(text(
                "SELECT COUNT(*) FROM paper.trades WHERE stock_id=:s AND exit_rule=:r AND "
                "(status IN ('pending_entry','open','exit_pending') OR (status='closed' AND exit_date >= :d))"),
                dict(s=d.stock_id, r=rule, d=_iso(data_date))).scalar()
            if busy:
                continue
            res = conn.execute(text(
                "INSERT INTO paper.trades (signal_date, stock_id, exit_rule, status) VALUES (:d,:s,:r,'pending_entry') "
                "ON CONFLICT (signal_date, stock_id, exit_rule) DO NOTHING"),
                dict(d=_iso(data_date), s=d.stock_id, r=rule))
            created += res.rowcount if res.rowcount and res.rowcount > 0 else 0
    return len(buys), len(blocked), created


def update_trades(conn, data_date: pd.Timestamp, fetch_events=fetch_events_finmind) -> bool:
    """重算所有未結案（以及剛出場不久）的交易，寫回 trades 與 daily_marks。回傳事件資料是否更新成功。"""
    cutoff = _iso(data_date - timedelta(days=RECOMPUTE_DAYS_AFTER_EXIT))
    tr = _df(conn, "SELECT signal_date, stock_id, exit_rule, status FROM paper.trades WHERE status IN "
                   "('pending_entry','open','exit_pending') OR (status='closed' AND exit_date >= :c)", c=cutoff)
    if tr.empty:
        return True
    tr["signal_date"] = pd.to_datetime(tr["signal_date"])
    ids = sorted(set(tr["stock_id"]) | {BENCHMARK})
    since = tr["signal_date"].min() - timedelta(days=PRICE_LOOKBACK_DAYS)

    events_ok = True
    try:
        ev_new = fetch_events(_iso(tr["signal_date"].min() - timedelta(days=10)), _iso(data_date), ids)
        if ev_new is not None and len(ev_new):
            save_events(conn, ev_new)
    except Exception as e:                                   # 抓不到就用快取，並在 run_log 標註
        events_ok = False
        print(f"[paper] ⚠️ 除權息資料更新失敗，沿用快取：{str(e)[:120]}", flush=True)

    prices = load_prices(conn, ids, since)
    events = load_events(conn, ids)
    bm = Prices(prices[BENCHMARK], events.get(BENCHMARK)) if BENCHMARK in prices else None
    cache: dict[str, Prices] = {}
    for r in tr.itertuples():
        if r.stock_id not in prices:
            continue
        P = cache.setdefault(r.stock_id, Prices(prices[r.stock_id], events.get(r.stock_id)))
        res = simulate_trade(r.exit_rule, r.signal_date, P, bm)
        key = dict(d=_iso(r.signal_date), s=r.stock_id, r=r.exit_rule)
        conn.execute(text(
            "UPDATE paper.trades SET status=:st, entry_date=:ed, entry_price=:ep, exit_date=:xd, exit_price=:xp, "
            "exit_reason=:xr, hold_days=:hd, gross_return=:g, net_return=:n, bm_return=:b, trigger_date=:td, "
            "last_price_date=:ld, last_price=:lp, unrealized_gross=:u, updated_at=CURRENT_TIMESTAMP "
            "WHERE signal_date=:d AND stock_id=:s AND exit_rule=:r"),
            dict(key, st=res["status"], ed=_iso(res["entry_date"]), ep=_f(res["entry_price"]), xd=_iso(res["exit_date"]),
                 xp=_f(res["exit_price"]), xr=res["exit_reason"], hd=res["hold_days"], g=_f(res["gross_return"]),
                 n=_f(res["net_return"]), b=_f(res["bm_return"]), td=_iso(res["trigger_date"]),
                 ld=_iso(res["last_price_date"]), lp=_f(res["last_price"]), u=_f(res["unrealized_gross"])))
        conn.execute(text("DELETE FROM paper.daily_marks WHERE signal_date=:d AND stock_id=:s AND exit_rule=:r"), key)
        for md, ret, cost in res["marks"]:
            conn.execute(text(
                "INSERT INTO paper.daily_marks (mark_date, signal_date, stock_id, exit_rule, day_return, cost) "
                "VALUES (:m,:d,:s,:r,:ret,:c)"), dict(key, m=_iso(md), ret=ret, c=cost))
    return events_ok


def run_daily(conn, decisions=None, fetch_events=fetch_events_finmind) -> dict:
    """每天執行一次（資料更新完之後）。可重複執行，不會重複記錄。"""
    ensure_schema(conn)
    d = latest_data_date(conn)
    if decisions is None:
        from decision_engine import pipeline
        decisions = pipeline.run_decision_system()
    if not decisions:
        raise RuntimeError("決策系統沒有回傳任何結果，今天不更新紙上交易。")
    n_buy, n_block, created = record_signals_and_create_trades(conn, decisions, d)
    events_ok = update_trades(conn, d, fetch_events)
    regime = getattr(decisions[0], "market_regime", "unknown")
    gap = _f(getattr(decisions[0], "market_gap_pct", None))
    conn.execute(text(
        "INSERT INTO paper.run_log AS rl (data_date, market_regime, market_gap_pct, n_buy_now, n_blocked_by_regime, "
        "n_trades_created, events_ok, rule_version) VALUES (:d,:r,:g,:b,:k,:c,:e,:v) "
        "ON CONFLICT (data_date) DO UPDATE SET run_at=CURRENT_TIMESTAMP, market_regime=excluded.market_regime, "
        "market_gap_pct=excluded.market_gap_pct, n_buy_now=excluded.n_buy_now, "
        "n_blocked_by_regime=excluded.n_blocked_by_regime, n_trades_created=rl.n_trades_created + "
        "excluded.n_trades_created, events_ok=excluded.events_ok, rule_version=excluded.rule_version"),
        dict(d=_iso(d), r=regime, g=gap, b=n_buy, k=n_block, c=created, e=int(events_ok), v=rule_version()))
    conn.execute(text("INSERT INTO paper.meta (key, value) VALUES ('start_date', :d) ON CONFLICT (key) DO NOTHING"),
                 dict(d=_iso(d)))
    _commit(conn)
    info = dict(data_date=d.date().isoformat(), regime=regime, buy_now=n_buy, blocked_by_regime=n_block,
                trades_created=created, events_ok=events_ok)
    print(f"[paper] {info}", flush=True)
    return info


# ====================================================================
# 四、報表
# ====================================================================
def summary(conn) -> dict:
    """回傳各出場規則的績效表、覆蓋率、規則版本等（供 CLI 與 Streamlit 頁面共用）。"""
    tr = _df(conn, "SELECT * FROM paper.trades")
    sig = _df(conn, "SELECT signal_date, rule_version FROM paper.signals")
    log = _df(conn, "SELECT * FROM paper.run_log ORDER BY data_date")
    start = conn.execute(text("SELECT value FROM paper.meta WHERE key='start_date'")).scalar()
    rows = []
    for rule in RULES:
        t = tr[tr["exit_rule"] == rule]
        c = t[t["status"] == "closed"]
        o = t[t["status"].isin(["open", "exit_pending"])]
        net = c["net_return"].astype(float)
        ex = (c["net_return"].astype(float) - c["bm_return"].astype(float)).dropna()
        rows.append({
            "出場規則": rule, "已出場筆數": len(c), "持有中": len(o), "待進場": int((t["status"] == "pending_entry").sum()),
            "勝率": float((net > 0).mean()) if len(c) else np.nan,
            "平均淨報酬": float(net.mean()) if len(c) else np.nan,
            "中位數": float(net.median()) if len(c) else np.nan,
            "最差單筆": float(net.min()) if len(c) else np.nan,
            "平均超額(對0050)": float(ex.mean()) if len(ex) else np.nan,
            "持有中平均未實現(毛)": float(o["unrealized_gross"].astype(float).mean()) if len(o) else np.nan,
            "樣本不足": len(c) < MIN_TRADES_FOR_READING,
        })
    cover = {}
    if start is not None:
        cal = _df(conn, f"SELECT DISTINCT date FROM {PRICE_TABLE} WHERE stock_id=:b AND date >= :s ORDER BY date",
                  b=BENCHMARK, s=start)
        cal_days = set(pd.to_datetime(cal["date"]).dt.date) if len(cal) else set()
        ran = set(pd.to_datetime(log["data_date"]).dt.date) if len(log) else set()
        cover = dict(start_date=str(start)[:10], trading_days=len(cal_days), recorded_days=len(ran & cal_days),
                     missed_days=sorted(str(x) for x in (cal_days - ran)))
    return dict(table=pd.DataFrame(rows), coverage=cover,
                versions=sorted(set(sig["rule_version"])) if len(sig) else [],
                regime_days=log["market_regime"].value_counts().to_dict() if len(log) else {},
                signals=len(sig), events_failed_days=int((log["events_ok"] == 0).sum()) if len(log) else 0)


def daily_returns(conn, rule: str) -> pd.DataFrame:
    """等權重、持有中部位的每日平均報酬（扣出場成本）；空手日為 0。供畫累積報酬曲線，屬近似值。"""
    m = _df(conn, "SELECT mark_date, day_return, cost FROM paper.daily_marks WHERE exit_rule=:r", r=rule)
    if m.empty:
        return pd.DataFrame(columns=["date", "return", "cum"])
    m["mark_date"] = pd.to_datetime(m["mark_date"])
    m["net"] = m["day_return"].astype(float) - m["cost"].astype(float)
    d = m.groupby("mark_date")["net"].mean().rename("return").reset_index().rename(columns={"mark_date": "date"})
    d["cum"] = (1 + d["return"]).cumprod() - 1
    return d


def print_report(conn) -> None:
    s = summary(conn)
    pd.set_option("display.width", 200)
    print("=" * 70, "\n紙上交易報告（前瞻測試，不是回測）\n" + "=" * 70)
    c = s["coverage"]
    if c:
        print(f"起始日 {c['start_date']}；交易日 {c['trading_days']} 天，已記錄 {c['recorded_days']} 天，漏跑 {len(c['missed_days'])} 天")
        if c["missed_days"]:
            print("  漏跑日期（這些天的訊號沒有被記錄）：", "、".join(c["missed_days"][:15]))
    print("規則版本：", "；".join(s["versions"]) or "（尚無訊號）")
    if len(s["versions"]) > 1:
        print("⚠️ 出現多個規則版本，績效不可直接合併解讀。")
    print("大盤狀態天數：", s["regime_days"], f"｜累計訊號 {s['signals']} 檔次")
    if s["events_failed_days"]:
        print(f"⚠️ 有 {s['events_failed_days']} 天除權息資料更新失敗（沿用快取，報酬可能略有誤差）")
    t = s["table"].copy()
    for col in ("勝率", "平均淨報酬", "中位數", "最差單筆", "平均超額(對0050)", "持有中平均未實現(毛)"):
        t[col] = t[col].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}" if col == "勝率" else f"{v:+.2%}")
    t["樣本不足"] = t["樣本不足"].map(lambda b: f"是（<{MIN_TRADES_FOR_READING} 筆）" if b else "否")
    print(t.to_string(index=False))
    print("\n勝率 = 淨報酬 > 0 的比例；淨報酬已扣來回成本 0.585%；超額 = 淨報酬 − 0050 同一窗口報酬。")
    print(f"已出場未滿 {MIN_TRADES_FOR_READING} 筆之前，這些數字只是雜訊，請不要下結論。")


# ====================================================================
# 五、指令列
# ====================================================================
def _load_env() -> None:
    """本機執行時載入專案根目錄的 .env（db.py 只讀環境變數與 st.secrets，不會自己讀 .env）。
    Actions 上 secrets 已是環境變數，.env 不存在時什麼都不做。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[paper] 找不到 python-dotenv，略過 .env 載入（pip install python-dotenv）")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, ".env"), os.path.join(os.getcwd(), ".env")):
        if os.path.exists(p):
            load_dotenv(p, override=False)
            return


def _connect():
    _load_env()
    if not os.environ.get("DB_USER"):
        print("[paper] ⚠️ 環境變數 DB_USER 是空的：.env 沒被讀到，或變數名稱不是 DB_USER。")
    from decision_engine import db
    return db.get_conn()


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "run"
    cm = _connect()
    with cm as conn:
        if cmd == "run":
            run_daily(conn)
        elif cmd == "report":
            ensure_schema(conn)
            print_report(conn)
        else:
            print("用法：python paper_trading.py [run|report]")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
