"""
波段交易決策系統 Swing Trading Decision System
主入口頁面（放在 dashboard_project 根目錄，取代原本的 dashboard.py 作為主入口）
呼叫 decision_engine.pipeline.run_decision_system()，把每檔股票的
Stock Score / Entry Score / 風險報酬比整合成 BUY_NOW / BUY_PULLBACK / WATCH / AVOID 四級訊號。

執行方式：streamlit run Swing_Trading_Decision_System.py
"""
import streamlit as st
import pandas as pd

import i18n
from decision_engine import pipeline
from watchlist import WATCHLIST

TEXT = {
    "page_title": {"zh": "波段交易決策系統", "en": "Swing Trading Decision System"},
    "title": {"zh": "波段交易決策系統", "en": "Swing Trading Decision System"},
    "data_updated": {"zh": "資料更新至 {date}", "en": "Data as of {date}"},
    "loading": {"zh": "正在計算所有股票的決策訊號…", "en": "Computing decision signals for all stocks…"},
    "no_data": {"zh": "目前沒有足夠資料可以計算，請確認資料庫已經有股價與技術指標資料。",
               "en": "Not enough data to compute signals yet. Please confirm price and indicator data exist."},

    "signal_buy_now": {"zh": "明天可買", "en": "Buy now"},
    "signal_pullback": {"zh": "等回檔", "en": "Wait for pullback"},
    "signal_watch": {"zh": "觀察中", "en": "Watching"},
    "signal_avoid": {"zh": "避免", "en": "Avoid"},
    "unit_stocks": {"zh": "{n} 檔", "en": "{n} stocks"},

    "col_stock": {"zh": "股票", "en": "Stock"},
    "col_stock_score": {"zh": "股票分數", "en": "Stock Score"},
    "col_entry_zone": {"zh": "進場區間", "en": "Entry zone"},
    "col_stop_target": {"zh": "技術面停損/目標", "en": "Technical stop/target"},
    "col_current_price": {"zh": "現價", "en": "Current price"},
    "col_zones": {"zh": "技術面進場價", "en": "Technical entry price"},
    "zones_format": {"zh": "{small}元以下小幅布局，{full}元以下大幅加碼",
                     "en": "Small position below {small}, full position below {full}"},
    "col_fair_value": {"zh": "基本面合理價格", "en": "Fundamental fair value"},
    "col_missing": {"zh": "尚缺條件", "en": "Missing conditions"},
    "col_exclude_reason": {"zh": "剔除原因", "en": "Exclusion reason"},

    "coverage_full": {"zh": "本次決策系統涵蓋 {n} / {total} 檔股票", "en": "This run covers {n} / {total} stocks"},
    "coverage_partial": {"zh": "⚠️ 本次決策系統只涵蓋 {n} / {total} 檔股票，以下股票因資料不足被跳過：",
                         "en": "⚠️ This run only covers {n} / {total} stocks. The following were skipped due to insufficient data:"},

    "search_label": {"zh": "🔍 搜尋股票代碼", "en": "🔍 Search by stock code"},
    "search_placeholder": {"zh": "輸入股票代碼，例如 2330", "en": "Enter stock code, e.g. 2330"},
    "search_result_title": {"zh": "查詢結果", "en": "Search result"},
    "search_not_in_watchlist": {"zh": "個股尚未新增至資料庫", "en": "This stock hasn't been added to the database yet"},
    "search_insufficient_data": {"zh": "個股已在觀察池中，但目前資料不足，尚未列入本次計算結果",
                                 "en": "This stock is in the watchlist, but there isn't enough data yet to include it in this run"},

    "explain_buy_now": {
        "zh": [
            "股票分數：技術趨勢、動能、相對強度、法人動向、產業資金流向、估值與毛利率趨勢綜合計算，滿分100（權重明細見下方*）",
            "明天可買的判斷條件（五項須同時成立）：估值未過虛(本益比≤同業基準1.5倍)、收盤價站上突破價、乖離幅度合理(未追高)、近3日法人合計買超、估算全年EPS優於去年",
            "進場區間：下緣為近20個交易日(不含當日)的最高價＝突破價，上緣為當日收盤價＋緩衝(0.3倍ATR14)",
            "基本面合理價格：優先以「今年上半年EPS×(去年全年EPS÷去年上半年EPS)」估算全年EPS（要求去年上半年、全年皆為正值獲利，且比例介於0.5~5倍之間），不符合則改用近四季(TTM)實際EPS合計；再乘以同產業(排除異常值)平均本益比得出",
            "技術面停損/目標：停損取近期滾動低點，目標取2倍風險或前波高點取保守者",
        ],
        "en": [
            "Stock Score: weighted combination of trend, momentum, relative strength, institutional flow, sector flow, valuation and margin trend, out of 100 (weights below*)",
            "Buy-now criteria (all five must hold): valuation not overextended (P/E ≤ 1.5× industry benchmark), close above breakout price, bias within range (not chasing), positive 3-day institutional net buy, estimated full-year EPS above last year's",
            "Entry zone: lower bound = highest price of the past 20 trading days (excl. today) = breakout price; upper bound = today's close + 0.3×ATR14",
            "Fundamental fair value: prioritizes estimating full-year EPS via H1 EPS × (last year's full-year EPS ÷ last year's H1 EPS) — requires last year's H1 and full-year figures to both be profitable, with the ratio between 0.5x-5x; otherwise falls back to trailing-twelve-months (TTM) actual EPS. Multiplied by the industry's average P/E (outliers excluded)",
            "Technical stop/target: stop loss uses the recent rolling low; target uses 2x risk or the prior swing high, whichever is more conservative",
        ],
    },
    "explain_pullback": {
        "zh": [
            "等回檔的意思：股票分數已達標(≥85)，但「明天可買」五項條件未全部成立（常見原因：估值過虛、乖離過大、法人未同步買超、或EPS未成長），故建議等回檔",
            "小幅布局門檻：突破價＋0.5倍ATR14，股價跌破此價位、量能未異常放大且支撐未破時可小幅布局",
            "大幅加碼門檻：突破價－0.5倍ATR14，股價跌破此價位、且法人未持續賣超時可大幅加碼",
            "基本面合理價格：計算方式同上方明天可買區塊",
            "技術面停損/目標：計算方式同上方明天可買區塊",
        ],
        "en": [
            "What \"wait for pullback\" means: Stock Score already passes (≥85), but not all five buy-now conditions hold (common reasons: overextended valuation, price too far from breakout, no institutional buying, or EPS not growing) — so wait for a pullback instead",
            "Small-position threshold: breakout price + 0.5×ATR14 — once price falls below this, without abnormal volume and support intact, a small position is reasonable",
            "Full-position threshold: breakout price − 0.5×ATR14 — once price falls below this, and institutional selling hasn't persisted, adding a full position is reasonable",
            "Fundamental fair value: same calculation as the Buy Now section above",
            "Technical stop/target: same calculation as the Buy Now section above",
        ],
    },

    "signal_table_title": {"zh": "股票分數對應訊號說明", "en": "Stock score to signal reference"},
    "weights_note": {
        "zh": "*股票分數計算權重：trend 20%、momentum 16%、relative_strength 16%、"
              "institutional_flow 16%、sector_flow 12%、valuation 10%、margin_trend 10%，尚未經過回測",
        "en": "*Stock Score weights: trend 20%, momentum 16%, relative_strength 16%, "
              "institutional_flow 16%, sector_flow 12%, valuation 10%, margin_trend 10% — not yet backtested",
    },
    "col_score_range": {"zh": "股票分數", "en": "Stock score"},
    "col_signal": {"zh": "訊號", "en": "Signal"},
    "col_condition": {"zh": "條件", "en": "Condition"},
    "cond_buy_now": {
        "zh": "品質達標(≥85分)，且五項條件同時成立：估值合理、站上突破價、乖離幅度合理(未追高)、法人買超、EPS優於去年",
        "en": "Passes quality bar (≥85), and all five hold: valuation reasonable, above breakout price, bias within range (not chasing), institutional net buying, EPS above last year's",
    },
    "cond_pullback": {
        "zh": "品質達標(≥85分)，但明天可買的五項條件未全部成立，故建議等回檔",
        "en": "Passes quality bar (≥85), but not all five buy-now conditions hold — wait for a pullback instead",
    },
    "cond_watch": {"zh": "品質及格但條件不夠齊全，持續追蹤",
                  "en": "Passes the minimum bar but conditions aren't fully met yet — keep watching"},
    "cond_avoid": {"zh": "品質不足，不列入考慮", "en": "Below quality bar — not under consideration"},
}


def t(key: str, **kwargs):
    lang = i18n.get_lang()
    val = TEXT[key][lang]
    if isinstance(val, list):
        return val
    return val.format(**kwargs) if kwargs else val


def format_price(v) -> str:
    """股價 < 100 顯示 2 位小數；100~1000 顯示 1 位小數；>= 1000 顯示整數並加千分位逗號"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if v < 100:
        return f"{v:.2f}"
    elif v < 1000:
        return f"{v:.1f}"
    else:
        return f"{v:,.0f}"


def format_range(rng) -> str:
    if not rng:
        return "-"
    lo, hi = rng
    return f"{format_price(lo)}–{format_price(hi)}"


def format_zones(d) -> str:
    if not d.pullback_zones:
        return "-"
    small = d.pullback_zones["zone1"][1]   # Zone1 上緣＝小幅布局的價格門檻
    full = d.pullback_zones["zone2"][1]    # Zone2 上緣＝大幅加碼的價格門檻
    return t("zones_format", small=format_price(small), full=format_price(full))


@st.cache_data(ttl=3600, persist="disk")
def _cached_run_decision_system():
    """
    包一層 Streamlit 快取，避免使用者每次跟頁面互動（切換語言、
    Streamlit 背景自動重跑等）都要重新打資料庫、重新算全部163檔股票的
    完整決策流程。快取1小時，跟資料每日更新的頻率搭配已經足夠新鮮。

    persist="disk"：改存到硬碟，不是只存在記憶體——這樣即使重啟
    Streamlit 程式（開發時常需要這麼做），快取依然保留，不會每次
    重啟都要重新完整計算一次。
    """
    return pipeline.run_decision_system()


st.set_page_config(page_title=t("page_title"), layout="wide")
i18n.init_language()
st.title(t("title"))

if st.sidebar.button(
    "🔄 " + ("重新整理資料" if i18n.get_lang() == "zh" else "Refresh data"),
    help="跑完更新腳本後點這裡，立即反映最新資料" if i18n.get_lang() == "zh"
         else "Click after running the update scripts to reflect the latest data immediately",
):
    st.cache_data.clear()
    st.rerun()

with st.spinner(t("loading")):
    decisions = _cached_run_decision_system()

if not decisions:
    st.warning(t("no_data"))
    st.stop()

lang = i18n.get_lang()
st.caption(t("data_updated", date=pd.Timestamp.today().strftime("%Y/%m/%d")))

# ---- 資料涵蓋率：確保「有股票被跳過」這件事看得到，不會悄悄消失 ----
covered_ids = {d.stock_id for d in decisions}
missing_ids = [sid for sid in WATCHLIST if sid not in covered_ids]
if missing_ids:
    st.warning(t("coverage_partial", n=len(covered_ids), total=len(WATCHLIST)) + "、".join(missing_ids))
else:
    st.caption(t("coverage_full", n=len(covered_ids), total=len(WATCHLIST)))


def get_name(d) -> str:
    if lang == "en" and getattr(d, "name_en", None):
        return f"{d.stock_id} {d.name_en}"
    return f"{d.stock_id} {d.name}"


buy_now = [d for d in decisions if d.signal == "BUY_NOW"]
buy_pullback = [d for d in decisions if d.signal == "BUY_PULLBACK"]
watch = [d for d in decisions if d.signal == "WATCH"]
avoid = [d for d in decisions if d.signal == "AVOID"]

BADGE_STYLE = (
    "display:inline-block;font-size:12px;padding:3px 10px;border-radius:6px;font-weight:500;"
)
BADGE_COLORS = {
    "buy_now": ("#E8F5E9", "#2E7D32"),
    "pullback": ("#FFF3E0", "#EF6C00"),
    "watch": ("#E3F2FD", "#1565C0"),
    "avoid": ("#FFEBEE", "#C62828"),
}


def render_badge(label: str, color_key: str, count: int):
    bg, fg = BADGE_COLORS[color_key]
    st.markdown(
        f"<span style='{BADGE_STYLE}background:{bg};color:{fg}'>{label}</span>"
        f"&nbsp;&nbsp;<span style='font-size:13px;color:#666'>{t('unit_stocks', n=count)}</span>",
        unsafe_allow_html=True,
    )


SIGNAL_BADGE_KEY = {
    "BUY_NOW": ("signal_buy_now", "buy_now"),
    "BUY_PULLBACK": ("signal_pullback", "pullback"),
    "WATCH": ("signal_watch", "watch"),
    "AVOID": ("signal_avoid", "avoid"),
}


def render_search_result(d):
    """依股票目前的分類，渲染出跟該分類表格對應的完整詳細卡片。"""
    label_key, color_key = SIGNAL_BADGE_KEY[d.signal]
    render_badge(t(label_key), color_key, 1)

    detail = {t("col_stock"): get_name(d), t("col_stock_score"): d.stock_score}
    if d.signal == "BUY_NOW":
        detail[t("col_entry_zone")] = format_range(d.entry_zone)
        detail[t("col_fair_value")] = format_price(d.fair_value_estimate)
        detail[t("col_stop_target")] = f"{format_price(d.stop_loss)} / {format_price(d.target)}"
    elif d.signal == "BUY_PULLBACK":
        detail[t("col_current_price")] = format_price(d.current_price)
        detail[t("col_fair_value")] = format_price(d.fair_value_estimate)
        detail[t("col_zones")] = format_zones(d)
        detail[t("col_stop_target")] = f"{format_price(d.stop_loss)} / {format_price(d.target)}"
    elif d.signal == "WATCH":
        detail[t("col_missing")] = "、".join(d.missing_conditions)
    elif d.signal == "AVOID":
        detail[t("col_exclude_reason")] = "、".join(d.exclude_reason)

    st.dataframe(pd.DataFrame([detail]), use_container_width=True, hide_index=True)


st.markdown("---")
search_query = st.text_input(t("search_label"), placeholder=t("search_placeholder"))
if search_query:
    query = search_query.strip()
    matched = [d for d in decisions if d.stock_id == query]
    if matched:
        render_search_result(matched[0])
    elif query in WATCHLIST:
        st.info(t("search_insufficient_data"))
    else:
        st.info(t("search_not_in_watchlist"))
st.markdown("---")

# ---- BUY_NOW：明天可買 ----
render_badge(t("signal_buy_now"), "buy_now", len(buy_now))
if buy_now:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_entry_zone"): format_range(d.entry_zone),
        t("col_fair_value"): format_price(d.fair_value_estimate),
        t("col_stop_target"): f"{format_price(d.stop_loss)} / {format_price(d.target)}",
    } for d in buy_now]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
for line in t("explain_buy_now"):
    st.caption(f"• {line}")

st.markdown("---")

# ---- BUY_PULLBACK：等回檔 ----
render_badge(t("signal_pullback"), "pullback", len(buy_pullback))
if buy_pullback:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_current_price"): format_price(d.current_price),
        t("col_fair_value"): format_price(d.fair_value_estimate),
        t("col_zones"): format_zones(d),
        t("col_stop_target"): f"{format_price(d.stop_loss)} / {format_price(d.target)}",
    } for d in buy_pullback]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
for line in t("explain_pullback"):
    st.caption(f"• {line}")

st.markdown("---")

# ---- WATCH：觀察中 ----
render_badge(t("signal_watch"), "watch", len(watch))
if watch:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_missing"): "、".join(d.missing_conditions),
    } for d in watch]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ---- AVOID：避免 ----
render_badge(t("signal_avoid"), "avoid", len(avoid))
if avoid:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_exclude_reason"): "、".join(d.exclude_reason),
    } for d in avoid]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ---- 股票分數對應訊號說明 ----
st.subheader(t("signal_table_title"))
ref_rows = [
    {t("col_score_range"): "≥ 85", t("col_signal"): t("signal_buy_now"), t("col_condition"): t("cond_buy_now")},
    {t("col_score_range"): "≥ 85", t("col_signal"): t("signal_pullback"), t("col_condition"): t("cond_pullback")},
    {t("col_score_range"): "65–84", t("col_signal"): t("signal_watch"), t("col_condition"): t("cond_watch")},
    {t("col_score_range"): "< 65", t("col_signal"): t("signal_avoid"), t("col_condition"): t("cond_avoid")},
]
st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
st.caption(t("weights_note"))

st.sidebar.markdown("---")
st.sidebar.caption("A project by I.H. Wang")
