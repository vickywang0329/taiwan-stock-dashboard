"""
波段交易決策系統
呼叫 decision_engine.pipeline.run_decision_system()，把每檔股票的
Stock Score / Entry Score / 風險報酬比整合成 BUY_NOW / BUY_PULLBACK / WATCH / AVOID 四級訊號。

放進 pages/ 資料夾先行測試；確認運作正常後，再考慮是否將整個 app 的
主入口（Streamlit Cloud 的 Main file path）改成這支檔案。
"""
import streamlit as st
import pandas as pd

import i18n
from decision_engine import pipeline

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
    "col_entry_score": {"zh": "進場分數", "en": "Entry Score"},
    "col_rr": {"zh": "風險報酬比", "en": "Risk/Reward"},
    "col_entry_zone": {"zh": "進場區間", "en": "Entry zone"},
    "col_stop_target": {"zh": "停損 / 目標", "en": "Stop / Target"},
    "col_current_price": {"zh": "現價", "en": "Current price"},
    "col_zones": {"zh": "小幅布局 / 大幅買進", "en": "Small position / Full position"},
    "col_missing": {"zh": "尚缺條件", "en": "Missing conditions"},
    "col_exclude_reason": {"zh": "剔除原因", "en": "Exclusion reason"},

    "explain_buy_now": {
        "zh": [
            "股票分數：技術趨勢、動能、相對強度、法人動向與產業資金流向綜合計算，滿分100",
            "進場分數：目前價格相對突破點的ATR標準化距離，衡量是否漲多、乖離過大",
            "風險報酬比：(目標價−進場價)÷(進場價−停損價)",
            "進場區間：下緣為近20個交易日(不含當日)的最高價＝突破價，上緣為當日收盤價＋緩衝(0.3倍ATR14)",
            "停損／目標：停損取近期滾動低點，目標取2倍風險或前波高點取保守者",
        ],
        "en": [
            "Stock Score: weighted combination of trend, momentum, relative strength, institutional flow and sector flow, out of 100",
            "Entry Score: ATR-normalized distance from the breakout price, measuring whether the price has run up too far",
            "Risk/Reward: (target − entry) ÷ (entry − stop loss)",
            "Entry zone: lower bound = highest price of the past 20 trading days (excl. today) = breakout price; upper bound = today's close + 0.3×ATR14",
            "Stop / Target: stop loss uses the recent rolling low; target uses 2x risk or the prior swing high, whichever is more conservative",
        ],
    },
    "explain_pullback": {
        "zh": [
            "小幅布局：突破價±0.5倍ATR14的淺回檔區，量能未異常放大且支撐未破時可小幅布局",
            "大幅買進：突破價−1.5倍ATR14的深回檔區，需額外確認法人未持續賣超才可大幅加碼",
            "停損／目標：計算方式同上方明天可買區塊",
        ],
        "en": [
            "Small position (Zone 1): shallow pullback within ±0.5×ATR14 of the breakout price; suitable for a small position if volume isn't abnormally high and support holds",
            "Full position (Zone 2): deeper pullback down to −1.5×ATR14 of the breakout price; only add a full position if institutional selling hasn't persisted",
            "Stop / Target: same calculation as the Buy Now section above",
        ],
    },

    "signal_table_title": {"zh": "股票分數對應訊號說明", "en": "Stock score to signal reference"},
    "col_score_range": {"zh": "股票分數", "en": "Stock score"},
    "col_signal": {"zh": "訊號", "en": "Signal"},
    "col_condition": {"zh": "條件", "en": "Condition"},
    "cond_buy_now": {"zh": "品質達標，且進場分數／風險報酬比／乖離幅度同時達標",
                     "en": "Passes quality bar, and entry score / risk-reward / bias are all within range"},
    "cond_pullback": {"zh": "品質達標，但目前股價漲多、乖離過大，等回檔",
                      "en": "Passes quality bar, but price has run up too far — wait for a pullback"},
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


st.set_page_config(page_title=t("page_title"), layout="wide")
i18n.init_language()
st.title(t("title"))

with st.spinner(t("loading")):
    decisions = pipeline.run_decision_system()

if not decisions:
    st.warning(t("no_data"))
    st.stop()

lang = i18n.get_lang()
st.caption(t("data_updated", date=pd.Timestamp.today().strftime("%Y/%m/%d")))


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


# ---- BUY_NOW：明天可買 ----
render_badge(t("signal_buy_now"), "buy_now", len(buy_now))
if buy_now:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_entry_score"): d.entry_score,
        t("col_rr"): d.rr_ratio,
        t("col_entry_zone"): format_range(d.entry_zone),
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
        t("col_zones"): (
            f"{format_range(d.pullback_zones['zone1'])} / {format_range(d.pullback_zones['zone2'])}"
            if d.pullback_zones else "-"
        ),
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

st.sidebar.markdown("---")
st.sidebar.caption("A project by I.H. Wang")
