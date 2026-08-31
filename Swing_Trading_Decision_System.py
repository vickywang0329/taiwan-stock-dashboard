"""
波段交易決策系統 Swing Trading Decision System
主入口頁面（放在 dashboard_project 根目錄，取代原本的 dashboard.py 作為主入口）

⚠️ 2026/08 架構簡化：使用者確認徹底拿掉「技術面型態明確」跟「風險報酬比
≥2.0」這兩個條件，訊號判斷只看兩件事：
  1. 基本面守門檢查（估值/EPS成長/毛利率趨勢，景氣循環股豁免毛利率檢查）
  2. 技術籌碼分數（Stock Score，trend/momentum/relative_strength/
     institutional_flow/sector_flow 五項加權）
三分類：買進(BUY_NOW) / 觀察(WATCH) / 避免(AVOID，再分基本面地雷、雙重不合)。
每張表格都附上個股單日收盤價，作為統一的參考基準。

執行方式：streamlit run Swing_Trading_Decision_System.py
"""
import streamlit as st
import pandas as pd

import i18n
from decision_engine import pipeline, scoring
from watchlist import WATCHLIST

TEXT = {
    "page_title": {"zh": "波段交易決策系統", "en": "Swing Trading Decision System"},
    "title": {"zh": "波段交易決策系統", "en": "Swing Trading Decision System"},
    "data_updated": {"zh": "資料更新至 {date}", "en": "Data as of {date}"},
    "loading": {"zh": "正在計算所有股票的決策訊號…", "en": "Computing decision signals for all stocks…"},
    "no_data": {"zh": "目前沒有足夠資料可以計算，請確認資料庫已經有股價與技術指標資料。",
               "en": "Not enough data to compute signals yet. Please confirm price and indicator data exist."},

    "signal_buy_now": {"zh": "買進", "en": "Buy"},
    "signal_watch": {"zh": "觀察", "en": "Watch"},
    "signal_avoid_fundamental": {"zh": "避免（基本面地雷）", "en": "Avoid (Fundamental red flag)"},
    "signal_avoid_double": {"zh": "避免（雙重不合）", "en": "Avoid (Fails both)"},
    "unit_stocks": {"zh": "{n} 檔", "en": "{n} stocks"},

    "col_stock": {"zh": "股票", "en": "Stock"},
    "col_stock_score": {"zh": "股票分數", "en": "Stock Score"},
    "col_close_price": {"zh": "收盤價", "en": "Close price"},
    "col_fair_value": {"zh": "基本面合理價格", "en": "Fundamental fair value"},
    "col_missing": {"zh": "尚缺條件", "en": "Missing conditions"},
    "col_fail_reason": {"zh": "基本面未過原因", "en": "Fundamentals fail reason"},

    # 對應 decision_engine/engine.py 回傳的語言中立代碼（WATCH 用）
    "reason_tech_score_not_high_enough": {"zh": "技術籌碼分數尚未達85分", "en": "Stock Score below 85"},
    "reason_trend_weak": {"zh": "技術趨勢偏弱", "en": "Weak trend"},
    "reason_momentum_weak": {"zh": "動能偏弱", "en": "Weak momentum"},
    "reason_relative_strength_weak": {"zh": "相對大盤強度不足", "en": "Weak relative strength vs. market"},
    "reason_institutional_flow_weak": {"zh": "法人動向偏弱", "en": "Weak institutional flow"},
    "reason_sector_flow_weak": {"zh": "產業資金流向偏弱", "en": "Weak sector fund flow"},

    # 對應基本面守門檢查沒過的原因代碼（AVOID 用）
    "reason_overvalued": {"zh": "估值過虛（本益比或本淨比明顯偏離同業）", "en": "Overvalued (P/E or P/B far above industry peers)"},
    "reason_eps_declining": {"zh": "估算全年EPS較去年衰退", "en": "Estimated full-year EPS declining vs. last year"},
    "reason_margin_severely_declining": {"zh": "毛利率較去年同期惡化", "en": "Gross margin rate worse than same period last year"},

    "coverage_full": {"zh": "本次決策系統涵蓋 {n} / {total} 檔股票", "en": "This run covers {n} / {total} stocks"},
    "coverage_partial": {"zh": "⚠️ 本次決策系統只涵蓋 {n} / {total} 檔股票，以下股票因資料不足被跳過：",
                         "en": "⚠️ This run only covers {n} / {total} stocks. The following were skipped due to insufficient data:"},

    "search_label": {"zh": "🔍 搜尋股票代碼", "en": "🔍 Search by stock code"},
    "search_placeholder": {"zh": "輸入股票代碼，例如 2330", "en": "Enter stock code, e.g. 2330"},
    "search_not_in_watchlist": {"zh": "個股尚未新增至資料庫", "en": "This stock hasn't been added to the database yet"},
    "search_insufficient_data": {"zh": "個股已在觀察池中，但目前資料不足，尚未列入本次計算結果",
                                 "en": "This stock is in the watchlist, but there isn't enough data yet to include it in this run"},
    "score_breakdown_prefix": {"zh": "股票分數計算權重：", "en": "Stock Score breakdown: "},
    "score_breakdown_suffix": {"zh": "，總計股票分數 {total:.1f} 分", "en": ", total Stock Score {total:.1f}"},
    "score_breakdown_unavailable": {"zh": "（無法取得子分數明細）", "en": "(sub-score breakdown unavailable)"},

    "explain_buy_now": {
        "zh": [
            "股票分數（技術籌碼分數）：技術趨勢、動能、相對強度、法人動向、產業資金流向五項加權計算，滿分100，完全不含基本面成分（權重明細見下方*）",
            "買進的判斷條件（只看兩件事）：① 基本面通過守門檢查（估值未過虛、EPS優於去年、毛利率沒有惡化——景氣循環股豁免毛利率檢查） ② 股票分數≥85",
            "基本面合理價格：一般股優先以「今年上半年EPS×(去年全年EPS÷去年上半年EPS)」估算全年EPS，乘以同業本益比基準；景氣循環股改用「每股淨值×同業本淨比基準」",
        ],
        "en": [
            "Stock Score (technical/institutional score): weighted combination of trend, momentum, relative strength, institutional flow and sector flow, out of 100, with no fundamental component at all (weights below*)",
            "Buy criteria (only two things matter): ① fundamentals pass the gate check (valuation not overextended, EPS above last year, no severe margin decline — cyclical stocks are exempt from the margin check) ② Stock Score ≥ 85",
            "Fundamental fair value: non-cyclical stocks estimate full-year EPS via H1 EPS × (last year's full-year EPS ÷ last year's H1 EPS), multiplied by the industry P/E benchmark; cyclical stocks use book value per share × industry P/B benchmark instead",
        ],
    },
    "explain_watch": {
        "zh": [
            "觀察的意思：基本面已經通過守門檢查（體質沒問題），但股票分數還沒到85分，時機尚未成熟，持續追蹤",
        ],
        "en": [
            "What \"watch\" means: fundamentals already pass the gate check (no red flags), but Stock Score hasn't reached 85 yet — not the right timing yet, keep tracking",
        ],
    },
    "explain_avoid": {
        "zh": [
            "避免（基本面地雷）：技術籌碼分數其實不低（≥85），但基本面守門檢查沒通過——不管線型多漂亮，基本面已經亮紅燈，不建議進場",
            "避免（雙重不合）：技術籌碼分數不夠高，基本面也沒通過守門檢查，兩邊都不合格",
        ],
        "en": [
            "Avoid (Fundamental red flag): the technical/institutional score is actually high (≥85), but fundamentals failed the gate check — no matter how good the chart looks, a fundamental red flag means this isn't a buy",
            "Avoid (Fails both): the technical/institutional score isn't high enough, and fundamentals also failed the gate check — neither side qualifies",
        ],
    },

    "signal_table_title": {"zh": "股票分數對應訊號說明", "en": "Stock score to signal reference"},
    "weights_note": {
        "zh": "*股票分數（技術籌碼分數）計算權重：trend 25%、momentum 20%、relative_strength 20%、"
              "institutional_flow 20%、sector_flow 15%（總計100%，不含任何基本面成分）。"
              "基本面（估值/EPS成長/毛利率趨勢）為獨立的守門檢查，不參與加權。尚未經過回測。",
        "en": "*Stock Score (technical/institutional) weights: trend 25%, momentum 20%, relative_strength 20%, "
              "institutional_flow 20%, sector_flow 15% (100% total, no fundamental component). "
              "Fundamentals (valuation/EPS growth/margin trend) are a separate gate check, not weighted. Not yet backtested.",
    },
    "col_score_range": {"zh": "股票分數", "en": "Stock score"},
    "col_signal": {"zh": "訊號", "en": "Signal"},
    "col_condition": {"zh": "條件", "en": "Condition"},
    "cond_buy_now": {
        "zh": "基本面通過守門檢查，且股票分數≥85",
        "en": "Fundamentals pass the gate check, and Stock Score ≥ 85",
    },
    "cond_watch": {
        "zh": "基本面通過守門檢查，但股票分數未達85",
        "en": "Fundamentals pass the gate check, but Stock Score is below 85",
    },
    "cond_avoid_fundamental": {
        "zh": "股票分數≥85，但基本面未通過守門檢查",
        "en": "Stock Score ≥ 85, but fundamentals fail the gate check",
    },
    "cond_avoid_double": {
        "zh": "股票分數未達85，且基本面也未通過守門檢查",
        "en": "Stock Score below 85, and fundamentals also fail the gate check",
    },
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


def translate_reasons(codes: list[str]) -> str:
    """把 decision_engine 回傳的語言中立代碼轉成目前語言對應的文字，
    找不到對照時原樣顯示代碼（保底，不會整個消失）。"""
    return "、".join(t(f"reason_{code}") if f"reason_{code}" in TEXT else code for code in codes)


@st.cache_data(ttl=3600, persist="disk")
def _cached_run_decision_system():
    """
    包一層 Streamlit 快取，避免使用者每次跟頁面互動都要重新打資料庫、
    重新算全部163檔股票的完整決策流程。快取1小時。
    persist="disk"：改存到硬碟，重啟 Streamlit 程式也不會清空快取。
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
watch = [d for d in decisions if d.signal == "WATCH"]
avoid_fundamental = [d for d in decisions if d.signal == "AVOID" and d.avoid_type == "fundamental_landmine"]
avoid_double = [d for d in decisions if d.signal == "AVOID" and d.avoid_type == "double_fail"]

BADGE_STYLE = (
    "display:inline-block;font-size:12px;padding:3px 10px;border-radius:6px;font-weight:500;"
)
BADGE_COLORS = {
    "buy_now": ("#E8F5E9", "#2E7D32"),
    "watch": ("#E3F2FD", "#1565C0"),
    "avoid_fundamental": ("#FFF3E0", "#EF6C00"),
    "avoid_double": ("#FFEBEE", "#C62828"),
}


def render_badge(label: str, color_key: str, count: int):
    bg, fg = BADGE_COLORS[color_key]
    st.markdown(
        f"<span style='{BADGE_STYLE}background:{bg};color:{fg}'>{label}</span>"
        f"&nbsp;&nbsp;<span style='font-size:13px;color:#666'>{t('unit_stocks', n=count)}</span>",
        unsafe_allow_html=True,
    )


def format_score_breakdown(d) -> str:
    """依 sub_scores 跟 scoring.TECH_WEIGHTS 動態組出「股票分數計算權重」這行完整明細。"""
    if not d.sub_scores:
        return t("score_breakdown_unavailable")
    parts = []
    for key, weight in scoring.TECH_WEIGHTS.items():
        value = d.sub_scores.get(key)
        value_str = f"{value:.1f}" if value is not None else "-"
        parts.append(f"{key}={value_str}({weight*100:.0f}%)")
    return t("score_breakdown_prefix") + "、".join(parts) + t("score_breakdown_suffix", total=d.stock_score)


def render_search_result(d):
    """依股票目前的分類，渲染出對應的完整詳細卡片。"""
    if d.signal == "BUY_NOW":
        render_badge(t("signal_buy_now"), "buy_now", 1)
    elif d.signal == "WATCH":
        render_badge(t("signal_watch"), "watch", 1)
    elif d.avoid_type == "fundamental_landmine":
        render_badge(t("signal_avoid_fundamental"), "avoid_fundamental", 1)
    else:
        render_badge(t("signal_avoid_double"), "avoid_double", 1)

    detail = {
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_close_price"): format_price(d.current_price),
    }
    if d.signal in ("BUY_NOW", "WATCH"):
        detail[t("col_fair_value")] = format_price(d.fair_value_estimate)
    if d.signal == "WATCH":
        detail[t("col_missing")] = translate_reasons(d.missing_conditions)
    elif d.signal == "AVOID":
        detail[t("col_fail_reason")] = translate_reasons(d.fundamentals_fail_reasons)

    st.dataframe(pd.DataFrame([detail]), use_container_width=True, hide_index=True)
    st.caption(format_score_breakdown(d))


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

# ---- BUY_NOW：買進 ----
render_badge(t("signal_buy_now"), "buy_now", len(buy_now))
if buy_now:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_close_price"): format_price(d.current_price),
        t("col_fair_value"): format_price(d.fair_value_estimate),
    } for d in buy_now]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
for line in t("explain_buy_now"):
    st.caption(f"• {line}")

st.markdown("---")

# ---- WATCH：觀察 ----
render_badge(t("signal_watch"), "watch", len(watch))
if watch:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_close_price"): format_price(d.current_price),
        t("col_fair_value"): format_price(d.fair_value_estimate),
        t("col_missing"): translate_reasons(d.missing_conditions),
    } for d in watch]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
for line in t("explain_watch"):
    st.caption(f"• {line}")

st.markdown("---")

# ---- AVOID：避免（基本面地雷）----
render_badge(t("signal_avoid_fundamental"), "avoid_fundamental", len(avoid_fundamental))
if avoid_fundamental:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_close_price"): format_price(d.current_price),
        t("col_fail_reason"): translate_reasons(d.fundamentals_fail_reasons),
    } for d in avoid_fundamental]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ---- AVOID：避免（雙重不合）----
render_badge(t("signal_avoid_double"), "avoid_double", len(avoid_double))
if avoid_double:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_close_price"): format_price(d.current_price),
        t("col_fail_reason"): translate_reasons(d.fundamentals_fail_reasons),
    } for d in avoid_double]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
for line in t("explain_avoid"):
    st.caption(f"• {line}")

st.markdown("---")

# ---- 股票分數對應訊號說明 ----
st.subheader(t("signal_table_title"))
ref_rows = [
    {t("col_score_range"): "≥ 85", t("col_signal"): t("signal_buy_now"), t("col_condition"): t("cond_buy_now")},
    {t("col_score_range"): "< 85", t("col_signal"): t("signal_watch"), t("col_condition"): t("cond_watch")},
    {t("col_score_range"): "≥ 85", t("col_signal"): t("signal_avoid_fundamental"), t("col_condition"): t("cond_avoid_fundamental")},
    {t("col_score_range"): "< 85", t("col_signal"): t("signal_avoid_double"), t("col_condition"): t("cond_avoid_double")},
]
st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
st.caption(t("weights_note"))

st.sidebar.markdown("---")
st.sidebar.caption("A project by I.H. Wang")
