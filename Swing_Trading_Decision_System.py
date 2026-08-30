"""
波段交易決策系統 Swing Trading Decision System
主入口頁面（放在 dashboard_project 根目錄，取代原本的 dashboard.py 作為主入口）
呼叫 decision_engine.pipeline.run_decision_system()，把每檔股票的
Stock Score / 技術面型態分類 / 風險報酬比整合成 BUY_NOW / BUY_PULLBACK / WATCH / AVOID 四級訊號。

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

    "signal_buy_now": {"zh": "明天可買", "en": "Buy now"},
    "signal_pullback": {"zh": "等回檔", "en": "Wait for pullback"},
    "signal_watch": {"zh": "觀察中", "en": "Watching"},
    "signal_avoid": {"zh": "避免", "en": "Avoid"},
    "unit_stocks": {"zh": "{n} 檔", "en": "{n} stocks"},

    "col_stock": {"zh": "股票", "en": "Stock"},
    "col_stock_score": {"zh": "股票分數", "en": "Stock Score"},
    "col_entry_price": {"zh": "進場價", "en": "Entry price"},
    "col_pattern_type": {"zh": "型態", "en": "Pattern"},
    "col_rr_ratio": {"zh": "風險報酬比", "en": "Risk/Reward"},
    "pattern_breakout": {"zh": "突破型", "en": "Breakout"},
    "pattern_pullback": {"zh": "回測型", "en": "Pullback"},
    "pattern_neutral": {"zh": "中立", "en": "Neutral"},
    "col_stop_target": {"zh": "技術面停損/目標", "en": "Technical stop/target"},
    "col_current_price": {"zh": "現價", "en": "Current price"},
    "col_zones": {"zh": "技術面進場價", "en": "Technical entry price"},
    "zones_format": {"zh": "{small}元以下小幅布局，{full}元以下大幅加碼",
                     "en": "Small position below {small}, full position below {full}"},
    "pullback_pattern_info": {"zh": "進場價{entry}元(風險報酬比{rr}，未達2.0門檻或其他條件未過)",
                              "en": "Entry {entry} (R/R {rr}, below the 2.0 threshold or other conditions not met)"},
    "col_fair_value": {"zh": "基本面合理價格", "en": "Fundamental fair value"},
    "col_missing": {"zh": "尚缺條件", "en": "Missing conditions"},
    "col_exclude_reason": {"zh": "剔除原因", "en": "Exclusion reason"},

    # 對應 decision_engine/engine.py 回傳的語言中立代碼
    "reason_institutional_not_buying": {"zh": "法人未同步買超", "en": "No institutional net buying"},
    "reason_volume_not_breakout": {"zh": "量能未突破", "en": "Volume hasn't confirmed breakout"},
    "reason_trend_weak": {"zh": "技術趨勢偏弱", "en": "Weak technical trend"},
    "reason_relative_strength_weak": {"zh": "相對大盤強度不足", "en": "Weak relative strength vs. market"},
    "reason_conditions_incomplete": {"zh": "綜合條件尚未齊全", "en": "Overall conditions not yet complete"},
    "reason_below_ma60": {"zh": "收盤價跌破 MA60", "en": "Close below MA60"},
    "reason_false_breakout": {"zh": "假突破訊號", "en": "False breakout signal"},
    "reason_quality_score_insufficient": {"zh": "綜合品質分數不足", "en": "Overall quality score insufficient"},

    "coverage_full": {"zh": "本次決策系統涵蓋 {n} / {total} 檔股票", "en": "This run covers {n} / {total} stocks"},
    "coverage_partial": {"zh": "⚠️ 本次決策系統只涵蓋 {n} / {total} 檔股票，以下股票因資料不足被跳過：",
                         "en": "⚠️ This run only covers {n} / {total} stocks. The following were skipped due to insufficient data:"},

    "search_label": {"zh": "🔍 搜尋股票代碼", "en": "🔍 Search by stock code"},
    "search_placeholder": {"zh": "輸入股票代碼，例如 2330", "en": "Enter stock code, e.g. 2330"},
    "search_result_title": {"zh": "查詢結果", "en": "Search result"},
    "search_not_in_watchlist": {"zh": "個股尚未新增至資料庫", "en": "This stock hasn't been added to the database yet"},
    "score_breakdown_prefix": {"zh": "股票分數計算權重：", "en": "Stock Score breakdown: "},
    "score_breakdown_suffix": {"zh": "，總計股票分數 {total:.1f} 分", "en": ", total Stock Score {total:.1f}"},
    "score_breakdown_unavailable": {"zh": "（無法取得子分數明細）", "en": "(sub-score breakdown unavailable)"},
    "search_insufficient_data": {"zh": "個股已在觀察池中，但目前資料不足，尚未列入本次計算結果",
                                 "en": "This stock is in the watchlist, but there isn't enough data yet to include it in this run"},

    "explain_buy_now": {
        "zh": [
            "股票分數：技術趨勢、動能、相對強度、法人動向、產業資金流向、估值與毛利率趨勢綜合計算，滿分100（權重明細見下方*）",
            "明天可買的判斷條件（五項須同時成立）：估值未過虛(本益比≤同業基準1.5倍)、收盤價站上突破價、技術面型態明確且風險報酬比≥2.0、近3日法人合計買超、估算全年EPS優於去年",
            "型態判定：突破型＝收盤價站上近20日高點且成交量達均量1.5倍以上；回測型＝股價貼近月線(MA20，容忍±1.5%)且成交量萎縮至均量0.8倍以下；兩者皆非則為中立，不給進場價",
            "進場價：突破型＝當日收盤價；回測型＝月線(MA20)",
            "技術面停損：突破型＝當日最低點－0.5倍ATR14；回測型＝月線－1.5倍ATR14",
            "技術面目標價：統一取近60個交易日最高價，作為「空間檢查」——若上方套牢區太近導致風險報酬比不足2.0，即使技術面突破也視為不值得進場",
            "基本面合理價格：優先以「今年上半年EPS×(去年全年EPS÷去年上半年EPS)」估算全年EPS（要求去年上半年、全年皆為正值獲利，且比例介於0.5~5倍之間），不符合則改用近四季(TTM)實際EPS合計；再乘以同產業(排除異常值)平均本益比得出",
        ],
        "en": [
            "Stock Score: weighted combination of trend, momentum, relative strength, institutional flow, sector flow, valuation and margin trend, out of 100 (weights below*)",
            "Buy-now criteria (all five must hold): valuation not overextended (P/E ≤ 1.5× industry benchmark), close above breakout price, clear technical pattern with risk/reward ≥ 2.0, positive 3-day institutional net buy, estimated full-year EPS above last year's",
            "Pattern classification: Breakout = close above the 20-day high with volume ≥1.5× average; Pullback = price near the 20-day MA (±1.5% tolerance) with volume shrinking to ≤0.8× average; otherwise Neutral (no entry price given)",
            "Entry price: Breakout = today's close; Pullback = MA20",
            "Technical stop loss: Breakout = today's low − 0.5×ATR14; Pullback = MA20 − 1.5×ATR14",
            "Technical target: uses the 60-day high as a \"space check\" — if the overhead resistance is too close (risk/reward below 2.0), the setup is rejected even if a technical breakout occurred",
            "Fundamental fair value: prioritizes estimating full-year EPS via H1 EPS × (last year's full-year EPS ÷ last year's H1 EPS) — requires last year's H1 and full-year figures to both be profitable, with the ratio between 0.5x-5x; otherwise falls back to trailing-twelve-months (TTM) actual EPS. Multiplied by the industry's average P/E (outliers excluded)",
        ],
    },
    "explain_pullback": {
        "zh": [
            "等回檔的意思：股票分數已達標(≥85)，但「明天可買」五項條件未全部成立（常見原因：估值過虛、技術面型態不明確或風險報酬比不足2.0、法人未同步買超、或EPS未成長），故建議等回檔",
            "若已偵測到明確型態（突破型/回測型）：進場價/停損/目標價計算方式同上方明天可買區塊，只是風險報酬比未達2.0或其他條件未過，欄位會標註風險報酬比數值供參考",
            "若型態中立（尚未形成明確突破或回測結構）：改顯示小幅布局／大幅加碼兩個價格門檻——突破價＋0.5倍ATR14以下可小幅布局，突破價－0.5倍ATR14以下且法人未持續賣超可大幅加碼",
            "基本面合理價格：計算方式同上方明天可買區塊",
        ],
        "en": [
            "What \"wait for pullback\" means: Stock Score already passes (≥85), but not all five buy-now conditions hold (common reasons: overextended valuation, unclear technical pattern or risk/reward below 2.0, no institutional buying, or EPS not growing) — so wait for a pullback instead",
            "If a clear pattern (breakout/pullback) was detected: entry/stop/target use the same calculation as the Buy Now section above, just with risk/reward below 2.0 or another condition unmet — the risk/reward value is shown for reference",
            "If the pattern is neutral (no confirmed breakout or pullback structure yet): shows the small-position / full-position price thresholds instead — below breakout price + 0.5×ATR14 for a small position, below breakout price − 0.5×ATR14 with no persistent institutional selling for a full position",
            "Fundamental fair value: same calculation as the Buy Now section above",
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
        "zh": "品質達標(≥85分)，且五項條件同時成立：估值合理、站上突破價、技術面型態明確且風險報酬比≥2.0、法人買超、EPS優於去年",
        "en": "Passes quality bar (≥85), and all five hold: valuation reasonable, above breakout price, clear technical pattern with risk/reward ≥ 2.0, institutional net buying, EPS above last year's",
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


def format_pattern_type(pattern_type: str | None) -> str:
    mapping = {
        "breakout": t("pattern_breakout"),
        "pullback": t("pattern_pullback"),
        "neutral": t("pattern_neutral"),
    }
    return mapping.get(pattern_type, "-")


def translate_reasons(codes: list[str]) -> str:
    """把 decision_engine 回傳的語言中立代碼（例如 'institutional_not_buying'）
    轉成目前語言對應的文字，找不到對照時原樣顯示代碼（保底，不會整個消失）。"""
    return "、".join(t(f"reason_{code}") if f"reason_{code}" in TEXT else code for code in codes)


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


def format_score_breakdown(d) -> str:
    """依 sub_scores 跟 scoring.WEIGHTS 動態組出「股票分數計算權重」這行完整明細。"""
    if not d.sub_scores:
        return t("score_breakdown_unavailable")
    parts = []
    for key, weight in scoring.WEIGHTS.items():
        value = d.sub_scores.get(key)
        value_str = f"{value:.1f}" if value is not None else "-"
        parts.append(f"{key}={value_str}({weight*100:.0f}%)")
    return t("score_breakdown_prefix") + "、".join(parts) + t("score_breakdown_suffix", total=d.stock_score)


def format_pullback_technical_info(d) -> str:
    """
    依 pattern_type 決定顯示內容：
    - 有明確型態（突破型/回測型）：顯示 pattern.py 算出來的進場價與風險報酬比，
      跟 BUY_NOW 表格用同一套演算法，維持系統一致性。
    - 型態中立：退回舊的 Zone1(小幅布局)/Zone2(大幅加碼) 區間指引。
    """
    if d.pattern_type and d.pattern_type != "neutral" and d.entry_price is not None:
        rr_str = f"{d.rr_ratio:.2f}" if d.rr_ratio is not None else "-"
        return t("pullback_pattern_info", entry=format_price(d.entry_price), rr=rr_str)
    return format_zones(d)


def render_search_result(d):
    """依股票目前的分類，渲染出跟該分類表格對應的完整詳細卡片。"""
    label_key, color_key = SIGNAL_BADGE_KEY[d.signal]
    render_badge(t(label_key), color_key, 1)

    detail = {t("col_stock"): get_name(d), t("col_stock_score"): d.stock_score}
    if d.signal == "BUY_NOW":
        detail[t("col_pattern_type")] = format_pattern_type(d.pattern_type)
        detail[t("col_entry_price")] = format_price(d.entry_price)
        detail[t("col_rr_ratio")] = f"{d.rr_ratio:.2f}" if d.rr_ratio is not None else "-"
        detail[t("col_fair_value")] = format_price(d.fair_value_estimate)
        detail[t("col_stop_target")] = f"{format_price(d.stop_loss)} / {format_price(d.target)}"
    elif d.signal == "BUY_PULLBACK":
        detail[t("col_current_price")] = format_price(d.current_price)
        detail[t("col_pattern_type")] = format_pattern_type(d.pattern_type)
        detail[t("col_fair_value")] = format_price(d.fair_value_estimate)
        detail[t("col_zones")] = format_pullback_technical_info(d)
        detail[t("col_stop_target")] = f"{format_price(d.stop_loss)} / {format_price(d.target)}"
    elif d.signal == "WATCH":
        detail[t("col_missing")] = translate_reasons(d.missing_conditions)
    elif d.signal == "AVOID":
        detail[t("col_exclude_reason")] = translate_reasons(d.exclude_reason)

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

# ---- BUY_NOW：明天可買 ----
render_badge(t("signal_buy_now"), "buy_now", len(buy_now))
if buy_now:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_stock_score"): d.stock_score,
        t("col_pattern_type"): format_pattern_type(d.pattern_type),
        t("col_entry_price"): format_price(d.entry_price),
        t("col_rr_ratio"): f"{d.rr_ratio:.2f}" if d.rr_ratio is not None else "-",
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
        t("col_pattern_type"): format_pattern_type(d.pattern_type),
        t("col_fair_value"): format_price(d.fair_value_estimate),
        t("col_zones"): format_pullback_technical_info(d),
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
        t("col_missing"): translate_reasons(d.missing_conditions),
    } for d in watch]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ---- AVOID：避免 ----
render_badge(t("signal_avoid"), "avoid", len(avoid))
if avoid:
    rows = [{
        t("col_stock"): get_name(d),
        t("col_exclude_reason"): translate_reasons(d.exclude_reason),
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
