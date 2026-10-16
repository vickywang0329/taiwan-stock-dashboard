"""
decision_engine/scoring.py
----------------------------
股票分數（Stock Score）與進場分數（Entry Score）。

⚠️ 下面所有權重與門檻都是「初始假設」，不是回測驗證過的數字，
   之後要用歷史資料回測校準，不要當作定案 —— 對照交接摘要備註。
   全部集中寫在 WEIGHTS / THRESHOLDS，方便之後直接改數字。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .valuation import (  # noqa: F401  重新匯出，pipeline.py 統一從 scoring 呼叫
    valuation_score, margin_trend_score, pb_score, eps_growing_score, CYCLICAL_INDUSTRIES,
)

# ---------------------------------------------------------------------------
# ⚠️ 2026/08 架構調整：基本面（估值/EPS成長/毛利率趨勢）全面退出加權計分，
# 改當「守門員」（在 engine.py 裡用布林值把關，不參與分數加權）——原因：
# 一檔技術籌碼面極強的股票，過去常因為估值分數普通、毛利率分數普通，
# 被拖累到門檻之下，等於基本面的平庸表現「稀釋」了技術面的強烈訊號。
# 拆開後，Stock Score 完全反映「技術籌碼面值不值得留意」，基本面則負責
# 在最後一關擋掉「技術面雖強、但基本面已經有明顯警訊」的地雷股，兩者
# 分工清楚，不再互相稀釋。
#
# TECH_WEIGHTS 五項權重總和100%，注意這剛好等於本專案最初始（還沒加入
# 估值檢查之前）的原始權重比例——不是走回頭路，而是把基本面獨立出來後，
# 純技術籌碼部分回歸原始比例分配。
# ---------------------------------------------------------------------------
TECH_WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.20,
    "relative_strength": 0.20,
    "institutional_flow": 0.20,
    "sector_flow": 0.15,
}
assert abs(sum(TECH_WEIGHTS.values()) - 1.0) < 1e-9


def compute_tech_score(sub_scores: dict) -> float:
    """
    技術籌碼分數（Stock Score，新架構下的核心分數，不含任何基本面成分）。
    sub_scores 需含 keys: trend, momentum, relative_strength,
    institutional_flow, sector_flow（各為 0-100）。
    """
    total = sum(TECH_WEIGHTS[k] * sub_scores[k] for k in TECH_WEIGHTS)
    return round(float(np.clip(total, 0, 100)), 1)


# ---------------------------------------------------------------------------
# 股票分數權重（滿分 100，各子分數 0-100 後加權平均）
# ⚠️ 舊版（基本面仍混入加權分數）保留在下面，新架構的 engine.py 已改用
# 上方的 compute_tech_score()，這裡不再是主要判斷依據，僅保留供對照。
#
# 景氣循環股（航運/半導體－記憶體/塑膠石化及煉油/鋼鐵，見 valuation.py
# 的 CYCLICAL_INDUSTRIES）用不同的一套權重：
# - valuation 從10%降到5%（P/E失真，改用P/B，但P/B同樣不是萬能，保守降低權重）
# - margin_trend 直接歸零（循環股毛利率起伏主要是商品價格循環造成，不是
#   公司自己的營運效率變化，用「今年比去年」的邏輯容易誤判；改成「跟長期
#   循環週期平均比」需要更長的歷史資料深度，暫不投入這項工程）
# - 空出來的15%權重，分配給新增的 eps_growing_score（EPS成長，連續分數），
#   循環股投資的精髓常常就是抓住「剛從谷底往上爬」的轉折點，這個訊號
#   對循環股格外重要，故給予比非循環股更高的權重（15% vs 10%）
# ---------------------------------------------------------------------------
WEIGHTS_NON_CYCLICAL = {
    "trend": 0.20,
    "momentum": 0.16,
    "relative_strength": 0.16,
    "institutional_flow": 0.16,
    "sector_flow": 0.12,
    "valuation": 0.05,
    "margin_trend": 0.05,
    "eps_growing_score": 0.10,
}
WEIGHTS_CYCLICAL = {
    "trend": 0.20,
    "momentum": 0.16,
    "relative_strength": 0.16,
    "institutional_flow": 0.16,
    "sector_flow": 0.12,
    "valuation": 0.05,   # 景氣循環股這裡放的是 pb_score()，不是 valuation_score()
    "margin_trend": 0.0,
    "eps_growing_score": 0.15,
}
assert abs(sum(WEIGHTS_NON_CYCLICAL.values()) - 1.0) < 1e-9
assert abs(sum(WEIGHTS_CYCLICAL.values()) - 1.0) < 1e-9
assert set(WEIGHTS_NON_CYCLICAL.keys()) == set(WEIGHTS_CYCLICAL.keys())

# 保留舊名稱 WEIGHTS 指向非景氣循環股版本，避免其他還沒更新的呼叫端直接出錯
WEIGHTS = WEIGHTS_NON_CYCLICAL

# ⚠️ 2026/08 修正記錄：毛利率是否「嚴重惡化」的判斷，已從這裡的絕對分數門檻
# （MARGIN_SEVERE_DECLINE_THRESHOLD=30），改成 valuation.is_margin_severely_declining()
# 裡的「相對衰退幅度>8%」判斷——絕對百分點門檻對高毛利率產業（如半導體－IC設計）
# 天生不公平，已用真實資料驗證後改用相對百分比，全部非景氣循環股統一套用同一個
# 相對門檻，不用再逐一產業手動設定絕對數字。

RS_LOOKBACK_DAYS = 20          # 相對強度比較區間
INSTITUTIONAL_LOOKBACK_DAYS = 10  # 法人買賣超觀察區間


# ---------------------------------------------------------------------------
# 子分數
# ---------------------------------------------------------------------------
TREND_SENSITIVITY = 0.03  # 舊版（百分比差距）敏感度，僅在缺 ATR 資料時作為備援使用

# ⚠️ 2026/10 修正：均線差距改用 ATR14（平均真實波幅）正規化，不再用「佔股價的
# 百分比」。原因：同樣是「高於季線10%」，對低波動股（ATR約1.5%）是很強的訊號，
# 對高波動股（ATR約4%）只是日常波動；而且舊版用固定3%的敏感度，兩個長期差距
# （MA10-MA60、收盤-MA60）常常直接頂到接近100，強勢股之間分不出高下。
# 改成「差距 ÷ ATR」之後，不同波動度的股票用同一把尺衡量；長期差距另外用較寬的
# 敏感度（短期差距本來就小、長期差距本來就大，不能共用同一個刻度）。
# 四個數字依序對應：收盤-MA5、MA5-MA10、MA10-MA60、收盤-MA60，單位都是「幾個ATR」。
# （用台積電396天歷史資料校準過：新算法分數分布的中位數約68、95百分位約85，
# 跟舊版的73／84大致相當，但不再被長期差距推到頂。）
# ⚠️ 這四個敏感度都是初始假設，之後用回測校準。
TREND_ATR_SENSITIVITY = (0.75, 0.75, 2.0, 3.0)


def trend_score(close: float, ma5: float, ma10: float, ma60: float,
                atr14: float | None = None) -> float:
    """
    技術趨勢分數（漸進式 0-100）：
    收盤價/MA5/MA10/MA60 四組關係，各自算出差距，用S型函數映射成分數，四組取平均。

    有 ATR14 時：差距用 ATR 正規化（單位＝幾個ATR），見 TREND_ATR_SENSITIVITY 說明。
    沒有 ATR14（缺資料）時：退回舊版的百分比差距算法，不讓整項分數因缺資料而消失。

    每組差距為0（剛好持平）→ 50分（中性）；差距為正（多頭排列成立）→ 分數越高於50分；
    差距為負 → 分數越低於50分。

    ⚠️ 2026/09 記錄：中期均線用 MA10 取代 MA20，反應更快，代價是雜訊較多。
    """
    if any(v in (None, 0) or pd.isna(v) for v in (ma5, ma10, ma60)):
        return 50.0

    if atr14 is not None and not pd.isna(atr14) and atr14 > 0:
        raw_gaps = [close - ma5, ma5 - ma10, ma10 - ma60, close - ma60]
        scores = [
            100 / (1 + np.exp(-(g / atr14) / sens))
            for g, sens in zip(raw_gaps, TREND_ATR_SENSITIVITY)
        ]
    else:
        gaps = [
            (close - ma5) / ma5,
            (ma5 - ma10) / ma10,
            (ma10 - ma60) / ma60,
            (close - ma60) / ma60,
        ]
        scores = [100 / (1 + np.exp(-gap / TREND_SENSITIVITY)) for gap in gaps]
    return float(np.clip(sum(scores) / len(scores), 0, 100))


def momentum_score(rsi14: float, macd_hist: float) -> float:
    """RSI 落在強勢區間(50-80)給高分，MACD 柱狀圖為正加分。"""
    if pd.isna(rsi14):
        rsi_part = 50
    elif 50 <= rsi14 <= 80:
        rsi_part = 100
    elif rsi14 > 80:
        rsi_part = 70  # 過熱，稍微扣分
    else:
        rsi_part = max(0, rsi14 / 50 * 60)  # <50 分數隨 RSI 下降遞減

    macd_part = 100 if (macd_hist is not None and macd_hist > 0) else 40
    return 0.6 * rsi_part + 0.4 * macd_part


def relative_strength_score(stock_return: float, benchmark_return: float) -> float:
    """股票近 N 日報酬 - 大盤近 N 日報酬，超額報酬越高分數越高。"""
    excess = stock_return - benchmark_return
    # 用 sigmoid 型映射把超額報酬（約 -20%~+20%）壓進 0-100
    score = 100 / (1 + np.exp(-excess / 0.05 * 1.0))
    return float(np.clip(score, 0, 100))


def institutional_flow_score(net_buy_days: int, total_days: int, net_amount_trend_up: bool) -> float:
    """近N日法人買超天數比例 + 買超金額是否呈上升趨勢。"""
    if total_days == 0:
        return 50
    day_ratio_score = 100 * net_buy_days / total_days
    trend_bonus = 10 if net_amount_trend_up else 0
    return float(np.clip(0.8 * day_ratio_score + trend_bonus, 0, 100))


def sector_flow_score(sector_rank_pct: float) -> float:
    """
    sector_rank_pct: 產業在資金輪動排名中的百分位（0=最強，1=最弱），
    直接沿用 sector_heatmap.py 既有的法人資金流向排名邏輯計算出來。
    """
    return float(np.clip(100 * (1 - sector_rank_pct), 0, 100))


# ---------------------------------------------------------------------------
# 股票分數（總分）
# ---------------------------------------------------------------------------
def compute_stock_score(sub_scores: dict, is_cyclical: bool = False) -> float:
    """
    sub_scores 需含 keys: trend, momentum, relative_strength,
    institutional_flow, sector_flow, valuation, margin_trend,
    eps_growing_score（各為 0-100）。
    is_cyclical=True 時改用 WEIGHTS_CYCLICAL（見上方說明）。
    """
    weights = WEIGHTS_CYCLICAL if is_cyclical else WEIGHTS_NON_CYCLICAL
    total = sum(weights[k] * sub_scores[k] for k in weights)
    return round(float(np.clip(total, 0, 100)), 1)


# ---------------------------------------------------------------------------
# 進場分數：衡量目前價格相對突破點的 ATR 標準化距離（乖離）
# ---------------------------------------------------------------------------
ENTRY_SCORE_SWEET_SPOT_ATR = 0.3  # 收盤在突破價 + 0.3 ATR 以內視為最佳進場區
ENTRY_SCORE_DECAY_ATR = 1.5       # 超過突破價 + 1.5 ATR，進場分數趨近 0


def compute_entry_score(close: float, breakout_price: float, atr14: float) -> float:
    """
    距離 = (close - breakout_price) / atr14
    距離 <= 0（尚未站上突破價）：進場分數隨距離越負越低（還沒到，先觀察）
    0 < 距離 <= sweet spot：接近滿分
    距離 > sweet spot：分數隨乖離幅度遞減（漲多、乖離過大）
    """
    if atr14 in (None, 0) or pd.isna(atr14):
        return 50.0
    distance = (close - breakout_price) / atr14

    if distance <= 0:
        # 還沒突破：距離 0 給 60 分，越負分數越低
        score = max(0, 60 + distance * 40)
    elif distance <= ENTRY_SCORE_SWEET_SPOT_ATR:
        score = 100 - (distance / ENTRY_SCORE_SWEET_SPOT_ATR) * 10  # 90~100
    else:
        # 超過甜蜜點後線性遞減到 DECAY_ATR 處歸零
        over = distance - ENTRY_SCORE_SWEET_SPOT_ATR
        span = ENTRY_SCORE_DECAY_ATR - ENTRY_SCORE_SWEET_SPOT_ATR
        score = max(0, 90 - 90 * (over / span))

    return round(float(np.clip(score, 0, 100)), 1)


def bias_ok(close: float, breakout_price: float, atr14: float) -> bool:
    """乖離幅度是否在可接受範圍內（決定 BUY_NOW vs BUY_PULLBACK 的條件之一）。"""
    if atr14 in (None, 0) or pd.isna(atr14):
        return False
    distance = (close - breakout_price) / atr14
    return 0 <= distance <= ENTRY_SCORE_SWEET_SPOT_ATR


# ---------------------------------------------------------------------------
# 動能新鮮度折扣（追高風險折扣）。兩個因子相乘：
#
# ⚠️ 2026/10 重新設計。舊版只看「連續站上MA10幾天」，有兩個問題：
#   1. 只看天數、不看漲幅：三天暴漲25%的股票天數很「新鮮」，完全不被折扣；
#      緩步墊高走了20天的股票卻被打到7折。真正的追高風險是「離均線多遠」。
#   2. 折扣最低7折，配上85分門檻，趨勢走超過20天的股票分數上限只剩70分，
#      永遠進不了買進——持續數月的長期龍頭（如台積電）因此幾乎不可能被選中。
#
# 新設計：
#   A. 乖離程度折扣（主要因子）：(收盤價 − MA10) ÷ ATR14，單位是「幾個ATR」。
#      ≤ EXT_FULL_ATR 不打折；≥ EXT_MAX_ATR 打到 EXT_MIN_MULTIPLIER，中間線性插值。
#      用 ATR 正規化，不同波動度的股票用同一把尺衡量。
#      （用台積電396天實際資料看過分布：乖離中位數0.6個ATR、90百分位2.3、
#      最大4.6，所以門檻2.0~4.5大約只會折扣最極端的約10%交易日。）
#   B. 持續天數折扣（次要因子，刻意放輕）：連續站上MA10的天數，最多只扣 5%。
#      只作為「趨勢已成熟」的輕微提示，不再單獨把長期趨勢股擋在買進門外。
#
# 長期龍頭只要沒有離均線太遠（乖離小），就只會被扣最多5%——原始分數夠高
# （約90分以上）仍然可以進入買進。
#
# ⚠️ 下面所有門檻、折扣係數都是初始假設，尚未回測驗證。
# ---------------------------------------------------------------------------
EXT_FULL_ATR = 2.0         # 乖離 ≤ 2個ATR：不打折
EXT_MAX_ATR = 4.5          # 乖離 ≥ 4.5個ATR：打到最低折扣
EXT_MIN_MULTIPLIER = 0.7   # 乖離折扣的最低係數

FRESHNESS_FULL_BONUS_DAYS = 10     # 連續站上均線10天以內：不打折
FRESHNESS_FULL_PENALTY_DAYS = 40   # 連續站上均線40天以上：打到最低（輕微）折扣
FRESHNESS_MIN_MULTIPLIER = 0.95    # 持續天數折扣最低係數：最多只扣5%


def days_above_ma(close_series: pd.Series, ma_series: pd.Series, max_lookback: int = 60) -> int:
    """
    計算收盤價連續站在均線之上的交易日數，從最新一天往回數，遇到跌破
    或缺資料就停止累計。用來判斷目前的多頭動能是「剛啟動」還是
    「已經延續很久」。

    close_series / ma_series 需要依日期由舊到新排序（跟 pipeline.py 裡
    hist DataFrame 的慣例一致）。
    """
    count = 0
    n = len(close_series)
    for i in range(n - 1, -1, -1):
        c, m = close_series.iloc[i], ma_series.iloc[i]
        if pd.isna(c) or pd.isna(m) or c <= m:
            break
        count += 1
        if count >= max_lookback:
            break
    return count


def momentum_freshness_multiplier(days_above_ma_count: int | None) -> float:
    """
    動能新鮮度折扣係數（1.0=完全不打折，FRESHNESS_MIN_MULTIPLIER=打到最深）。
    連續站上均線的天數，落在 FULL_BONUS 跟 FULL_PENALTY 兩個門檻之間時，
    用線性插值算出中間的折扣程度。
    """
    if days_above_ma_count is None:
        return 1.0
    if days_above_ma_count <= FRESHNESS_FULL_BONUS_DAYS:
        return 1.0
    if days_above_ma_count >= FRESHNESS_FULL_PENALTY_DAYS:
        return FRESHNESS_MIN_MULTIPLIER

    span = FRESHNESS_FULL_PENALTY_DAYS - FRESHNESS_FULL_BONUS_DAYS
    progress = (days_above_ma_count - FRESHNESS_FULL_BONUS_DAYS) / span
    return 1.0 - progress * (1.0 - FRESHNESS_MIN_MULTIPLIER)


def extension_multiplier(close: float, ma10: float, atr14: float) -> float:
    """
    乖離程度折扣係數：(收盤 − MA10) ÷ ATR14 = 距離均線幾個ATR。
    收盤在均線之下（乖離為負）或缺資料 → 1.0（不打折；弱勢已由趨勢分數反映）。
    """
    if any(v is None or pd.isna(v) for v in (close, ma10, atr14)) or atr14 <= 0:
        return 1.0
    ext = (close - ma10) / atr14
    if ext <= EXT_FULL_ATR:
        return 1.0
    if ext >= EXT_MAX_ATR:
        return EXT_MIN_MULTIPLIER
    progress = (ext - EXT_FULL_ATR) / (EXT_MAX_ATR - EXT_FULL_ATR)
    return 1.0 - progress * (1.0 - EXT_MIN_MULTIPLIER)


def combined_freshness_multiplier(days_above_ma_count: int | None, close: float,
                                  ma10: float, atr14: float) -> float:
    """最終折扣係數 = 乖離程度折扣 × 持續天數折扣（最低約 0.7 × 0.95 ≈ 0.665）。"""
    return extension_multiplier(close, ma10, atr14) * momentum_freshness_multiplier(days_above_ma_count)


# ---------------------------------------------------------------------------
# ⚠️ 2026/10 新增：大盤環境判斷（市場過濾）
#
# 波段交易的第一道過濾：大盤在空頭走勢時，個股的多頭訊號勝率會明顯下降。
# 用大盤基準（0050）的季線（MA60，台股習慣的多空分界）與月線（MA20）判斷：
#   bull（多頭）：收盤價 ≥ MA60
#   bear（空頭）：收盤價 < MA60，且 MA20 也在 MA60 之下（短中期趨勢都向下）
#   neutral（盤整／轉弱觀察）：收盤價 < MA60，但 MA20 仍在 MA60 之上（剛跌破，尚未確認）
# 空頭時，BUY_NOW 一律降為 WATCH（見 engine.decide）；其他狀態不影響訊號。
#
# 為什麼空頭要「收盤<MA60 且 MA20<MA60」兩個條件？用台積電396天資料做代理測試：
# 只用「收盤<MA20 且 <MA60」會在空頭狀態來回切換13次，雜訊太多；
# 加上「MA20<MA60」只切換3次，穩定很多。（⚠️ 只是單一股票代理測試，不是大盤實測。）
# ---------------------------------------------------------------------------
MARKET_MIN_ROWS = 60  # 至少要有60筆大盤收盤價才算得出MA60


def market_regime(bm_close: pd.Series) -> tuple[str, float | None]:
    """
    回傳 (環境狀態, 收盤價相對MA60的偏離百分比)。
    狀態："bull" / "neutral" / "bear" / "unknown"（資料不足，不啟用過濾）。
    bm_close 需依日期由舊到新排序。
    """
    if bm_close is None or len(bm_close) < MARKET_MIN_ROWS:
        return "unknown", None
    ma20 = bm_close.tail(20).mean()
    ma60 = bm_close.tail(60).mean()
    close = bm_close.iloc[-1]
    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma60) or ma60 <= 0:
        return "unknown", None
    gap_pct = float((close / ma60 - 1) * 100)
    if close >= ma60:
        return "bull", gap_pct
    if ma20 < ma60:
        return "bear", gap_pct
    return "neutral", gap_pct


# ---------------------------------------------------------------------------
# ⚠️ 2026/10 新增：成交量確認（量能折扣）
#
# 用「上漲日成交量 ÷ 下跌日成交量」（Up/Down Volume Ratio，IBD/CANSLIM 常用的
# 籌碼累積／出貨指標）衡量近20個交易日的量價配合：
#   ratio ≥ 1.0：上漲日量大於下跌日量（買盤主導），不打折
#   ratio 越低：下跌日量越大（出貨跡象），折扣越深，ratio ≤ 0.6 打到最低
# 跟動能新鮮度折扣一樣，是乘在五項加權分數之後的係數，不新增第六個加權項目，
# 不改動使用者已確認的 TECH_WEIGHTS。
# 用台積電396天資料看過分布：中位數1.13、25百分位0.72、10百分位0.43。
# ⚠️ 下面的區間與最低係數都是初始假設，尚未回測驗證。
# ---------------------------------------------------------------------------
VOLUME_LOOKBACK_DAYS = 20
VOLUME_MIN_VALID_DAYS = 10     # 有效漲跌日數不足時，不下判斷（回傳1.0）
VOLUME_FULL_RATIO = 1.0        # ratio ≥ 1.0：不打折
VOLUME_MIN_RATIO = 0.6         # ratio ≤ 0.6：打到最低係數
VOLUME_MIN_MULTIPLIER = 0.88


def updown_volume_ratio(close: pd.Series, volume: pd.Series,
                        n: int = VOLUME_LOOKBACK_DAYS) -> float | None:
    """近 n 日「上漲日總量 ÷ 下跌日總量」。平盤日不計。資料不足回傳 None。
    下跌日量為0（全部上漲）時回傳 3.0（視為極強，上限值）。"""
    if close is None or volume is None or len(close) < 3:
        return None
    c = close.tail(n + 1).reset_index(drop=True)
    v = volume.tail(n + 1).reset_index(drop=True)
    chg = c.diff().iloc[1:]
    vol = v.iloc[1:]
    valid = (~chg.isna()) & (~vol.isna()) & (chg != 0)
    if int(valid.sum()) < VOLUME_MIN_VALID_DAYS:
        return None
    up = float(vol[valid & (chg > 0)].sum())
    down = float(vol[valid & (chg < 0)].sum())
    if down <= 0:
        return 3.0 if up > 0 else None
    return up / down


def volume_multiplier(ratio: float | None) -> float:
    """量能折扣係數（1.0＝不打折）。缺資料回傳 1.0（中性，不因缺資料受罰）。"""
    if ratio is None or pd.isna(ratio) or ratio >= VOLUME_FULL_RATIO:
        return 1.0
    if ratio <= VOLUME_MIN_RATIO:
        return VOLUME_MIN_MULTIPLIER
    progress = (VOLUME_FULL_RATIO - ratio) / (VOLUME_FULL_RATIO - VOLUME_MIN_RATIO)
    return 1.0 - progress * (1.0 - VOLUME_MIN_MULTIPLIER)
