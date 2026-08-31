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
TREND_SENSITIVITY = 0.03  # ⚠️ 初始假設，之後用回測校準


def trend_score(close: float, ma5: float, ma20: float, ma60: float) -> float:
    """
    技術趨勢分數（漸進式 0-100，不是離散的0/25/50/75/100）：
    收盤價/MA5/MA20/MA60 四組關係，各自算出百分比差距，
    用S型函數映射成分數，四組取平均——跟 relative_strength_score /
    margin_trend_score / valuation_score 用同一套設計語言。

    每組差距為0（剛好持平）→ 50分（中性）；差距為正（多頭排列成立）
    → 分數越高於50分，差距越大分數越接近100；差距為負 → 分數越低於50分，
    差距越大分數越接近0。跟舊版離散給分最大的差異：能反映「多頭排列的
    強弱程度」，不會因為「剛好突破一點點」跟「大幅突破」給一樣的分數，
    也不會因為「些微跌破」就整項直接歸零。
    """
    if any(v in (None, 0) or pd.isna(v) for v in (ma5, ma20, ma60)):
        return 50.0

    gaps = [
        (close - ma5) / ma5,
        (ma5 - ma20) / ma20,
        (ma20 - ma60) / ma60,
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
