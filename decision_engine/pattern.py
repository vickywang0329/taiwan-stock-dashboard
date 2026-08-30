"""
decision_engine/pattern.py
----------------------------
技術面型態分類（突破型 / 回測型 / 中立），以及對應的進場價、停損價、
目標價、風險報酬比計算。取代原本用固定ATR倍數＋固定2倍風險回推目標價
的僵化公式。

⚠️ volume_ratio 門檻（1.5倍/0.8倍）、回測容忍區間（±1.5%）、停損ATR倍數
（0.5倍/1.5倍）、RR門檻（2.0）都是使用者依交易經驗訂出的初始假設，
之後要用回測校準。

已與使用者確認定案：
1. 型態分類：
   - 突破型：收盤價 >= 近20日高點(不含當日) 且 volume_ratio >= 1.5
   - 回測型：股價在月線(MA20)附近（±1.5%容忍區間）且 volume_ratio <= 0.8
   - 都不符合 → 中立（沒有清楚的技術面型態，不給進場價）
2. 進場價與停損價：
   - 突破型：進場價 = 收盤價；停損價 = 當日最低價 − 0.5倍ATR14
   - 回測型：進場價 = MA20；停損價 = MA20 − 1.5倍ATR14
3. 目標價（兩種型態共用）= 近60日最高價(prior_swing_high)，
   當作「空間檢查」：如果目標價離進場價太近（代表上方套牢區近、
   空間不足），算出來的RR會偏低，被步驟四正確擋下——這解決了原本
   「突破型目標價用固定倍數回推、RR永遠等於固定值、驗證形同虛設」
   的邏輯漏洞。
4. 驗證關卡：型態必須明確（不是中立）且 RR >= 2.0，才算「可執行」的
   設定，否則不能進 BUY_NOW，降級為 BUY_PULLBACK。
"""
from __future__ import annotations
import pandas as pd

BREAKOUT_VOLUME_RATIO_MIN = 1.5   # ⚠️ 初始假設，之後回測校準
PULLBACK_VOLUME_RATIO_MAX = 0.8   # ⚠️ 初始假設，之後回測校準
PULLBACK_MA20_TOLERANCE = 0.015   # ±1.5%，⚠️ 初始假設，之後回測校準
BREAKOUT_STOPLOSS_ATR_MULT = 0.5  # ⚠️ 初始假設，之後回測校準
PULLBACK_STOPLOSS_ATR_MULT = 1.5  # ⚠️ 初始假設，之後回測校準
RR_MIN = 2.0                      # ⚠️ 初始假設，之後回測校準


def classify_pattern(close: float, breakout_price_20d: float, ma20: float | None,
                      volume_ratio: float | None) -> str:
    """
    回傳 "breakout"（突破型）、"pullback"（回測型）或 "neutral"（中立）。
    breakout_price_20d：近20日高點（不含當日），跟 zones.compute_breakout_price 同一份邏輯。
    """
    if volume_ratio is None or pd.isna(volume_ratio):
        volume_ratio = 0.0

    is_breakout = close >= breakout_price_20d and volume_ratio >= BREAKOUT_VOLUME_RATIO_MIN
    if is_breakout:
        return "breakout"

    if ma20 is not None and not pd.isna(ma20) and ma20 > 0:
        near_ma20 = abs(close - ma20) / ma20 <= PULLBACK_MA20_TOLERANCE
        is_pullback = near_ma20 and volume_ratio <= PULLBACK_VOLUME_RATIO_MAX
        if is_pullback:
            return "pullback"

    return "neutral"


def compute_entry_and_stop(
    pattern: str, close: float, today_low: float | None, ma20: float | None, atr14: float | None,
) -> tuple[float, float] | None:
    """
    依型態回傳 (進場價, 停損價)。pattern="neutral" 回傳 None（不給進場價）。
    """
    if atr14 is None or pd.isna(atr14):
        atr14 = 0.0

    if pattern == "breakout":
        if today_low is None or pd.isna(today_low):
            return None
        entry_price = close
        stop_loss = today_low - BREAKOUT_STOPLOSS_ATR_MULT * atr14
        return entry_price, stop_loss

    if pattern == "pullback":
        if ma20 is None or pd.isna(ma20):
            return None
        entry_price = ma20
        stop_loss = ma20 - PULLBACK_STOPLOSS_ATR_MULT * atr14
        return entry_price, stop_loss

    return None


def compute_target_and_rr(
    entry_price: float, stop_loss: float, prior_swing_high: float | None,
) -> tuple[float, float] | None:
    """
    目標價統一取 prior_swing_high（近期歷史高點，不含當日），當作空間檢查。
    回傳 (目標價, 風險報酬比)。風險<=0、缺少歷史高點資料、或目標價低於進場價
    （代表這段可抓到的歷史範圍裡，找不到比今天更高的壓力區——這在股價創新高
    時可能發生，此時空間檢查沒有意義，誠實回傳 None，不要硬算出負的風險報酬比
    誤導使用者），都回傳 None。
    """
    if prior_swing_high is None or pd.isna(prior_swing_high):
        return None
    risk = entry_price - stop_loss
    if risk <= 0:
        return None
    reward = prior_swing_high - entry_price
    if reward <= 0:
        return None
    rr = reward / risk
    return prior_swing_high, rr


def is_valid_setup(pattern: str, rr: float | None) -> bool:
    """步驟四：型態必須明確（不是中立）且 RR >= 2.0，才算「可執行」的設定。"""
    if pattern == "neutral":
        return False
    if rr is None or rr < RR_MIN:
        return False
    return True
