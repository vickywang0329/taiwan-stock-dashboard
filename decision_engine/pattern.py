"""
decision_engine/pattern.py
----------------------------
技術面型態分類（突破型 / 回測型 / 中立），以及對應的進場價、停損價、
目標價、風險報酬比計算。取代原本用固定ATR倍數＋固定2倍風險回推目標價
的僵化公式。

⚠️ volume_ratio 門檻（1.5倍/0.8倍）、拉回容忍區間（5~8%，先取6%）、
停損ATR倍數（0.5倍/1.5倍）、RR門檻（2.0）都是使用者依交易經驗訂出的
初始假設，之後要用回測校準。

⚠️ 2026/08 修正記錄：原本「回測型」定義成「股價貼近月線(MA20)±1.5%」，
實測發現：股票分數要拿到高分，本身就需要 trend 子分數也高，而 trend
分數高代表收盤價已經「大幅」站上均線，天生就很難同時貼近月線——導致
「股票分數高」跟「回測型」兩個條件互相排斥，全觀察池163檔裡，能同時
拿到高分又被分類成回測型的股票，實測結果是 0 檔。已改成「從近期高點
小幅拉回」的定義，改善這個結構性矛盾。

已與使用者確認定案：
1. 型態分類：
   - 突破型：收盤價 >= 近20日高點(不含當日) 且 volume_ratio >= 1.5
   - 回測型：收盤價從近20日高點小幅拉回(0~6%之間) 且 收盤價仍在月線之上
             (確認多頭結構未破壞) 且 volume_ratio <= 0.8（量縮，代表只是
             獲利了結，不是恐慌性賣壓）
   - 都不符合 → 中立（沒有清楚的技術面型態）
2. 進場價與停損價：
   - 突破型：進場價 = 收盤價；停損價 = 當日最低價 − 0.5倍ATR14
   - 回測型：進場價 = 收盤價（現在就可以進場，不是等未來某個價位）；
             停損價 = MA20 − 1.5倍ATR14（跌破月線視為多頭結構被破壞）
3. 目標價（兩種型態共用）= 近期歷史高點(prior_swing_high)，當作「空間檢查」：
   如果目標價離進場價太近，算出來的RR會偏低，被步驟四正確擋下。
4. 驗證關卡：型態必須明確（不是中立）且 RR >= 2.0，才算「可執行」的設定。
"""
from __future__ import annotations
import pandas as pd

BREAKOUT_VOLUME_RATIO_MIN = 1.5   # ⚠️ 初始假設，之後回測校準
PULLBACK_VOLUME_RATIO_MAX = 0.8   # ⚠️ 初始假設，之後回測校準
PULLBACK_FROM_HIGH_MAX = 0.06     # 從近期高點拉回幅度上限6%（使用者確認區間5~8%，先取中間值），⚠️ 初始假設
BREAKOUT_STOPLOSS_ATR_MULT = 0.5  # ⚠️ 初始假設，之後回測校準
PULLBACK_STOPLOSS_ATR_MULT = 1.5  # ⚠️ 初始假設，之後回測校準
RR_MIN = 2.0                      # ⚠️ 初始假設，之後回測校準


def classify_pattern(close: float, breakout_price_20d: float, ma20: float | None,
                      volume_ratio: float | None) -> str:
    """
    回傳 "breakout"（突破型）、"pullback"（回測型）或 "neutral"（中立）。
    breakout_price_20d：近20日高點（不含當日），跟 zones.compute_breakout_price 同一份邏輯。

    回測型（已重新定義）：從近20日高點小幅拉回（0~6%），且收盤價仍站在
    月線之上（多頭結構未破壞），且量縮——取代原本「貼近月線」的定義，
    避免跟「股票分數要高、trend子分數也要高」這個前提互相排斥。
    """
    if volume_ratio is None or pd.isna(volume_ratio):
        volume_ratio = 0.0

    is_breakout = close >= breakout_price_20d and volume_ratio >= BREAKOUT_VOLUME_RATIO_MIN
    if is_breakout:
        return "breakout"

    if breakout_price_20d and breakout_price_20d > 0:
        pullback_pct = (breakout_price_20d - close) / breakout_price_20d
        above_ma20 = ma20 is not None and not pd.isna(ma20) and close > ma20
        is_pullback = (
            0 < pullback_pct <= PULLBACK_FROM_HIGH_MAX
            and above_ma20
            and volume_ratio <= PULLBACK_VOLUME_RATIO_MAX
        )
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
        # 進場價改用收盤價（現在就可以進場），停損仍錨定月線
        # （跌破月線視為多頭結構被破壞，這點沿用原本設計）
        if ma20 is None or pd.isna(ma20):
            return None
        entry_price = close
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
    """型態必須明確（不是中立）且 RR >= 2.0，才算「可執行」的設定。"""
    if pattern == "neutral":
        return False
    if rr is None or rr < RR_MIN:
        return False
    return True
