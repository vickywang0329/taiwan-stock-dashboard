"""
decision_engine/zones.py
----------------------------
突破價、進場區間、拉回買進 Zone1/Zone2 的計算。

公式對照交接摘要已確認的定案版本：
- 突破價 = 近20個交易日（不含當日）最高價
- 進場區間：下緣＝突破價，上緣＝當日收盤價 + 0.3倍ATR14
- Zone1（小幅布局）＝突破價 ± 0.5倍ATR14
- Zone2（大幅買進）＝突破價 − 1.5倍ATR14 ~ 突破價 − 0.5倍ATR14

⚠️ 0.3 / 0.5 / 1.5 這幾個倍數是初始假設，之後要用回測校準，不是定案的最佳值。
"""
from __future__ import annotations
import pandas as pd


def compute_breakout_price(high_20d: pd.Series) -> float:
    """突破價 = 近20個交易日（不含當日）的最高價"""
    return float(high_20d.max())


def compute_entry_zone(close: float, breakout_price: float, atr14: float) -> tuple[float, float]:
    """
    進場區間：下緣為突破價，上緣為當日收盤價 + 0.3倍ATR14
    （只有已經確認 BUY_NOW 訊號才會呼叫這個函式，此時 close 理論上已 >= breakout_price）
    """
    if atr14 is None or pd.isna(atr14):
        atr14 = 0.0
    lower = breakout_price
    upper = close + 0.3 * atr14
    # 防呆：避免異常資料造成上緣小於下緣
    if upper < lower:
        upper = lower
    return (round(lower, 2), round(upper, 2))


def compute_pullback_zones(breakout_price: float, atr14: float) -> dict:
    """
    Zone1（小幅布局）：突破價 ± 0.5倍ATR14
    Zone2（大幅買進）：突破價 − 1.5倍ATR14 ~ 突破價 − 0.5倍ATR14
    回傳格式：{"zone1": (low, high), "zone2": (low, high)}
    """
    if atr14 is None or pd.isna(atr14):
        atr14 = 0.0
    zone1_low = breakout_price - 0.5 * atr14
    zone1_high = breakout_price + 0.5 * atr14
    zone2_low = breakout_price - 1.5 * atr14
    zone2_high = breakout_price - 0.5 * atr14
    return {
        "zone1": (round(zone1_low, 2), round(zone1_high, 2)),
        "zone2": (round(zone2_low, 2), round(zone2_high, 2)),
    }


def classify_pullback_position(current_price: float, pullback_zones: dict) -> str:
    """
    拿「現價」去比對 Zone1/Zone2，判斷現在實際處於哪個狀態，
    給出可以直接行動的結論，而不是只丟兩個區間數字讓使用者自己比對。

    回傳四種狀態之一：
    - "above_zone1"：現價還高於 Zone1 上緣，還沒回檔到位，繼續等待
    - "in_zone1"：現價落在 Zone1 範圍內，可以小幅布局
    - "in_zone2"：現價落在 Zone2 範圍內，可以大幅買進
    - "below_zone2"：現價已經跌破 Zone2 下緣，代表跌破原本設定的回檔支撐，
      這種情況不建議依原計畫進場，應該重新評估（可能代表趨勢已轉弱）
    """
    z1_low, z1_high = pullback_zones["zone1"]
    z2_low, z2_high = pullback_zones["zone2"]

    if current_price > z1_high:
        return "above_zone1"
    if z1_low <= current_price <= z1_high:
        return "in_zone1"
    if z2_low <= current_price < z1_low:
        return "in_zone2"
    return "below_zone2"
