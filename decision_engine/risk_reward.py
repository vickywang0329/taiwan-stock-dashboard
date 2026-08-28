"""
decision_engine/risk_reward.py
----------------------------------
停損、目標價、風險報酬比的計算。

公式對照交接摘要已確認的定案版本：
- 停損：近20個交易日（含當日）最低價
- 目標：進場價 + 2倍風險（2R），或前波高點（近60日最高），兩者取較保守者
- 風險報酬比：(目標價 - 進場價) / (進場價 - 停損價)

⚠️ 「2倍風險」這個倍數是初始假設，之後要用回測校準，不是定案的最佳值。
"""
from __future__ import annotations
import pandas as pd


def compute_stop_loss(low_20d: pd.Series) -> float:
    """停損 = 近20個交易日（含當日）的最低價"""
    return float(low_20d.min())


def compute_target(entry_price: float, stop_loss: float, prior_swing_high: float) -> float:
    """
    目標價 = entry + 2 * 風險(entry - stop)，或前波高點，兩者取較保守者（較低值）。
    如果前波高點不存在、或前波高點反而低於進場價（代表股價已經創新高、
    沒有參考意義），就只用 2R 目標。
    """
    risk = entry_price - stop_loss
    target_2r = entry_price + 2 * risk

    if prior_swing_high is None or pd.isna(prior_swing_high) or prior_swing_high <= entry_price:
        return round(target_2r, 2)

    return round(min(target_2r, prior_swing_high), 2)


def compute_rr_ratio(entry_price: float, stop_loss: float, target: float) -> float:
    """風險報酬比 = (目標價 - 進場價) / (進場價 - 停損價)"""
    risk = entry_price - stop_loss
    if risk <= 0:
        return 0.0
    reward = target - entry_price
    return round(reward / risk, 2)
