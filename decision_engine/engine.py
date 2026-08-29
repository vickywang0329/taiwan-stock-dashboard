"""
decision_engine/engine.py
----------------------------
Decision Engine：把 Stock Score / Entry Score / 風險報酬比 / 乖離幅度
整合成最終訊號（Final Signal）。

四級訊號（對照交接摘要）：
  股票分數 <65                              -> AVOID   避免
  65-84                                     -> WATCH   觀察中
  >=85 且 進場分數/風險報酬比/乖離幅度同時達標  -> BUY_NOW      明天可買
  >=85 但未同時達標                          -> BUY_PULLBACK  等回檔

⚠️ 下面 THRESHOLDS 一樣是初始假設，之後用回測校準。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from . import scoring, zones, risk_reward

THRESHOLDS = {
    "avoid_below": 65,
    "watch_below": 85,
}

SIGNAL_LABELS = {
    "BUY_NOW": "明天可買",
    "BUY_PULLBACK": "等回檔",
    "WATCH": "觀察中",
    "AVOID": "避免",
}


@dataclass
class StockDecision:
    stock_id: str
    name: str
    stock_score: float
    signal: str
    name_en: str | None = None  # 英文名稱，供頁面依語言切換顯示（decision_engine 本身不判斷語言）
    entry_zone: tuple[float, float] | None = None
    pullback_zones: dict | None = None
    pullback_position: str | None = None  # 現價實際落在哪個回檔位置，見 zones.classify_pullback_position()
    fair_value_estimate: float | None = None  # 依基本面估算的合理價格＝估算全年EPS×產業同業本益比基準
    stop_loss: float | None = None
    target: float | None = None
    current_price: float | None = None
    missing_conditions: list[str] = field(default_factory=list)  # WATCH 用：尚缺條件
    exclude_reason: list[str] = field(default_factory=list)      # AVOID 用：剔除原因


def _missing_conditions_for_watch(sub_scores: dict, institutional_ok: bool, breakout_ok: bool) -> list[str]:
    """給觀察中的股票列出「尚缺條件」，方便使用者知道還差什麼。"""
    reasons = []
    if not institutional_ok:
        reasons.append("法人未同步買超")
    if not breakout_ok:
        reasons.append("量能未突破")
    if sub_scores.get("trend", 100) < 60:
        reasons.append("技術趨勢偏弱")
    if sub_scores.get("relative_strength", 100) < 50:
        reasons.append("相對大盤強度不足")
    return reasons or ["綜合條件尚未齊全"]


def _exclude_reason_for_avoid(close: float, ma60: float, is_false_breakout: bool) -> list[str]:
    """給避免的股票列出「剔除原因」。"""
    reasons = []
    if close < ma60:
        reasons.append("收盤價跌破 MA60")
    if is_false_breakout:
        reasons.append("假突破訊號")
    return reasons or ["綜合品質分數不足"]


def decide(
    stock_id: str,
    name: str,
    close: float,
    high_20d: "pd.Series",
    low_20d: "pd.Series",
    prior_swing_high: float,
    atr14: float,
    ma60: float,
    sub_scores: dict,
    institutional_ok: bool,
    is_false_breakout: bool,
    is_overvalued: bool = False,
    fair_value_estimate: float | None = None,
    eps_growing: bool = True,
) -> StockDecision:
    """
    單一股票的完整決策流程。
    high_20d / low_20d 為「近20個交易日不含當日」的序列，用來算突破價與停損。

    BUY_NOW 判斷邏輯（已與使用者確認定案，拿掉風險報酬比門檻——2倍風險目標價公式
    數學上永遠不可能超過2.0，而門檻卻設2.5，兩者矛盾，故改用清楚的布林條件，
    entry_score/rr_ratio 因不再影響訊號判斷，已完全移除，不再計算）：

        BUY_NOW ⟺ stock_score>=85
                  AND NOT is_overvalued        （估值合理：本益比 <= 同業基準 x 1.5）
                  AND breakout_ok               （技術面：收盤價 >= 突破價）
                  AND bias_ok                   （技術面：乖離幅度合理，未追高）
                  AND institutional_ok          （籌碼面：近3日法人合計買超為正）
                  AND eps_growing               （基本面：估算全年EPS優於去年全年EPS）

    is_overvalued：本益比是否明顯偏離同業基準（> 1.5倍）。
    eps_growing：估算全年EPS 是否較去年全年EPS 成長。
    fair_value_estimate：依基本面估算的合理價格，直接附在結果上供頁面顯示。
    """
    stock_score = scoring.compute_stock_score(sub_scores)

    if stock_score < THRESHOLDS["avoid_below"]:
        return StockDecision(
            stock_id=stock_id,
            name=name,
            stock_score=stock_score,
            signal="AVOID",
            current_price=close,
            fair_value_estimate=fair_value_estimate,
            exclude_reason=_exclude_reason_for_avoid(close, ma60, is_false_breakout),
        )

    breakout_price = zones.compute_breakout_price(high_20d)
    breakout_ok = close >= breakout_price

    if stock_score < THRESHOLDS["watch_below"]:
        return StockDecision(
            stock_id=stock_id,
            name=name,
            stock_score=stock_score,
            signal="WATCH",
            current_price=close,
            fair_value_estimate=fair_value_estimate,
            missing_conditions=_missing_conditions_for_watch(sub_scores, institutional_ok, breakout_ok),
        )

    # stock_score >= 85：判斷 BUY_NOW 五項條件（entry_score/rr_ratio 已於
    # 使用者確認後移除，不再計算——原因見上方函式說明的數學矛盾）
    stop_loss = risk_reward.compute_stop_loss(low_20d)
    target = risk_reward.compute_target(close, stop_loss, prior_swing_high)
    bias_ok = scoring.bias_ok(close, breakout_price, atr14)

    all_ok = (
        not is_overvalued
        and breakout_ok
        and bias_ok
        and institutional_ok
        and eps_growing
    )

    if all_ok:
        entry_zone = zones.compute_entry_zone(close, breakout_price, atr14)
        return StockDecision(
            stock_id=stock_id,
            name=name,
            stock_score=stock_score,
            signal="BUY_NOW",
            entry_zone=entry_zone,
            fair_value_estimate=fair_value_estimate,
            stop_loss=stop_loss,
            target=target,
            current_price=close,
        )
    else:
        pullback = zones.compute_pullback_zones(breakout_price, atr14)
        pullback_position = zones.classify_pullback_position(close, pullback)
        return StockDecision(
            stock_id=stock_id,
            name=name,
            stock_score=stock_score,
            signal="BUY_PULLBACK",
            pullback_zones=pullback,
            pullback_position=pullback_position,
            fair_value_estimate=fair_value_estimate,
            stop_loss=stop_loss,
            target=target,
            current_price=close,
        )
