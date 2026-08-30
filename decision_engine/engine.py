"""
decision_engine/engine.py
----------------------------
Decision Engine：把 Stock Score / 技術面型態分類 / 風險報酬比
整合成最終訊號（Final Signal）。

四級訊號：
  股票分數 <65                              -> AVOID   避免
  65-84                                     -> WATCH   觀察中
  >=85 且 五項條件同時達標                    -> BUY_NOW      明天可買
  >=85 但未同時達標                          -> BUY_PULLBACK  等回檔

⚠️ 下面 THRESHOLDS 一樣是初始假設，之後用回測校準。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from . import scoring, zones, risk_reward, pattern

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
    entry_price: float | None = None  # 依型態分類算出的單一進場價（突破型=收盤價，回測型=MA20）
    pattern_type: str | None = None   # "breakout"（突破型）/ "pullback"（回測型）/ "neutral"（中立）
    rr_ratio: float | None = None     # 風險報酬比（目標價-進場價)/(進場價-停損價)，目標價統一取60日高點
    pullback_zones: dict | None = None
    pullback_position: str | None = None  # 現價實際落在哪個回檔位置，見 zones.classify_pullback_position()
    fair_value_estimate: float | None = None  # 依基本面估算的合理價格＝估算全年EPS×產業同業本益比基準
    sub_scores: dict | None = None  # 七項原始子分數（trend/momentum/.../margin_trend），供頁面完整揭露計算成分
    stop_loss: float | None = None
    target: float | None = None
    current_price: float | None = None
    missing_conditions: list[str] = field(default_factory=list)  # WATCH 用：尚缺條件
    exclude_reason: list[str] = field(default_factory=list)      # AVOID 用：剔除原因


def _missing_conditions_for_watch(sub_scores: dict, institutional_ok: bool, breakout_ok: bool) -> list[str]:
    """
    給觀察中的股票列出「尚缺條件」，方便使用者知道還差什麼。
    ⚠️ 回傳的是語言中立的代碼（不是中文字串），decision_engine 本身不判斷
    使用者介面語言——比照 name/name_en 分離的原則，翻譯交給前端的 t() 處理，
    這裡只負責產生穩定、可查表的代碼。
    """
    reasons = []
    if not institutional_ok:
        reasons.append("institutional_not_buying")
    if not breakout_ok:
        reasons.append("volume_not_breakout")
    if sub_scores.get("trend", 100) < 60:
        reasons.append("trend_weak")
    if sub_scores.get("relative_strength", 100) < 50:
        reasons.append("relative_strength_weak")
    return reasons or ["conditions_incomplete"]


def _exclude_reason_for_avoid(close: float, ma60: float, is_false_breakout: bool) -> list[str]:
    """給避免的股票列出「剔除原因」，同樣回傳語言中立的代碼。"""
    reasons = []
    if close < ma60:
        reasons.append("below_ma60")
    if is_false_breakout:
        reasons.append("false_breakout")
    return reasons or ["quality_score_insufficient"]


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
    ma20: float | None = None,
    volume_ratio: float | None = None,
    today_low: float | None = None,
) -> StockDecision:
    """
    單一股票的完整決策流程。
    high_20d / low_20d 為「近20個交易日不含當日」的序列，用來算突破價與停損。

    BUY_NOW 判斷邏輯（已與使用者確認定案）：

        BUY_NOW ⟺ stock_score>=85
                  AND NOT is_overvalued        （估值合理：本益比 <= 同業基準 x 1.5）
                  AND breakout_ok               （技術面：收盤價 >= 突破價）
                  AND setup_valid                （技術面：型態明確且RR>=2.0，見pattern.py）
                  AND institutional_ok          （籌碼面：近3日法人合計買超為正）
                  AND eps_growing               （基本面：估算全年EPS優於去年全年EPS）

    ⚠️ setup_valid 取代了原本的 bias_ok（乖離幅度是否合理），原因是
    pattern.py 的空間檢查機制（用60日高點回推RR）比原本武斷的0.3倍ATR
    窄幅範圍更嚴謹、更有數據支撐，兩者概念上都是在防止追高，功能重疊，
    故用更精進的機制取代，條件總數維持五項不變。

    is_overvalued：本益比是否明顯偏離同業基準（> 1.5倍）。
    eps_growing：估算全年EPS 是否較去年全年EPS 成長。
    fair_value_estimate：依基本面估算的合理價格，直接附在結果上供頁面顯示。
    ma20/volume_ratio/today_low：型態分類與新進場價/停損價計算需要的額外資料。
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
            sub_scores=sub_scores,
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
            sub_scores=sub_scores,
            missing_conditions=_missing_conditions_for_watch(sub_scores, institutional_ok, breakout_ok),
        )

    # stock_score >= 85：先做型態分類，再算進場價/停損/目標價/RR
    pattern_type = pattern.classify_pattern(close, breakout_price, ma20, volume_ratio)
    entry_and_stop = pattern.compute_entry_and_stop(pattern_type, close, today_low, ma20, atr14)

    if entry_and_stop is not None:
        entry_price, stop_loss = entry_and_stop
        target_and_rr = pattern.compute_target_and_rr(entry_price, stop_loss, prior_swing_high)
        target, rr = target_and_rr if target_and_rr is not None else (None, None)
    else:
        entry_price = stop_loss = target = rr = None

    setup_valid = pattern.is_valid_setup(pattern_type, rr)

    all_ok = (
        not is_overvalued
        and breakout_ok
        and setup_valid
        and institutional_ok
        and eps_growing
    )

    if all_ok:
        return StockDecision(
            stock_id=stock_id,
            name=name,
            stock_score=stock_score,
            signal="BUY_NOW",
            entry_price=entry_price,
            pattern_type=pattern_type,
            rr_ratio=rr,
            fair_value_estimate=fair_value_estimate,
            sub_scores=sub_scores,
            stop_loss=stop_loss,
            target=target,
            current_price=close,
        )
    else:
        if pattern_type != "neutral" and entry_price is not None:
            # 有偵測到明確型態（突破型或回測型），只是RR不足2.0、或其他條件
            # （估值/籌碼/基本面）沒過，仍然沿用同一套 pattern.py 演算法算出來的
            # 進場價/停損/目標價/RR，維持整套系統技術面判斷方式的一致性，
            # 不要切回另一套 Zone1/Zone2 機制——使用者才不會覺得「同樣講技術面，
            # 兩個地方算法卻不一樣」。
            return StockDecision(
                stock_id=stock_id,
                name=name,
                stock_score=stock_score,
                signal="BUY_PULLBACK",
                entry_price=entry_price,
                pattern_type=pattern_type,
                rr_ratio=rr,
                fair_value_estimate=fair_value_estimate,
                sub_scores=sub_scores,
                stop_loss=stop_loss,
                target=target,
                current_price=close,
            )
        else:
            # 型態中立——真的沒有明確技術結構可以算出精確進場價，
            # 這種情況才退回 Zone1/Zone2 這種「等回檔到某個區間」的指引，
            # 停損/目標改用 risk_reward.py 的滾動低點版本。
            pullback = zones.compute_pullback_zones(breakout_price, atr14)
            pullback_position = zones.classify_pullback_position(close, pullback)
            fallback_stop_loss = risk_reward.compute_stop_loss(low_20d)
            fallback_target = risk_reward.compute_target(close, fallback_stop_loss, prior_swing_high)
            return StockDecision(
                stock_id=stock_id,
                name=name,
                stock_score=stock_score,
                signal="BUY_PULLBACK",
                pattern_type=pattern_type,
                pullback_zones=pullback,
                pullback_position=pullback_position,
                fair_value_estimate=fair_value_estimate,
                sub_scores=sub_scores,
                stop_loss=fallback_stop_loss,
                target=fallback_target,
                current_price=close,
            )
