"""
decision_engine/engine.py
----------------------------
Decision Engine：把「技術籌碼分數(Stock Score)」跟「基本面守門檢查」
整合成最終訊號（Final Signal）。

⚠️ 2026/08 架構簡化記錄：使用者確認徹底拿掉「技術面型態明確」跟
「風險報酬比≥2.0」這兩個條件——不只是不讓它們影響訊號判斷，是連
pattern.py 整條計算路徑都從決策流程裡移除，不再附掛任何「純參考資訊」
欄位，避免架構裡留著沒有實際作用的殘骸。訊號判斷現在只看兩件事：

  1. 基本面守門檢查（見 check_fundamentals()）
     - 景氣循環股：估值(P/B)未過虛 + EPS優於去年（2項）
     - 非景氣循環股：估值(P/E)未過虛 + EPS優於去年 + 毛利率沒有嚴重惡化（3項）
  2. 技術籌碼分數（Stock Score，僅由 trend/momentum/relative_strength/
     institutional_flow/sector_flow 五項加權，見 scoring.TECH_WEIGHTS）

三級訊號：
  基本面未通過                       -> AVOID（再分兩種子類型，見 avoid_type）
  基本面通過 且 股票分數<85           -> WATCH   觀察
  基本面通過 且 股票分數>=85          -> BUY_NOW 買進

⚠️ 下面 THRESHOLDS 一樣是初始假設，之後用回測校準。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from . import scoring

THRESHOLDS = {
    "tech_score_buy_now": 85,  # 技術籌碼分數要達到這個門檻，才能進 BUY_NOW
}

SIGNAL_LABELS = {
    "BUY_NOW": "買進",
    "WATCH": "觀察",
    "AVOID": "避免",
}

AVOID_TYPE_LABELS = {
    "fundamental_landmine": "基本面地雷",   # 技術籌碼分數尚可，但基本面未過關
    "double_fail": "雙重不合",              # 技術籌碼分數也不夠、基本面也沒過
}


@dataclass
class StockDecision:
    stock_id: str
    name: str
    stock_score: float  # ＝技術籌碼分數（tech_score），不含任何基本面成分
    signal: str          # "BUY_NOW" / "WATCH" / "AVOID"
    name_en: str | None = None
    avoid_type: str | None = None  # signal="AVOID" 時才有值："fundamental_landmine" 或 "double_fail"
    fundamentals_pass: bool | None = None
    fundamentals_fail_reasons: list[str] = field(default_factory=list)
    fair_value_estimate: float | None = None
    sub_scores: dict | None = None  # 五項技術籌碼子分數，供頁面完整揭露計算成分
    current_price: float | None = None  # 個股單日收盤價，三種分類都會填入，供頁面統一顯示
    missing_conditions: list[str] = field(default_factory=list)  # WATCH 用：尚缺條件


def check_fundamentals(
    is_overvalued: bool, eps_growing: bool, margin_severely_declining: bool,
    is_cyclical: bool,
) -> tuple[bool, list[str]]:
    """
    基本面守門檢查：
      估值未過虛 AND EPS優於去年 AND (景氣循環股豁免 OR 毛利率沒有明顯衰退)
    回傳 (是否通過, 沒通過的具體原因代碼清單)。

    毛利率是否「明顯衰退」用相對衰退幅度判斷（相對去年同期衰退超過8%），
    不是絕對百分點——見 valuation.is_margin_severely_declining()。

    景氣循環股只檢查前兩項（豁免毛利率檢查）——理由跟 margin_trend
    權重歸零一致：循環股毛利率暴跌可能只是商品價格循環造成，不是公司
    自己出了什麼特有問題，不該跟一般公司用同一套邏輯懲罰。
    """
    reasons = []
    if is_overvalued:
        reasons.append("overvalued")
    if not eps_growing:
        reasons.append("eps_declining")
    if not is_cyclical and margin_severely_declining:
        reasons.append("margin_severely_declining")
    return (len(reasons) == 0), reasons


def _missing_conditions_for_watch(sub_scores: dict) -> list[str]:
    """
    給觀察中的股票列出「尚缺條件」。新架構下，決定 WATCH 的唯一原因就是
    「技術籌碼分數未達85分」，這裡額外拆解五項子分數裡哪幾項偏弱（<50分），
    幫助使用者知道具體是哪個環節拖累了分數，不是只給一句籠統的說明。
    """
    reasons = ["tech_score_not_high_enough"]
    for key in ("trend", "momentum", "relative_strength", "institutional_flow", "sector_flow"):
        if sub_scores.get(key, 100) < 50:
            reasons.append(f"{key}_weak")
    return reasons


def decide(
    stock_id: str,
    name: str,
    close: float,
    sub_scores: dict,
    is_overvalued: bool,
    eps_growing: bool,
    margin_severely_declining: bool,
    is_cyclical: bool,
    fair_value_estimate: float | None = None,
) -> StockDecision:
    """
    單一股票的完整決策流程。訊號只由「基本面守門」+「技術籌碼分數」
    兩個維度決定，見模組頂端說明，不涉及任何技術面型態或風險報酬比判斷。
    """
    tech_score = scoring.compute_tech_score(sub_scores)

    fundamentals_pass, fail_reasons = check_fundamentals(
        is_overvalued, eps_growing, margin_severely_declining, is_cyclical,
    )

    if not fundamentals_pass:
        avoid_type = "fundamental_landmine" if tech_score >= THRESHOLDS["tech_score_buy_now"] else "double_fail"
        return StockDecision(
            stock_id=stock_id,
            name=name,
            stock_score=tech_score,
            signal="AVOID",
            avoid_type=avoid_type,
            fundamentals_pass=False,
            fundamentals_fail_reasons=fail_reasons,
            fair_value_estimate=fair_value_estimate,
            sub_scores=sub_scores,
            current_price=close,
        )

    if tech_score >= THRESHOLDS["tech_score_buy_now"]:
        return StockDecision(
            stock_id=stock_id,
            name=name,
            stock_score=tech_score,
            signal="BUY_NOW",
            fundamentals_pass=True,
            fair_value_estimate=fair_value_estimate,
            sub_scores=sub_scores,
            current_price=close,
        )

    return StockDecision(
        stock_id=stock_id,
        name=name,
        stock_score=tech_score,
        signal="WATCH",
        fundamentals_pass=True,
        fair_value_estimate=fair_value_estimate,
        sub_scores=sub_scores,
        current_price=close,
        missing_conditions=_missing_conditions_for_watch(sub_scores),
    )
