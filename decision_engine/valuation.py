"""
decision_engine/valuation.py
----------------------------
估值檢查：估算全年 EPS、算本益比、跟同產業基準比較，判斷是否「估值過虛」。

⚠️ 這裡的門檻（1.5倍同業平均）是初始假設，之後要用回測校準。

⚠️ 2026/08 修正記錄：原本誤以為 raw.eps_quarterly.eps_cumulative 是台股慣例的
「季累計數」，實際用大立光公開財報數字查證後發現，FinMind 這個欄位其實是「單季數」
（例如 2024Q2 存的 33.70 元，正好對應大立光公開法說會公布的「單季EPS 33.7元」，
不是「上半年累計 79.49元」）。原本的「還原單季」邏輯因此是對已經是單季的數字做了
錯誤的二次處理，導致 EPS 估算值系統性偏低、本益比被嚴重高估。已改為直接加總。

計算邏輯（已依正確資料格式修正）：
1. 全年 EPS 估算，優先用「季節調整外推法」：
   全年估算 = 今年上半年單季EPS加總 × (去年全年單季EPS加總 ÷ 去年上半年單季EPS加總)
   （財報空窗期，今年上半年資料還沒公布齊全時）備援改用 TTM（近四季單季EPS直接加總）
2. 本益比 = 現價 ÷ 估算全年EPS
3. 同產業基準 = 同產業其他股票本益比，排除虧損股（PE<=0）與極端值後的平均
4. 本益比 > 同業基準 × 1.5 倍 → 估值過虛（valuation 子分數不通過）
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _quarter_of(d: pd.Timestamp) -> int:
    """財報季底月份對應到第幾季：3月=Q1, 6月=Q2, 9月=Q3, 12月=Q4"""
    return {3: 1, 6: 2, 9: 3, 12: 4}.get(d.month, 0)


def _sum_standalone_eps(df: pd.DataFrame, year: int, quarters: list[int]) -> float | None:
    """
    加總指定年度、指定季度清單的單季EPS。
    df 的 eps_cumulative 欄位實際存的是「單季EPS」（已查證，見模組說明），
    要湊出「上半年」或「全年」，直接加總對應的單季數字即可，不需要還原。
    只要缺任何一季，就回傳 None，避免用不完整的資料湊出錯誤結果。
    """
    subset = df[(df["year"] == year) & (df["quarter"].isin(quarters))]
    if len(subset) != len(quarters):
        return None
    return float(subset["eps_cumulative"].sum())


def estimate_annual_eps(stock_eps_df: pd.DataFrame, today: pd.Timestamp | None = None) -> tuple[float | None, str]:
    """
    估算全年 EPS。回傳 (估算值, 使用的方法)，方法為 "extrapolation" 或 "ttm"，
    資料不足時回傳 (None, "insufficient_data")。
    """
    if today is None:
        today = pd.Timestamp.today()

    if stock_eps_df.empty:
        return None, "insufficient_data"

    df = stock_eps_df.sort_values("date").reset_index(drop=True).copy()
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].apply(_quarter_of)

    this_year, last_year = today.year, today.year - 1

    h1_this_val = _sum_standalone_eps(df, this_year, [1, 2])
    h1_last_val = _sum_standalone_eps(df, last_year, [1, 2])
    fy_last_val = _sum_standalone_eps(df, last_year, [1, 2, 3, 4])

    # ---- 優先：季節調整外推法 ----
    # ⚠️ 防呆：外推法的分母是「去年上半年EPS」，景氣循環股（如記憶體、PCB）
    # 常有某一期接近打平甚至虧損的情況，這時候除法會爆炸或正負號顛倒，
    # 算出離譜的合理價格（曾實測出現過負值、或十萬元等級的異常結果）。
    # 這裡要求：① 去年上半年、去年全年都必須是正值獲利（不是打平或虧損）
    #          ② 全年/上半年的比例要落在合理範圍內（0.5~5倍），
    #             超出這個範圍代表分母過小、比例不可靠，一律改用 TTM 備援。
    #          ③ 算出來的估算值本身也必須是正值——如果「今年上半年」本身
    #             是虧損，即使去年基準正常，外推結果一樣會是負的，這種
    #             情況在物理意義上（沒有負的股價）視為外推法不適用，改用TTM。
    extrapolation_was_loss = False
    if h1_this_val is not None and h1_last_val is not None and fy_last_val is not None:
        if h1_last_val > 0 and fy_last_val > 0:
            ratio = fy_last_val / h1_last_val
            if 0.5 <= ratio <= 5.0:
                estimate = h1_this_val * ratio
                if estimate > 0:
                    return estimate, "extrapolation"
                extrapolation_was_loss = True  # 外推法算出虧損，記錄下來，仍繼續嘗試TTM

    # ---- 備援：TTM（近四季單季EPS直接加總，財報空窗期、去年獲利不穩定
    #      或比例不合理時使用；因為原始資料本身就是單季數，不需要再還原）----
    # 同樣要求加總結果必須是正值——如果近四季實際上是虧損（TTM<=0），
    # 代表這家公司近期財務狀況不佳，本益比估值法在這種情況下得不到
    # 有意義的結果（不存在負的股價）。
    #
    # ⚠️ 這裡明確標記成 "loss"（虧損），跟 "insufficient_data"（真的缺資料）
    # 分開——兩者都會讓 estimated_annual_eps 回傳 None，但代表的意義完全不同：
    # 「虧損」是我們已經確實掌握、且應該給最差分數的負面資訊；
    # 「缺資料」則是真的不知道，不該懲罰。呼叫端(pipeline.py)要用 method
    # 這個回傳值去判斷該給哪一種待遇，不能只看 estimated_annual_eps 是不是 None。
    df_sorted = df.sort_values("date")
    recent_quarters = df_sorted.tail(4)
    if len(recent_quarters) == 4:
        ttm = float(recent_quarters["eps_cumulative"].sum())
        if ttm > 0:
            return ttm, "ttm"
        return None, "loss"

    if extrapolation_was_loss:
        return None, "loss"  # 沒有足夠資料算TTM，但外推法已經明確算出虧損

    return None, "insufficient_data"


def compute_pe(current_price: float, estimated_annual_eps: float | None) -> float | None:
    """本益比 = 現價 ÷ 估算全年EPS。EPS <= 0（虧損或估算失敗）視為無法計算，回傳 None。"""
    if estimated_annual_eps is None or estimated_annual_eps <= 0:
        return None
    return current_price / estimated_annual_eps


def get_last_year_full_year_eps(stock_eps_df: pd.DataFrame, today: pd.Timestamp | None = None) -> float | None:
    """取得「去年全年EPS」（四個單季加總），供判斷今年估算EPS是否較去年成長使用。"""
    if today is None:
        today = pd.Timestamp.today()
    if stock_eps_df.empty:
        return None

    df = stock_eps_df.copy()
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].apply(_quarter_of)

    last_year = today.year - 1
    return _sum_standalone_eps(df, last_year, [1, 2, 3, 4])


# ⚠️ 使用者手動指定的產業別本益比排除門檻（2026/08 定案，非回測結果，
# 之後若發現不合理可再調整）。PE 超過對應門檻視為異常值，不列入同業
# 平均本益比的計算。沒有列在這裡的產業（例如ETF），只排除虧損股，
# 不做上限排除。
INDUSTRY_PE_CUTOFF = {
    "PCB／載板／CCL": 30,
    "光學元件": 30,
    "光通訊": 60,
    "半導體－IC設計": 70,
    "半導體－封測": 40,
    "半導體－晶圓代工": 35,
    "半導體－記憶體": 30,
    "塑膠/石化及煉油": 15,
    "工業電腦/物聯網": 30,
    "散熱": 30,
    "測試儀器": 30,
    "精密機構件": 30,
    "網通設備": 30,
    "航運": 15,
    "被動元件": 30,
    "連接器/線纜": 30,
    "金控－壽險為主": 25,
    "金控－證券為主": 25,
    "金控－銀行": 25,
    "鋼鐵": 18,
    "電信服務商": 30,
    "電子代工": 20,
    "電源管理/重電": 35,
    "食品/多角化零售": 20,
}


def eps_growing(estimated_annual_eps: float | None, last_year_full_year_eps: float | None) -> bool:
    """
    今年估算全年EPS 是否較去年全年EPS 成長。
    缺資料時預設為 True（不因資料缺漏而卡關，跟 valuation 的中性原則一致）。
    """
    if estimated_annual_eps is None or last_year_full_year_eps is None:
        return True
    return estimated_annual_eps > last_year_full_year_eps


def compute_industry_pe_benchmark(pe_by_industry: pd.DataFrame) -> dict[str, float]:
    """
    pe_by_industry 需含 industry、pe 兩欄（每檔股票一列）。
    排除 PE<=0（虧損股），並用使用者手動指定的產業別門檻（INDUSTRY_PE_CUTOFF）
    排除異常值後，計算每個產業的平均本益比。

    ⚠️ 這裡改用使用者依照對各產業合理本益比範圍的主觀判斷手動指定的門檻，
    取代原本的 IQR（四分位距）統計方法——原因是 IQR 在樣本數小的產業
    (部分產業只有3-5檔股票)判斷不夠穩定，且只排除「過高」異常值、
    不排除「過低」異常值。手動門檻雖然主觀，但至少確定、可預期、
    容易逐一檢查是否合理。這些數字之後若有需要可以再調整。

    ⚠️ 備援規則：如果整個產業「所有股票的本益比都超過門檻」（篩選後
    一檔都不剩），不會讓這個產業的基準本益比整個算不出來（那樣會讓
    這個產業的估值檢查形同虛設，沒有任何一檔股票會被判定為過虛）。
    這種情況下，直接把「門檻值本身」當作這個產業的基準——邏輯上等於
    「連最便宜的一檔都比我認為合理的上限還貴，那就用這個上限本身當
    及格線」，比起完全沒有基準，這樣至少還能篩出「比整個產業裡最低
    的都還貴1.5倍」這種真正誇張的個股。

    沒有在 INDUSTRY_PE_CUTOFF 裡指定門檻的產業（例如ETF，本身沒有EPS/本益比
    這種傳統股票估值概念），只排除虧損股，不做上限排除，取剩餘股票的平均。

    回傳 {產業: 平均本益比}。
    """
    result = {}
    for industry, group in pe_by_industry.groupby("industry"):
        valid = group[group["pe"] > 0]["pe"]
        if valid.empty:
            continue

        cutoff = INDUSTRY_PE_CUTOFF.get(industry)
        if cutoff is None:
            result[industry] = float(valid.mean())
            continue

        cleaned = valid[valid <= cutoff]
        if cleaned.empty:
            # 全部股票都超過門檻，直接用門檻值本身當基準，不留空
            result[industry] = float(cutoff)
        else:
            result[industry] = float(cleaned.mean())
    return result


def get_gross_margin_at(stock_financials_df: pd.DataFrame, year: int, quarter: int) -> float | None:
    """取得指定年度、季度（累計）的毛利率。營收用 毛利+營業成本 反推，避免猜測FinMind營收欄位名稱。"""
    if stock_financials_df.empty:
        return None
    df = stock_financials_df.copy()
    df["fyear"] = df["date"].dt.year
    df["fquarter"] = df["date"].apply(_quarter_of)
    row = df[(df["fyear"] == year) & (df["fquarter"] == quarter)]
    if row.empty:
        return None

    gp = row.iloc[0].get("gross_profit")
    cogs = row.iloc[0].get("cost_of_goods_sold")
    if gp is None or cogs is None or pd.isna(gp) or pd.isna(cogs):
        return None

    revenue = gp + cogs  # 會計恆等式：營收 = 毛利 + 營業成本
    if revenue == 0:
        return None
    return gp / revenue


def _find_latest_available_quarter(stock_financials_df: pd.DataFrame, today: pd.Timestamp) -> tuple[int, int] | None:
    """找出「今年」最新一筆有毛利/成本資料的季度，回傳 (year, quarter)，找不到回傳 None。"""
    if stock_financials_df.empty:
        return None
    df = stock_financials_df.dropna(subset=["gross_profit", "cost_of_goods_sold"])
    if df.empty:
        return None
    df = df.copy()
    df["fyear"] = df["date"].dt.year
    df["fquarter"] = df["date"].apply(_quarter_of)
    this_year_rows = df[df["fyear"] == today.year]
    if this_year_rows.empty:
        return None
    latest = this_year_rows.sort_values("date").iloc[-1]
    return int(latest["fyear"]), int(latest["fquarter"])


MARGIN_SENSITIVITY = 2.0  # ⚠️ 初始假設，之後用回測校準（供 margin_trend_score 這個連續分數使用，
                          # 該函式目前不在決策流程裡被呼叫，保留供未來參考或其他用途）


def margin_trend_score(stock_financials_df: pd.DataFrame, today: pd.Timestamp | None = None) -> float:
    """
    毛利率趨勢分數（漸進式 0-100，不是硬指標）：
    拿「今年最新一期累計毛利率」跟「去年同一期累計毛利率」比較（同期比同期），
    用 S 型函數把變化幅度（百分點）映射成分數：
    - 毛利率持平（變化=0） → 50分（中性）
    - 毛利率上升越多 → 分數越接近100
    - 毛利率下降越多 → 分數越接近0
    缺資料時回傳中性 50 分。
    """
    if today is None:
        today = pd.Timestamp.today()

    latest_period = _find_latest_available_quarter(stock_financials_df, today)
    if latest_period is None:
        return 50.0
    year, quarter = latest_period

    margin_this = get_gross_margin_at(stock_financials_df, year, quarter)
    margin_last = get_gross_margin_at(stock_financials_df, year - 1, quarter)
    if margin_this is None or margin_last is None:
        return 50.0

    change_pct_points = (margin_this - margin_last) * 100
    score = 100 / (1 + np.exp(-change_pct_points / MARGIN_SENSITIVITY))
    return float(np.clip(score, 0, 100))


# ---------------------------------------------------------------------------
# ⚠️ 2026/08 修正記錄：原本用「絕對百分點」判斷毛利率是否嚴重惡化（分數<30，
# 對應約-1.69個百分點），實測發現對高毛利率產業（例如半導體－IC設計，平均
# 毛利率39.21%）天生不公平——同樣絕對降3個百分點，對50%毛利率的公司只是
# 相對衰退6%，對10%毛利率的公司卻是相對衰退30%，嚴重程度天差地遠。
# 已改用「相對衰退幅度」，套用在全部非景氣循環股，不用再逐一產業手動設定
# 絕對門檻。
# ---------------------------------------------------------------------------
MARGIN_RELATIVE_DECLINE_THRESHOLD = 0.08  # 相對衰退超過8%視為「毛利率明顯衰退」，使用者確認定案


def is_margin_severely_declining(
    stock_financials_df: pd.DataFrame, today: pd.Timestamp | None = None,
    threshold: float = MARGIN_RELATIVE_DECLINE_THRESHOLD,
) -> bool:
    """
    判斷毛利率是否「明顯衰退」，用相對衰退幅度而非絕對百分點：

        相對衰退幅度 = (去年同期毛利率 − 今年毛利率) ÷ 去年同期毛利率

    相對衰退幅度 > 門檻（預設8%）→ 判定明顯衰退。

    特殊情況：
    - 今年毛利率轉為負值（去年是正值）→ 直接判定明顯衰退，不需要算相對
      幅度才能判斷（由盈轉虧比任何相對比例都嚴重）。
    - 去年同期毛利率 <=0（基期本身就是虧損）→ 相對衰退計算沒有意義（分母
      不是正常基準），回傳 False（不判定為明顯衰退，中性處理，不因基期
      異常而誤傷）。
    - 缺資料 → 回傳 False（中性，不因缺資料而受罰，跟系統其他地方一致）。
    """
    if today is None:
        today = pd.Timestamp.today()

    latest_period = _find_latest_available_quarter(stock_financials_df, today)
    if latest_period is None:
        return False
    year, quarter = latest_period

    margin_this = get_gross_margin_at(stock_financials_df, year, quarter)
    margin_last = get_gross_margin_at(stock_financials_df, year - 1, quarter)
    if margin_this is None or margin_last is None or margin_last <= 0:
        return False

    if margin_this <= 0:
        return True

    relative_decline = (margin_last - margin_this) / margin_last
    return relative_decline > threshold


OVERVALUATION_THRESHOLD = 1.5  # ⚠️ 初始假設，之後用回測校準，供 is_overvalued() 的硬門檻使用
VALUATION_SENSITIVITY = 0.25   # ⚠️ 初始假設，之後用回測校準，供 valuation_score() 的S型函數使用


def valuation_score(pe: float | None, industry_avg_pe: float | None, is_loss: bool = False) -> float:
    """
    估值子分數（漸進式 0-100，不是硬門檻——已依使用者要求從門檻式改成漸進式，
    呼應 relative_strength_score / margin_trend_score 同樣的S型函數設計手法）：
    - is_loss=True（EPS估算方法明確判定近期虧損，見 estimate_annual_eps 的
      method 回傳值）→ 0分（最低分）。這是「已經知道財務狀況不好」，
      不該跟真正缺資料一樣給中性分數。
    - 真正缺資料（PE或同業基準算不出來，且不是因為虧損）→ 50分（真正中性：
      代表「不知道」，數值上剛好等於「本益比=同業基準」的分數，但語意不同，
      這是刻意設計成一致，避免缺資料時給出比多數正常股票更高的不合理分數）。
    - 本益比 vs 同業基準的相對差距（(pe/industry_avg_pe)-1），用S型函數映射：
      本益比=同業基準 → 50分；越便宜分數越接近100；越貴分數越接近0。
      本益比為同業基準1.5倍（原本的「過虛」硬門檻）時，分數約12分（明顯偏低但非死板歸零）。
    """
    if is_loss:
        return 0.0
    if pe is None or industry_avg_pe is None or industry_avg_pe <= 0:
        return 50.0
    excess_ratio = (pe / industry_avg_pe) - 1.0
    score = 100 / (1 + np.exp(excess_ratio / VALUATION_SENSITIVITY))
    return float(np.clip(score, 0, 100))


def is_overvalued(
    pe: float | None, industry_avg_pe: float | None,
    is_loss: bool = False, threshold_multiple: float = OVERVALUATION_THRESHOLD,
) -> bool:
    """
    布林版本，供 Decision Engine 判斷是否要強制排除 BUY_NOW（即使其他各項條件都達標）。
    ⚠️ 這裡維持原本的門檻式硬判斷，跟上面改成漸進式的 valuation_score() 是
    兩個不同用途的機制：這裡是「能不能買」的強制關卡，valuation_score() 是
    「股票分數」裡的其中一項評分，兩者刻意保持獨立，不互相影響。
    虧損公司（is_loss=True）視同「過虛」，一併強制排除，跟 valuation_score
    給最低分的原則一致——「公司在虧錢」本身就是不該追高進場的理由。
    真正缺資料時，維持「不算過虛」（不強制排除）。
    """
    if is_loss:
        return True
    if pe is None or industry_avg_pe is None or industry_avg_pe <= 0:
        return False
    return pe > industry_avg_pe * threshold_multiple


# ---------------------------------------------------------------------------
# 景氣循環股：valuation 改用 P/B（股價淨值比），margin_trend 權重歸零
# 已與使用者確認：用真實EPS波動度（變異係數）診斷驗證過，塑膠/石化及煉油、
# 半導體－記憶體 是最強力驗證的兩個，鋼鐵中等驗證，航運則因資料歷史深度
# 不足未能驗證（但有真實歷史數據佐證其循環特性），四類統一歸類為景氣循環股。
# ---------------------------------------------------------------------------
CYCLICAL_INDUSTRIES = {
    "半導體－記憶體",
    "塑膠/石化及煉油",
    "鋼鐵",
    "航運",
}

# ---------------------------------------------------------------------------
# 龍頭股：本益比容忍倍數比一般股票更寬鬆，反映產業龍頭享有的規模／流動性／
# 市場地位溢價——用一般同業的本益比水準去衡量龍頭股「貴不貴」並不公平
# （例如聯發科實測本益比70倍，但拿來比較的「半導體－IC設計」同業基準
# 只有23.83倍，主要是中小型IC設計公司，跟聯發科規模、市場地位差距懸殊，
# 混在一起比較容易誤判成「明顯過虛」）。
# ⚠️ 使用者確認定案：台積電/聯發科/大立光，統一給2.0倍門檻，之後可再增列
# 其他產業龍頭股，或依個別公司給不同倍數。
# ---------------------------------------------------------------------------
LEADING_STOCKS_THRESHOLD = {
    "2330": 3.0,  # 台灣積體電路製造
    "2454": 3.0,  # 聯發科技
    "3008": 3.0,  # 大立光電
}

MIN_EPS_FOR_SHARE_ESTIMATE = 0.05  # ⚠️ 初始假設，EPS絕對值小於這個門檻時，反推股數不可靠，不採用


def estimate_shares_outstanding(net_income: float | None, eps_standalone: float | None) -> float | None:
    """
    股數估算 = 淨利 ÷ 單季EPS。
    ⚠️ 這是估算值，不是真實股數（FinMind沒有直接提供股數欄位）。
    比照EPS估算的防呆原則：EPS絕對值太小時，除法會爆炸或失真，這種情況
    直接回傳 None，不採用這筆估算，避免用不可靠的股數污染後續的P/B計算。
    """
    if net_income is None or eps_standalone is None:
        return None
    if pd.isna(net_income) or pd.isna(eps_standalone):
        return None
    if abs(eps_standalone) < MIN_EPS_FOR_SHARE_ESTIMATE:
        return None
    shares = net_income / eps_standalone
    if shares <= 0:
        return None
    return shares


def compute_book_value_per_share(stock_financials_df: pd.DataFrame) -> tuple[float | None, str]:
    """
    每股淨值 = 最新一筆股東權益 ÷ 用同一季反推出來的股數。
    equity（股東權益）是資產負債表科目，屬於「某個時間點的餘額」，
    不是像EPS/毛利那樣的「累計流量」，所以直接取最新一筆可用的資料即可，
    不需要像EPS那樣做單季/累計的轉換。

    回傳 (每股淨值, 狀態)，狀態為 "ok" / "negative_equity" / "insufficient_data"，
    比照 estimate_annual_eps 的 (值, method) 設計——"股東權益為負"（淨值已經
    虧光，比一般EPS虧損更嚴重的警訊）跟"真的缺資料"是完全不同的意義，
    不該混在一起都回傳 None，要讓呼叫端能分別處理（比照 valuation_score
    對 is_loss 跟缺資料給不同分數的原則）。
    """
    if stock_financials_df.empty:
        return None, "insufficient_data"

    df = stock_financials_df.dropna(subset=["equity", "net_income", "eps_cumulative"]).sort_values("date")
    if df.empty:
        return None, "insufficient_data"

    latest = df.iloc[-1]
    shares = estimate_shares_outstanding(latest["net_income"], latest["eps_cumulative"])
    if shares is None:
        return None, "insufficient_data"

    equity = latest["equity"]
    if pd.isna(equity):
        return None, "insufficient_data"
    if equity <= 0:
        return None, "negative_equity"

    return float(equity / shares), "ok"


def compute_pb(current_price: float, book_value_per_share: float | None) -> float | None:
    """P/B = 現價 ÷ 每股淨值。淨值算不出來或非正值，回傳 None。"""
    if book_value_per_share is None or book_value_per_share <= 0:
        return None
    return current_price / book_value_per_share


def compute_industry_pb_benchmark(pb_by_industry: pd.DataFrame) -> dict[str, float]:
    """
    pb_by_industry 需含 industry、pb 兩欄。
    ⚠️ 這裡用 IQR（四分位距）統計方法排除異常值，跟本益比那邊改成使用者
    手動指定門檻不同——P/B 目前沒有另外請使用者逐一產業訂定門檻，
    考量開發時間，先用回歸統計方法頂著，之後有需要可以再比照P/E的做法
    改成手動門檻。
    """
    result = {}
    for industry, group in pb_by_industry.groupby("industry"):
        valid = group[group["pb"] > 0]["pb"]
        if len(valid) < 2:
            if len(valid) == 1:
                result[industry] = float(valid.iloc[0])
            continue
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        cleaned = valid[valid <= upper_bound]
        if cleaned.empty:
            continue
        result[industry] = float(cleaned.mean())
    return result


PB_SENSITIVITY = 0.25  # ⚠️ 初始假設，之後用回測校準，跟 valuation_score 的 VALUATION_SENSITIVITY 同樣手法


def pb_score(pb: float | None, industry_avg_pb: float | None, has_negative_equity: bool = False) -> float:
    """
    景氣循環股專用的估值子分數（用P/B取代P/E，S型函數，漸進式）：
    - has_negative_equity=True（股東權益為負，淨值已經虧光）→ 0分，比EPS虧損更嚴重的警訊
    - 真正缺資料 → 50分中性
    - P/B vs 同業基準的相對差距，用S型函數映射：P/B=同業基準 → 50分
    """
    if has_negative_equity:
        return 0.0
    if pb is None or industry_avg_pb is None or industry_avg_pb <= 0:
        return 50.0
    excess_ratio = (pb / industry_avg_pb) - 1.0
    score = 100 / (1 + np.exp(excess_ratio / PB_SENSITIVITY))
    return float(np.clip(score, 0, 100))


EPS_GROWTH_SENSITIVITY = 0.15  # ⚠️ 初始假設，之後用回測校準


def eps_growing_score(estimated_annual_eps: float | None, last_year_full_year_eps: float | None) -> float:
    """
    連續型 EPS 成長分數（取代原本只有「有沒有成長」的布林值，改成S型函數映射
    成長幅度）：成長率=0（打平）→ 50分；成長越多分數越接近100；
    衰退越多分數越接近0。用 |去年EPS| 當分母，即使去年是虧損（負值），
    只要今年由虧轉盈，成長率算出來仍然會是正值，正確反映「轉機」。
    缺資料時回傳中性50分。
    """
    if estimated_annual_eps is None or last_year_full_year_eps is None:
        return 50.0
    if last_year_full_year_eps == 0:
        return 50.0
    growth_rate = (estimated_annual_eps - last_year_full_year_eps) / abs(last_year_full_year_eps)
    score = 100 / (1 + np.exp(-growth_rate / EPS_GROWTH_SENSITIVITY))
    return float(np.clip(score, 0, 100))
