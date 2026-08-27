"""
台股產業板塊熱力圖
放進 pages/ 資料夾，會自動出現在 Streamlit 側邊欄選單
執行方式（從主程式所在資料夾）：streamlit run 波段投資儀表板.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

DB_CONFIG = {
    "user": st.secrets.get("db_user", "postgres"),
    "password": st.secrets["db_password"],
    "host": st.secrets["db_host"],
    "port": st.secrets.get("db_port", "5432"),
    "database": st.secrets.get("db_name", "postgres"),
}

TEXT = {
    "title": {"zh": "台股產業板塊熱力圖", "en": "Taiwan Sector Heatmap"},
    "data_updated": {"zh": "資料更新至 {date}", "en": "Data as of {date}"},
    "rotation_sentence": {"zh": "資金近3日主要朝「<span style='color:#EF5350'>{sector}</span>」類股輪動",
                          "en": "Money has mainly been rotating toward the <span style='color:#EF5350'>{sector}</span> sector over the past 3 days"},
    "trend_10d_top_in": {"zh": "近10日最高淨流入：「<span style='color:#EF5350'>{sector}</span>」",
                         "en": "Highest 10-day net inflow: <span style='color:#EF5350'>{sector}</span>"},
    "trend_10d_second_in": {"zh": "近10日次高淨流入：「<span style='color:#EF5350'>{sector}</span>」",
                            "en": "2nd highest 10-day net inflow: <span style='color:#EF5350'>{sector}</span>"},
    "trend_10d_top_out": {"zh": "近10日最高淨流出：「<span style='color:#42A5F5'>{sector}</span>」",
                          "en": "Highest 10-day net outflow: <span style='color:#42A5F5'>{sector}</span>"},
    "trend_10d_second_out": {"zh": "近10日次高淨流出：「<span style='color:#42A5F5'>{sector}</span>」",
                             "en": "2nd highest 10-day net outflow: <span style='color:#42A5F5'>{sector}</span>"},
    "trend_10d_insufficient": {"zh": "產業數量不足，無法計算近10日淨流入/流出排名（需要至少4個產業）。",
                               "en": "Not enough sectors to rank 10-day net inflow/outflow (needs at least 4)."},
    "no_data": {"zh": "目前資料不足以計算資金輪動分數（可能是資料庫還沒有足夠天數的資料）。",
               "en": "Not enough data yet to compute the rotation score."},
    "formula_title": {"zh": "這個指標怎麼算的？", "en": "How is this score calculated?"},
    "formula_body": {
        "zh": """
**資金輪動分數 = 法人資金強度 × 80% + 價格動能 × 20%**

- **法人資金強度**：該產業所有股票「近N日三大法人合計買賣超股數 × 收盤價」加總，代表法人實際投入的資金規模
- **價格動能**：該產業所有股票「近N日報酬率」的平均值
- 因為兩者單位不同（金額 vs 百分比），會先各自做 0~100 的正規化（在所有產業中排名相對高低），再依權重加總
- 分數最高的產業，就是資金相對集中流入的族群
- 頁面上同時顯示「近3日」與「近10日」兩個版本，算法完全相同、只差時間窗口：3日反映短期資金去向，10日用來確認這股趨勢是否有延續性——若兩者一致，代表輪動較為穩定，不只是單日雜訊
- 「近10日最高/次高淨流入」= 10日分數排名最高、次高的產業；「近10日最高/次高淨流出」= 10日分數排名最低、次低的產業。這四項各自獨立排名，並非追蹤同一筆資金從某產業流向另一產業

⚠️ 這是機械化計算出的參考指標，不代表個股基本面，仍需搭配技術面與籌碼面確認。
""",
        "en": """
**Rotation score = Institutional strength × 80% + Price momentum × 20%**

- **Institutional strength**: sum of (N-day institutional net shares × close price) across all stocks in the sector — reflects the actual capital institutions have deployed
- **Price momentum**: average N-day return across all stocks in the sector
- Since the two are on different scales (currency vs percentage), each is normalized to a 0-100 relative rank across sectors before applying the weights
- The sector with the highest score is where capital has been relatively concentrating
- The page shows both a 3-day and a 10-day version using the identical formula, differing only in the lookback window: the 3-day view shows short-term capital flow, while the 10-day view checks whether that trend has persistence — agreement between the two suggests a more stable rotation rather than single-day noise
- "Highest/2nd highest 10-day net inflow" = the sectors ranked highest and 2nd highest by 10-day score; "highest/2nd highest 10-day net outflow" = ranked lowest and 2nd lowest. These four rankings are independent — they do not track the same capital moving from one sector to another

⚠️ This is a mechanically computed reference indicator, not a fundamental assessment — confirm with technical and institutional-flow signals before acting.
""",
    },
    "treemap_title": {"zh": "各股報酬率（顏色）／成交值（大小）", "en": "Stock return (color) / trading value (size)"},
    "treemap_period_label": {"zh": "顏色顯示區間", "en": "Color period"},
    "period_3d": {"zh": "3日", "en": "3d"},
    "period_10d": {"zh": "10日", "en": "10d"},
    "period_1m": {"zh": "1個月", "en": "1 month"},
    "period_1q": {"zh": "1季", "en": "1 quarter"},
    "treemap_caption": {"zh": "顏色代表所選區間的個股報酬率，方塊大小代表最新成交值", "en": "Color reflects the stock's return over the selected period; box size reflects latest trading value"},
    "sector_score_title": {"zh": "各產業資金輪動分數排行", "en": "Sector rotation score ranking"},
    "col_sector": {"zh": "產業別", "en": "Sector"},
    "industry_mapping_title": {"zh": "產業分類對照表", "en": "Industry Classification Reference"},
    "sort_by_label": {"zh": "排序依據", "en": "Sort by"},
    "sort_order_label": {"zh": "排序方向", "en": "Sort order"},
    "sort_desc": {"zh": "由高到低", "en": "High to low"},
    "sort_asc": {"zh": "由低到高", "en": "Low to high"},
    "industry_filter_label": {"zh": "依產業篩選", "en": "Filter by sector"},
    "industry_filter_all": {"zh": "全部產業", "en": "All sectors"},
    "col_mapping_stock_id": {"zh": "股票代碼", "en": "Stock code"},
    "col_mapping_name": {"zh": "公司名稱", "en": "Company name"},
    "col_score_3d": {"zh": "3日分數", "en": "3d score"},
    "col_score_10d": {"zh": "10日分數", "en": "10d score"},
    "col_inst_strength_3d": {"zh": "3日法人資金強度(新台幣元,近似值)", "en": "3d institutional strength (TWD, approx.)"},
    "col_inst_strength_10d": {"zh": "10日法人資金強度(新台幣元,近似值)", "en": "10d institutional strength (TWD, approx.)"},
    "col_momentum_3d": {"zh": "3日動能%", "en": "3d momentum %"},
    "col_momentum_10d": {"zh": "10日動能%", "en": "10d momentum %"},
}


def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("lang", "zh")
    template = TEXT[key][lang]
    return template.format(**kwargs) if kwargs else template


@st.cache_resource
def get_engine():
    safe_password = quote_plus(DB_CONFIG["password"])
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{safe_password}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


@st.cache_data(ttl=3600)
def load_industry_mapping() -> pd.DataFrame:
    """完整的股票-產業對照表，直接查 raw.stock_info，不受股價資料是否齊全影響"""
    engine = get_engine()
    df = pd.read_sql(
        text("""
            SELECT stock_id, name_zh, name_en, industry AS industry_zh, industry_en
            FROM raw.stock_info
            WHERE industry IS NOT NULL AND industry != ''
            ORDER BY industry, stock_id
        """),
        engine,
    )
    return df


@st.cache_data(ttl=3600)
def load_sector_data() -> pd.DataFrame:
    """
    抓每檔股票：最新收盤價、最新成交值、近3日與近10日的三大法人合計買賣超股數之和、
    近3日與近10日的價格報酬率，並關聯產業分類
    """
    engine = get_engine()
    query = text("""
        WITH ranked AS (
            SELECT
                m.date, m.stock_id, m.close, m.trading_value,
                m.institutional_total_net,
                LAG(m.close, 3) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_3d_ago,
                LAG(m.close, 10) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_10d_ago,
                LAG(m.close, 20) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_20d_ago,
                LAG(m.close, 60) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_60d_ago,
                SUM(m.institutional_total_net) OVER (
                    PARTITION BY m.stock_id ORDER BY m.date
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS inst_net_3d_shares,
                SUM(m.institutional_total_net) OVER (
                    PARTITION BY m.stock_id ORDER BY m.date
                    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
                ) AS inst_net_10d_shares,
                ROW_NUMBER() OVER (PARTITION BY m.stock_id ORDER BY m.date DESC) AS rn
            FROM staging.daily_master m
        )
        SELECT r.date, r.stock_id, r.close, r.trading_value,
               r.close_3d_ago, r.close_10d_ago, r.close_20d_ago, r.close_60d_ago,
               r.inst_net_3d_shares, r.inst_net_10d_shares,
               s.industry AS industry_zh, s.industry_en, s.name_zh, s.name_en
        FROM ranked r
        LEFT JOIN raw.stock_info s ON s.stock_id = r.stock_id
        WHERE r.rn = 1
    """)
    df = pd.read_sql(query, engine)

    df["momentum_3d"] = (df["close"] - df["close_3d_ago"]) / df["close_3d_ago"] * 100
    df["inst_strength_3d"] = df["inst_net_3d_shares"] * df["close"]
    df["momentum_10d"] = (df["close"] - df["close_10d_ago"]) / df["close_10d_ago"] * 100
    df["inst_strength_10d"] = df["inst_net_10d_shares"] * df["close"]
    df["momentum_20d"] = (df["close"] - df["close_20d_ago"]) / df["close_20d_ago"] * 100
    df["momentum_60d"] = (df["close"] - df["close_60d_ago"]) / df["close_60d_ago"] * 100

    df = df.dropna(subset=["industry_zh"])
    df = df[df["industry_zh"].str.len() > 0]
    return df


def minmax_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo) * 100


def compute_sector_scores(df: pd.DataFrame, inst_col: str, momentum_col: str) -> pd.DataFrame:
    valid = df.dropna(subset=[inst_col, momentum_col])
    sector_df = valid.groupby("industry").agg(
        inst_strength=(inst_col, "sum"),
        momentum=(momentum_col, "mean"),
        stock_count=("stock_id", "count"),
    ).reset_index()

    sector_df["norm_inst"] = minmax_normalize(sector_df["inst_strength"])
    sector_df["norm_momentum"] = minmax_normalize(sector_df["momentum"])
    sector_df["rotation_score"] = sector_df["norm_inst"] * 0.8 + sector_df["norm_momentum"] * 0.2
    sector_df = sector_df.sort_values("rotation_score", ascending=False).reset_index(drop=True)
    return sector_df


st.set_page_config(page_title=t("title"), layout="wide")

st.markdown("""
<style>
[data-testid="stTable"] table th, [data-testid="stTable"] table td {
    text-align: center !important;
}
[data-testid="stTable"] {
    max-height: 480px;
    overflow-y: auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state["lang"] = "zh"

st.title(t("title"))

st.sidebar.markdown("---")
st.sidebar.caption("A project by I.H. Wang")

df = load_sector_data()

if not df.empty:
    latest_date = pd.to_datetime(df["date"]).max().strftime("%Y/%m/%d")
    st.caption(t("data_updated", date=latest_date))

lang = st.session_state.get("lang", "zh")
df["industry"] = df["industry_en"] if lang == "en" else df["industry_zh"]

if df.empty or len(df["industry"].unique()) < 2:
    st.warning(t("no_data"))
    st.stop()

sector_scores_3d = compute_sector_scores(df, "inst_strength_3d", "momentum_3d")
sector_scores_10d = compute_sector_scores(df, "inst_strength_10d", "momentum_10d")

if sector_scores_3d.empty:
    st.warning(t("no_data"))
    st.stop()

top_sector_3d = sector_scores_3d.iloc[0]["industry"]

st.markdown(f"### {t('rotation_sentence', sector=top_sector_3d)}", unsafe_allow_html=True)

if len(sector_scores_10d) >= 4:
    ranked_10d = sector_scores_10d.sort_values("rotation_score", ascending=False).reset_index(drop=True)
    top_in = ranked_10d.iloc[0]["industry"]
    second_in = ranked_10d.iloc[1]["industry"]
    top_out = ranked_10d.iloc[-1]["industry"]
    second_out = ranked_10d.iloc[-2]["industry"]

    st.markdown(f"##### {t('trend_10d_top_in', sector=top_in)}", unsafe_allow_html=True)
    st.markdown(f"##### {t('trend_10d_second_in', sector=second_in)}", unsafe_allow_html=True)
    st.markdown(f"##### {t('trend_10d_top_out', sector=top_out)}", unsafe_allow_html=True)
    st.markdown(f"##### {t('trend_10d_second_out', sector=second_out)}", unsafe_allow_html=True)
elif not sector_scores_10d.empty:
    st.caption(t("trend_10d_insufficient"))

with st.expander(t("formula_title")):
    st.markdown(t("formula_body"))

# ---- Treemap：股票依產業分組，方塊大小=成交值，顏色=所選區間報酬率 ----
st.subheader(t("treemap_title"))

period_options = {
    "3d": t("period_3d"),
    "10d": t("period_10d"),
    "1m": t("period_1m"),
    "1q": t("period_1q"),
}
selected_period = st.radio(
    t("treemap_period_label"), options=list(period_options.keys()),
    format_func=lambda k: period_options[k], horizontal=True,
)
color_col = {"3d": "momentum_3d", "10d": "momentum_10d", "1m": "momentum_20d", "1q": "momentum_60d"}[selected_period]
st.caption(t("treemap_caption"))

lang = st.session_state.get("lang", "zh")
name_col = "name_zh" if lang == "zh" else "name_en"
df["label"] = df["stock_id"] + "  " + df[name_col].fillna("")

fig = px.treemap(
    df,
    path=["industry", "label"],
    values="trading_value",
    color=color_col,
    color_continuous_scale=["#1D9E75", "#F1EFE8", "#D85A30"],  # 綠跌-灰平-紅漲，符合台股慣例
    color_continuous_midpoint=0,
)
fig.update_layout(height=550, margin=dict(t=10, l=10, r=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---- 產業分數排行表：3日與10日並列 ----
st.subheader(t("sector_score_title"))

merged_scores = sector_scores_3d[["industry", "rotation_score", "inst_strength", "momentum"]].rename(
    columns={"rotation_score": "score_3d", "inst_strength": "inst_3d", "momentum": "momentum_3d"}
).merge(
    sector_scores_10d[["industry", "rotation_score", "inst_strength", "momentum"]].rename(
        columns={"rotation_score": "score_10d", "inst_strength": "inst_10d", "momentum": "momentum_10d"}
    ),
    on="industry", how="outer",
)

score_sort_options = {
    t("col_score_3d"): "score_3d",
    t("col_inst_strength_3d"): "inst_3d",
    t("col_momentum_3d"): "momentum_3d",
    t("col_score_10d"): "score_10d",
    t("col_inst_strength_10d"): "inst_10d",
    t("col_momentum_10d"): "momentum_10d",
}
score_scol1, score_scol2 = st.columns(2)
score_sort_label = score_scol1.selectbox(t("sort_by_label"), options=list(score_sort_options.keys()), index=0)
score_sort_order = score_scol2.radio(
    t("sort_order_label"), options=[t("sort_desc"), t("sort_asc")], horizontal=True, key="score_sort_order",
)
score_ascending = score_sort_order == t("sort_asc")
merged_scores = merged_scores.sort_values(
    score_sort_options[score_sort_label], ascending=score_ascending, na_position="last"
)

display_df = merged_scores.rename(columns={
    "industry": t("col_sector"),
    "score_3d": t("col_score_3d"),
    "score_10d": t("col_score_10d"),
    "inst_3d": t("col_inst_strength_3d"),
    "inst_10d": t("col_inst_strength_10d"),
    "momentum_3d": t("col_momentum_3d"),
    "momentum_10d": t("col_momentum_10d"),
})
st.table(
    display_df.style.format({
        t("col_score_3d"): "{:.1f}", t("col_score_10d"): "{:.1f}",
        t("col_inst_strength_3d"): "{:,.0f}", t("col_inst_strength_10d"): "{:,.0f}",
        t("col_momentum_3d"): "{:.2f}", t("col_momentum_10d"): "{:.2f}",
    }, na_rep="-").hide(axis="index")
)

# ---- 產業分類對照表 ----
st.subheader(t("industry_mapping_title"))

mapping_df = load_industry_mapping()
mapping_name_col = "name_zh" if lang == "zh" else "name_en"
mapping_industry_col = "industry_zh" if lang == "zh" else "industry_en"

industry_options = [t("industry_filter_all")] + sorted(mapping_df[mapping_industry_col].dropna().unique().tolist())
selected_industry = st.selectbox(t("industry_filter_label"), options=industry_options)

if selected_industry != t("industry_filter_all"):
    mapping_df = mapping_df[mapping_df[mapping_industry_col] == selected_industry]

mapping_display = mapping_df[["stock_id", mapping_name_col, mapping_industry_col]].rename(columns={
    "stock_id": t("col_mapping_stock_id"),
    mapping_name_col: t("col_mapping_name"),
    mapping_industry_col: t("col_sector"),
})
st.table(mapping_display.style.hide(axis="index"))
