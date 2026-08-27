"""
台股波段投資儀表板 - 第一版（支援繁體中文 / 英文切換）
讀取 staging.daily_master（價格 + 籌碼）與 mart.technical_indicators（技術指標）
執行方式：streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

from watchlist import WATCHLIST

# ---- 資料庫連線設定，請依你的實際狀況修改 ----
DB_CONFIG = {
    "user": st.secrets.get("db_user", "postgres"),
    "password": st.secrets["db_password"],
    "host": st.secrets["db_host"],
    "port": st.secrets.get("db_port", "5432"),
    "database": st.secrets.get("db_name", "postgres"),
}

# ---- 語言字典 ----
TEXT = {
    "page_title": {"zh": "台股波段儀表板", "en": "TW Swing Trading Dashboard"},
    "title": {"zh": "台股波段投資儀表板", "en": "Taiwan Swing Trading Dashboard"},
    "data_updated": {"zh": "資料更新至 {date}", "en": "Data as of {date}"},
    "language": {"zh": "語言", "en": "Language"},
    "stock_code": {"zh": "股票代碼", "en": "Stock code"},
    "lookback_days": {"zh": "顯示天數", "en": "Days to show"},
    "watchlist_count": {"zh": "觀察池目前共有 {n} 檔股票", "en": "Watchlist currently has {n} stocks"},
    "no_data_db": {"zh": "資料庫裡目前沒有任何股票資料，請先執行資料抓取與轉換腳本。",
                   "en": "No stock data found in the database. Please run the fetch and transform scripts first."},
    "no_data_stock": {"zh": "找不到股票 {sid} 的資料，請確認資料庫裡已經有這檔股票。",
                      "en": "No data found for stock {sid}. Please confirm it exists in the database."},
    "tab_radar": {"zh": "選股雷達", "en": "Screening radar"},
    "tab_overview": {"zh": "總覽", "en": "Overview"},
    "tab_technical": {"zh": "技術訊號", "en": "Technical signals"},
    "tab_chips": {"zh": "籌碼面", "en": "Institutional flows"},
    "tab_watchlist": {"zh": "觀察池", "en": "Watchlist"},
    "refresh_button": {"zh": "🔄 重新整理資料", "en": "🔄 Refresh data"},
    "refresh_help": {"zh": "跑完更新腳本後點這裡，立即反映最新的觀察池與資料",
                     "en": "Click after running the update scripts to reflect the latest watchlist and data immediately"},
    "watchlist_subheader": {"zh": "觀察池：資料庫目前涵蓋的股票", "en": "Watchlist: stocks currently covered in the database"},
    "watchlist_defined_count": {"zh": "清單定義檔（watchlist.py）共 {n} 檔", "en": "Defined in watchlist.py: {n} stocks"},
    "watchlist_actual_count": {"zh": "資料庫實際涵蓋 {n} 檔", "en": "Actually covered in database: {n} stocks"},
    "watchlist_missing": {"zh": "尚未成功寫入資料庫的股票：{codes}",
                          "en": "Stocks not yet in the database: {codes}"},
    "watchlist_missing_none": {"zh": "清單裡所有股票都已成功寫入資料庫，兩邊數量一致。",
                               "en": "All stocks in the list are present in the database — counts match."},
    "col_latest_date": {"zh": "最新資料日期", "en": "Latest data date"},
    "radar_subheader": {"zh": "選股雷達：均線多頭排列 + 相對強度 + 量能突破",
                        "en": "Screening radar: bullish MA alignment + relative strength + volume breakout"},
    "bullish_ma_count": {"zh": "均線多頭排列", "en": "Bullish MA alignment"},
    "unit_stocks": {"zh": "{n} 檔", "en": "{n} stocks"},
    "volume_breakout_count": {"zh": "其中同時量能突破", "en": "Also with volume breakout"},
    "rs_caption": {"zh": "相對強度 = 個股近20日報酬 − 0050近20日報酬，正值代表跑贏大盤",
                  "en": "Relative strength = stock's 20-day return − 0050's 20-day return; positive means outperforming the market"},
    "indicator_basis": {
        "zh": "以上指標皆以「日線」資料計算：均線多頭排列 = 日 MA5 > 日 MA20 > 日 MA60（5/20/60個交易日收盤價均線）；相對強度 = 近20個交易日（約1個月）日收盤價報酬率相減；量能突破 = 當日成交量 ÷ 近20日日均量 > 1.5倍。皆非週線或月線。",
        "en": "All indicators use daily data: bullish MA alignment = daily MA5 > MA20 > MA60 (5/20/60-day closing price averages); relative strength = 20-trading-day (~1 month) daily return difference; volume breakout = today's volume ÷ 20-day average daily volume > 1.5x. None of these use weekly or monthly bars.",
    },
    "radar_table_title": {"zh": "均線多頭排列股票列表", "en": "Bullish MA Alignment Stock List"},
    "filter_title": {"zh": "篩選條件（可複選）", "en": "Filters (multi-select)"},
    "filter_bullish": {"zh": "均線多頭排列", "en": "Bullish MA alignment"},
    "filter_rs_positive": {"zh": "相對強度 > 0（跑贏大盤）", "en": "Relative strength > 0 (outperforming market)"},
    "filter_breakout": {"zh": "量能突破", "en": "Volume breakout"},
    "filter_result_count": {"zh": "篩選結果：{n} 檔", "en": "Filtered results: {n} stocks"},
    "filter_no_result": {"zh": "目前篩選條件下沒有符合的股票，請調整條件。",
                         "en": "No stocks match the current filters. Try adjusting them."},
    "breakout_filter_label": {"zh": "量能突破篩選", "en": "Volume breakout filter"},
    "breakout_all": {"zh": "全部", "en": "All"},
    "breakout_true": {"zh": "只顯示 True", "en": "True only"},
    "breakout_false": {"zh": "只顯示 False", "en": "False only"},
    "sort_by_label": {"zh": "排序依據", "en": "Sort by"},
    "sort_order_label": {"zh": "排序方向", "en": "Sort order"},
    "sort_desc": {"zh": "由高到低", "en": "High to low"},
    "sort_asc": {"zh": "由低到高", "en": "Low to high"},
    "col_stock_id": {"zh": "股票代碼", "en": "Stock code"},
    "col_name": {"zh": "公司名稱", "en": "Company name"},
    "col_close": {"zh": "收盤價(新台幣元)", "en": "Close (TWD)"},
    "col_rs": {"zh": "相對強度(%)", "en": "Rel. strength (%)"},
    "col_vol_ratio": {"zh": "量能倍數", "en": "Volume ratio"},
    "col_rsi": {"zh": "RSI(14)", "en": "RSI(14)"},
    "col_inst_net": {"zh": "三大法人合計買賣超(股)", "en": "Institutional net total (shares)"},
    "col_vol_breakout": {"zh": "量能突破", "en": "Volume breakout"},
    "col_date": {"zh": "資料日期", "en": "Date"},
    "price_trend": {"zh": "{sid} 價格走勢", "en": "{sid} price trend"},
    "kline": {"zh": "K線", "en": "Candles"},
    "latest_close": {"zh": "最新收盤價(新台幣元)", "en": "Latest close (TWD)"},
    "volume": {"zh": "成交量(股)", "en": "Volume (shares)"},
    "inst_net_total": {"zh": "三大法人合計買賣超(股)", "en": "Institutional net total (shares)"},
    "technical_indicators": {"zh": "技術指標", "en": "Technical indicators"},
    "chips": {"zh": "籌碼面", "en": "Institutional flows"},
    "foreign": {"zh": "外資", "en": "Foreign"},
    "trust": {"zh": "投信", "en": "Investment trust"},
    "dealer": {"zh": "自營商", "en": "Dealer"},
    "inst_net_shares": {"zh": "三大法人買賣超（股）", "en": "Institutional net buy/sell (shares)"},
    "margin_balance": {"zh": "融資餘額(張)", "en": "Margin balance (lots)"},
    "short_balance": {"zh": "融券餘額(張)", "en": "Short balance (lots)"},
    "margin_short_title": {"zh": "融資融券餘額(張)", "en": "Margin & short balance (lots)"},
    "price_axis": {"zh": "股價(新台幣元)", "en": "Price (TWD)"},
    "macd_title": {"zh": "MACD(新台幣元)", "en": "MACD (TWD)"},
}


def format_close_price(v):
    """股價 < 100 顯示 2 位小數；100~1000 顯示 1 位小數；>= 1000 顯示整數並加千分位逗號"""
    if pd.isna(v):
        return "-"
    if v < 100:
        return f"{v:.2f}"
    elif v < 1000:
        return f"{v:.1f}"
    else:
        return f"{v:,.0f}"


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
def load_stock_names() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT stock_id, name_zh, name_en FROM raw.stock_info"),
        engine,
    )
    return df


def stock_label(stock_id: str, names_df: pd.DataFrame) -> str:
    """依目前語言組出下拉選單顯示文字，例如「2330 台灣積體電路製造」"""
    row = names_df[names_df["stock_id"] == stock_id]
    if row.empty:
        return stock_id
    lang = st.session_state.get("lang", "zh")
    name = row["name_zh"].iloc[0] if lang == "zh" else row["name_en"].iloc[0]
    if pd.isna(name):
        return stock_id
    return f"{stock_id}  {name}"


@st.cache_data(ttl=3600)
def load_latest_data_date():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(date) FROM staging.daily_master")).scalar()
    return result


@st.cache_data(ttl=3600)
def load_available_stocks() -> list:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT DISTINCT stock_id FROM staging.daily_master ORDER BY stock_id"),
        engine,
    )
    return df["stock_id"].tolist()


@st.cache_data(ttl=3600)
def load_all_latest() -> pd.DataFrame:
    """選股雷達用：抓每檔股票最新一天的價格、技術指標，並算出20日前收盤價與20日均量"""
    engine = get_engine()
    query = text("""
        WITH ranked AS (
            SELECT
                m.date, m.stock_id, m.close, m.volume, m.institutional_total_net,
                t.ma5, t.ma20, t.ma60, t.rsi_14,
                LAG(m.close, 20) OVER (PARTITION BY m.stock_id ORDER BY m.date) AS close_20d_ago,
                AVG(m.volume) OVER (
                    PARTITION BY m.stock_id ORDER BY m.date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS avg_volume_20,
                ROW_NUMBER() OVER (PARTITION BY m.stock_id ORDER BY m.date DESC) AS rn
            FROM staging.daily_master m
            LEFT JOIN mart.technical_indicators t ON t.stock_id = m.stock_id AND t.date = m.date
        )
        SELECT * FROM ranked WHERE rn = 1
    """)
    df = pd.read_sql(query, engine)

    df["return_20d"] = (df["close"] - df["close_20d_ago"]) / df["close_20d_ago"] * 100
    df["volume_ratio"] = df["volume"] / df["avg_volume_20"]

    # 用 0050 當大盤基準，算相對強度（個股20日報酬 - 大盤20日報酬）
    benchmark_row = df[df["stock_id"] == "0050"]
    if not benchmark_row.empty and pd.notna(benchmark_row["return_20d"].iloc[0]):
        benchmark_return = float(benchmark_row["return_20d"].iloc[0])
        df["relative_strength"] = df["return_20d"] - benchmark_return
    else:
        # 0050 當天算不出20日報酬（例如資料筆數不足），相對強度整欄留空但不影響其他欄位
        df["relative_strength"] = pd.Series([pd.NA] * len(df), dtype="Float64")

    return df


@st.cache_data(ttl=3600)
def load_watchlist_detail() -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT stock_id, MAX(date) AS latest_date
        FROM staging.daily_master
        GROUP BY stock_id
        ORDER BY stock_id
    """)
    dates_df = pd.read_sql(query, engine)

    latest_df = load_all_latest()[["stock_id", "close"]]
    return dates_df.merge(latest_df, on="stock_id", how="left")


@st.cache_data(ttl=3600)
def load_master(stock_id: str) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM staging.daily_master WHERE stock_id = :sid ORDER BY date"),
        engine, params={"sid": stock_id},
    )
    return df


@st.cache_data(ttl=3600)
def load_indicators(stock_id: str) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM mart.technical_indicators WHERE stock_id = :sid ORDER BY date"),
        engine, params={"sid": stock_id},
    )
    return df


# ---- 語言初始化 ----
if "lang" not in st.session_state:
    st.session_state["lang"] = "zh"

st.set_page_config(page_title=t("page_title"), layout="wide")

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

lang_choice = st.sidebar.radio(
    t("language"), options=["zh", "en"],
    format_func=lambda x: "繁體中文" if x == "zh" else "English",
    index=0 if st.session_state["lang"] == "zh" else 1,
    horizontal=True,
)
st.session_state["lang"] = lang_choice

st.title(t("title"))

latest_date = load_latest_data_date()
if latest_date is not None:
    st.caption(t("data_updated", date=latest_date.strftime("%Y/%m/%d")))

available_stocks = load_available_stocks()

if not available_stocks:
    st.error(t("no_data_db"))
    st.stop()

names_df = load_stock_names()

stock_id = st.sidebar.selectbox(
    t("stock_code"), options=available_stocks, index=0,
    format_func=lambda sid: stock_label(sid, names_df),
)
lookback_days = st.sidebar.slider(t("lookback_days"), min_value=30, max_value=396, value=180)
st.sidebar.caption(t("watchlist_count", n=len(available_stocks)))

if st.sidebar.button(t("refresh_button"), help=t("refresh_help")):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("A project by I.H. Wang")

master_df = load_master(stock_id)
ind_df = load_indicators(stock_id)

if master_df.empty:
    st.warning(t("no_data_stock", sid=stock_id))
    st.stop()

df = master_df.merge(ind_df, on=["date", "stock_id"], how="left")
df = df.tail(lookback_days)

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    t("tab_radar"), t("tab_overview"), t("tab_technical"), t("tab_chips"), t("tab_watchlist"),
])

with tab0:
    st.subheader(t("radar_subheader"))
    latest_df = load_all_latest()
    latest_df = latest_df.dropna(subset=["ma5", "ma20", "ma60"])

    # 0050、00631L、00632R 一併納入（0050 用自身當大盤基準，相對強度理論上為0；
    # 正2/反1 均線訊號參考意義較低但非計算錯誤，一併顯示由使用者自行判斷）
    screen_df = latest_df.copy()

    screen_df["is_bullish"] = (
        (screen_df["ma5"] > screen_df["ma20"]) & (screen_df["ma20"] > screen_df["ma60"])
    )
    screen_df["is_breakout"] = screen_df["volume_ratio"] > 1.5

    bullish_all = screen_df[screen_df["is_bullish"]]

    col1, col2 = st.columns(2)
    col1.metric(t("bullish_ma_count"), t("unit_stocks", n=len(bullish_all)))
    col2.metric(t("volume_breakout_count"), t("unit_stocks", n=int(bullish_all["is_breakout"].sum())))

    st.caption(t("rs_caption"))
    st.caption(t("indicator_basis"))

    # ---- 篩選勾選框：讓使用者自由組合「均線多頭排列 + 相對強度>0 + 量能突破」 ----
    st.markdown(f"##### {t('filter_title')}")
    fcol1, fcol2, fcol3 = st.columns(3)
    filter_bullish = fcol1.checkbox(t("filter_bullish"), value=True)
    filter_rs = fcol2.checkbox(t("filter_rs_positive"), value=False)
    filter_breakout = fcol3.checkbox(t("filter_breakout"), value=False)

    filtered = screen_df.copy()
    if filter_bullish:
        filtered = filtered[filtered["is_bullish"]]
    if filter_rs:
        filtered = filtered[filtered["relative_strength"] > 0]
    if filter_breakout:
        filtered = filtered[filtered["is_breakout"]]

    # ---- 排序控制項 ----
    sort_options = {
        t("col_rs"): "relative_strength",
        t("col_vol_ratio"): "volume_ratio",
        t("col_rsi"): "rsi_14",
        t("col_inst_net"): "institutional_total_net",
    }
    scol1, scol2 = st.columns(2)
    sort_label = scol1.selectbox(t("sort_by_label"), options=list(sort_options.keys()), index=0)
    sort_order = scol2.radio(
        t("sort_order_label"), options=[t("sort_desc"), t("sort_asc")], horizontal=True,
    )
    ascending = sort_order == t("sort_asc")
    filtered = filtered.sort_values(sort_options[sort_label], ascending=ascending, na_position="last")

    name_col = "name_zh" if st.session_state.get("lang", "zh") == "zh" else "name_en"
    filtered = filtered.merge(names_df[["stock_id", name_col]], on="stock_id", how="left")

    st.markdown(f"##### {t('radar_table_title')}")
    st.caption(t("filter_result_count", n=len(filtered)))

    if filtered.empty:
        st.info(t("filter_no_result"))
    else:
        display_df = filtered[[
            "stock_id", name_col, "close", "relative_strength", "volume_ratio",
            "rsi_14", "institutional_total_net", "is_breakout", "date",
        ]].rename(columns={
            "stock_id": t("col_stock_id"), name_col: t("col_name"), "close": t("col_close"),
            "relative_strength": t("col_rs"), "volume_ratio": t("col_vol_ratio"),
            "rsi_14": t("col_rsi"), "institutional_total_net": t("col_inst_net"),
            "is_breakout": t("col_vol_breakout"), "date": t("col_date"),
        })
        st.table(
            display_df.style.format({
                t("col_close"): format_close_price,
                t("col_rs"): "{:.1f}", t("col_vol_ratio"): "{:.2f}", t("col_rsi"): "{:.1f}",
                t("col_inst_net"): "{:,.0f}",
            }, na_rep="-").hide(axis="index")
        )

with tab1:
    st.subheader(t("price_trend", sid=stock_label(stock_id, names_df)))
    fig = go.Figure(data=[go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=t("kline"),
    )])
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma5"], name="MA5", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma60"], name="MA60", line=dict(width=1)))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, yaxis_title=t("price_axis"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    latest = df.iloc[-1]
    col1.metric(t("latest_close"), f"{latest['close']:.1f}")
    col2.metric(t("volume"), f"{latest['volume']:,.0f}")
    col3.metric(t("inst_net_total"), f"{latest['institutional_total_net']:,.0f}")

with tab2:
    st.subheader(t("technical_indicators"))

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], name="RSI(14)"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(height=250, title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal"))
    fig_macd.update_layout(height=250, title=t("macd_title"))
    st.plotly_chart(fig_macd, use_container_width=True)

    fig_kd = go.Figure()
    fig_kd.add_trace(go.Scatter(x=df["date"], y=df["kd_k"], name="K"))
    fig_kd.add_trace(go.Scatter(x=df["date"], y=df["kd_d"], name="D"))
    fig_kd.update_layout(height=250, title="KD")
    st.plotly_chart(fig_kd, use_container_width=True)

with tab3:
    st.subheader(t("chips"))

    fig_inst = go.Figure()
    fig_inst.add_trace(go.Bar(x=df["date"], y=df["foreign_net"], name=t("foreign")))
    fig_inst.add_trace(go.Bar(x=df["date"], y=df["investment_trust_net"], name=t("trust")))
    fig_inst.add_trace(go.Bar(x=df["date"], y=df["dealer_net"], name=t("dealer")))
    fig_inst.update_layout(
        height=350, barmode="relative", title=t("inst_net_shares"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        margin=dict(t=80),
    )
    st.plotly_chart(fig_inst, use_container_width=True)

    fig_margin = go.Figure()
    fig_margin.add_trace(go.Scatter(x=df["date"], y=df["margin_balance"], name=t("margin_balance")))
    fig_margin.add_trace(go.Scatter(x=df["date"], y=df["short_balance"], name=t("short_balance"),
                                     yaxis="y2"))
    fig_margin.update_layout(
        height=350, title=t("margin_short_title"),
        yaxis=dict(title=t("margin_balance")),
        yaxis2=dict(title=t("short_balance"), overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        margin=dict(t=80),
    )
    st.plotly_chart(fig_margin, use_container_width=True)

with tab4:
    st.subheader(t("watchlist_subheader"))

    detail_df = load_watchlist_detail()
    name_col = "name_zh" if st.session_state.get("lang", "zh") == "zh" else "name_en"
    detail_df = detail_df.merge(names_df[["stock_id", name_col]], on="stock_id", how="left")

    col1, col2 = st.columns(2)
    col1.metric(t("watchlist_defined_count", n=len(WATCHLIST)), "")
    col2.metric(t("watchlist_actual_count", n=len(detail_df)), "")

    missing = sorted(set(WATCHLIST) - set(detail_df["stock_id"]))
    if missing:
        st.warning(t("watchlist_missing", codes="、".join(missing)))
    else:
        st.success(t("watchlist_missing_none"))

    display_watchlist = detail_df[["stock_id", name_col, "latest_date", "close"]].rename(columns={
        "stock_id": t("col_stock_id"),
        name_col: t("col_name"),
        "latest_date": t("col_latest_date"),
        "close": t("col_close"),
    })
    st.table(
        display_watchlist.style.format({t("col_close"): format_close_price}).hide(axis="index")
    )
