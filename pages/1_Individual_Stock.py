"""
Individual Stock — 個股總覽
從原本的 dashboard.py 拆出，只保留總覽／技術訊號／籌碼面三個 tab，
移除選股雷達（已由 Swing_Trading_Decision_System.py 的決策引擎取代）
與觀察池（獨立拆到 Observation_Pool.py）。
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

import i18n

# ---- 資料庫連線設定 ----
DB_CONFIG = {
    "user": st.secrets.get("db_user", "postgres"),
    "password": st.secrets["db_password"],
    "host": st.secrets["db_host"],
    "port": st.secrets.get("db_port", "5432"),
    "database": st.secrets.get("db_name", "postgres"),
}

TEXT = {
    "page_title": {"zh": "個股總覽", "en": "Individual Stock"},
    "title": {"zh": "個股總覽", "en": "Individual Stock"},
    "data_updated": {"zh": "資料更新至 {date}", "en": "Data as of {date}"},
    "stock_code": {"zh": "股票代碼", "en": "Stock code"},
    "lookback_days": {"zh": "顯示天數", "en": "Days to show"},
    "watchlist_count": {"zh": "觀察池目前共有 {n} 檔股票", "en": "Watchlist currently has {n} stocks"},
    "no_data_db": {"zh": "資料庫裡目前沒有任何股票資料，請先執行資料抓取與轉換腳本。",
                   "en": "No stock data found in the database. Please run the fetch and transform scripts first."},
    "no_data_stock": {"zh": "找不到股票 {sid} 的資料，請確認資料庫裡已經有這檔股票。",
                      "en": "No data found for stock {sid}. Please confirm it exists in the database."},
    "tab_overview": {"zh": "總覽", "en": "Overview"},
    "tab_technical": {"zh": "技術訊號", "en": "Technical signals"},
    "tab_chips": {"zh": "籌碼面", "en": "Institutional flows"},
    "refresh_button": {"zh": "🔄 重新整理資料", "en": "🔄 Refresh data"},
    "refresh_help": {"zh": "跑完更新腳本後點這裡，立即反映最新的觀察池與資料",
                     "en": "Click after running the update scripts to reflect the latest watchlist and data immediately"},
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


def t(key: str, **kwargs) -> str:
    lang = i18n.get_lang()
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
    lang = i18n.get_lang()
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


st.set_page_config(page_title=t("page_title"), layout="wide")
i18n.init_language()

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

tab1, tab2, tab3 = st.tabs([t("tab_overview"), t("tab_technical"), t("tab_chips")])

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
