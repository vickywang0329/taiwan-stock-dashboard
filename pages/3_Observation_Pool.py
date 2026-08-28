"""
Observation Pool — 當前觀察池
從原本的 dashboard.py 拆出，獨立成一個頁面。
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

from watchlist import WATCHLIST
import i18n

DB_CONFIG = {
    "user": st.secrets.get("db_user", "postgres"),
    "password": st.secrets["db_password"],
    "host": st.secrets["db_host"],
    "port": st.secrets.get("db_port", "5432"),
    "database": st.secrets.get("db_name", "postgres"),
}

TEXT = {
    "page_title": {"zh": "當前觀察池", "en": "Observation Pool"},
    "title": {"zh": "當前觀察池", "en": "Observation Pool"},
    "watchlist_subheader": {"zh": "觀察池：資料庫目前涵蓋的股票", "en": "Watchlist: stocks currently covered in the database"},
    "watchlist_defined_count": {"zh": "清單定義檔（watchlist.py）共 {n} 檔", "en": "Defined in watchlist.py: {n} stocks"},
    "watchlist_actual_count": {"zh": "資料庫實際涵蓋 {n} 檔", "en": "Actually covered in database: {n} stocks"},
    "watchlist_missing": {"zh": "尚未成功寫入資料庫的股票：{codes}",
                          "en": "Stocks not yet in the database: {codes}"},
    "watchlist_missing_none": {"zh": "清單裡所有股票都已成功寫入資料庫，兩邊數量一致。",
                               "en": "All stocks in the list are present in the database — counts match."},
    "col_stock_id": {"zh": "股票代碼", "en": "Stock code"},
    "col_name": {"zh": "公司名稱", "en": "Company name"},
    "col_latest_date": {"zh": "最新資料日期", "en": "Latest data date"},
    "col_close": {"zh": "收盤價(新台幣元)", "en": "Close (TWD)"},
    "refresh_button": {"zh": "🔄 重新整理資料", "en": "🔄 Refresh data"},
    "refresh_help": {"zh": "跑完更新腳本後點這裡，立即反映最新的觀察池與資料",
                     "en": "Click after running the update scripts to reflect the latest watchlist and data immediately"},
}


def t(key: str, **kwargs) -> str:
    lang = i18n.get_lang()
    template = TEXT[key][lang]
    return template.format(**kwargs) if kwargs else template


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


@st.cache_data(ttl=3600)
def load_all_latest() -> pd.DataFrame:
    """每檔股票最新一天的收盤價（供觀察池表格顯示用）"""
    engine = get_engine()
    query = text("""
        WITH ranked AS (
            SELECT m.date, m.stock_id, m.close,
                   ROW_NUMBER() OVER (PARTITION BY m.stock_id ORDER BY m.date DESC) AS rn
            FROM staging.daily_master m
        )
        SELECT date, stock_id, close FROM ranked WHERE rn = 1
    """)
    return pd.read_sql(query, engine)


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

i18n.init_language()

st.title(t("title"))

if st.sidebar.button(t("refresh_button"), help=t("refresh_help")):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("A project by I.H. Wang")

names_df = load_stock_names()
detail_df = load_watchlist_detail()

lang = i18n.get_lang()
name_col = "name_zh" if lang == "zh" else "name_en"
detail_df = detail_df.merge(names_df[["stock_id", name_col]], on="stock_id", how="left")

st.subheader(t("watchlist_subheader"))

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
