"""
i18n.py
--------
共用的語言切換元件

每個 Streamlit 頁面（dashboard.py 以及 pages/ 資料夾底下的每一支）
只要在檔案最上方呼叫一次 init_language() 確保：
1. 不管使用者從哪個頁面進來側邊欄都會出現「語言 / Language」切換選單
2. 所有頁面共用同一個 st.session_state["lang"]

"""
import streamlit as st


def init_language() -> str:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "zh"

    lang_choice = st.sidebar.radio(
        "語言 / Language",
        options=["zh", "en"],
        format_func=lambda x: "繁體中文" if x == "zh" else "English",
        index=0 if st.session_state["lang"] == "zh" else 1,
        horizontal=True,
        key="lang_radio",  # 固定 key，確保每個頁面的元件狀態綁定同一個 session_state 欄位
    )
    st.session_state["lang"] = lang_choice
    return lang_choice


def get_lang() -> str:
    """單純讀取目前語言，不渲染任何 UI 元件"""
    return st.session_state.get("lang", "zh")
