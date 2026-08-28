"""
i18n.py
--------
共用的語言切換元件。

每個 Streamlit 頁面（dashboard.py 以及 pages/ 資料夾底下的每一支）
只要在檔案最上方呼叫一次 init_language()，就能確保：

1. 不管使用者從哪個頁面進來，側邊欄都會出現「語言 / Language」切換選單
   （不再只有主頁面才有，其他頁面之前只能被動讀取語言、沒辦法切換）
2. 所有頁面共用同一個 st.session_state["lang"]，在任何一頁切換語言，
   切到別的頁面時語言設定會維持一致，不會跳回中文
"""
import streamlit as st


def init_language() -> str:
    """
    在側邊欄渲染語言切換單選鈕，回傳目前選定的語言代碼（"zh" 或 "en"）。
    請在每個頁面檔案最上方（st.set_page_config 之後）呼叫這個函式。
    """
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
    """單純讀取目前語言，不渲染任何 UI 元件（給不需要顯示切換按鈕、只想讀值的地方用）。"""
    return st.session_state.get("lang", "zh")
