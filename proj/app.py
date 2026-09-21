# -*- coding: utf-8 -*-
"""
ETF 損益分析與預測系統 —— 主程式

這個檔案只負責「組裝分頁」，每個分頁的實作在 tabs/ 資料夾。
執行：streamlit run app.py
"""
import os

# 1. 強制鎖定 TensorFlow 只用 1 個 CPU 執行緒（防止 Thread 競爭導致 Segfault）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# 2. 停用 oneDNN 自訂運算（能大幅降低 C++ 層級記憶體崩潰的機率）
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 3. 隱藏不必要的 TensorFlow C++ Log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import tensorflow as tf  # 必須放在 os.environ 設定之後！
import streamlit as st

from config import APP_TITLE
from tabs import (tab0_health, tab1_technical, tab2_dca,
                  tab3_scanner, tab4_predict, tab5_news)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"📈 {APP_TITLE}")
st.markdown("---")

TABS = [
    ("🩺 系統健檢", tab0_health.render),
    ("📊 個股歷史趨勢與技術分析", tab1_technical.render),
    ("💰 定期定額回測", tab2_dca.render),
    ("📺 市場趨勢即時掃描", tab3_scanner.render),
    ("🤖 AI 多模態月度預測", tab4_predict.render),
    ("📰 新聞情緒分析引擎", tab5_news.render),
]

for tab, (_, render) in zip(st.tabs([name for name, _ in TABS]), TABS):
    with tab:
        render()
