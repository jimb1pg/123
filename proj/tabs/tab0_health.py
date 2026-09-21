# -*- coding: utf-8 -*-
"""分頁零：系統健檢。第一次使用、或任何東西壞掉時，都先來這裡。"""
import importlib.metadata as md
import platform
import sys

import pandas as pd
import streamlit as st

from config import LOG_FILE
from core import datasource, model as M, sentiment as S, storage

PACKAGES = ["streamlit", "pandas", "numpy", "plotly", "yfinance", "FinMind",
            "scikit-learn", "transformers", "torch", "tensorflow", "tensorflow-cpu"]


def _versions() -> pd.DataFrame:
    rows = []
    for name in PACKAGES:
        try:
            rows.append({"套件": name, "版本": md.version(name), "狀態": "✅ 已安裝"})
        except md.PackageNotFoundError:
            rows.append({"套件": name, "版本": "-", "狀態": "— 未安裝"})
    return pd.DataFrame(rows)


def _gpu_note() -> str:
    """說明目前有沒有在用 GPU。Windows 上的 TensorFlow 一律是 CPU。"""
    if not M.TF_AVAILABLE:
        return "未安裝 TensorFlow —— TAB4 只能讀取已存結果。"
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as e:
        return f"TensorFlow 載入異常：{e}"

    if gpus:
        return f"✅ 偵測到 {len(gpus)} 個 GPU，訓練會使用 GPU 加速。"
    if platform.system() == "Windows":
        return ("ℹ️ 目前使用 CPU 訓練。TensorFlow 2.11 起已不支援 Windows 原生 GPU，"
                "需要 WSL2 才能用顯卡。本專題模型不大，CPU 訓練約 3～10 分鐘，"
                "不建議為了 GPU 重建環境。")
    return "ℹ️ 目前使用 CPU 訓練（未偵測到可用 GPU）。"


def render():
    st.subheader("🩺 系統健檢")
    st.caption("第一次使用請由上往下跑一次。之後任何分頁出問題，也先回來這裡檢查。")

    # ---------- 步驟一：資料來源 ----------
    with st.container(border=True):
        st.markdown("### 步驟一　股價資料來源")
        st.caption(
            f"FinMind 登入狀態："
            f"{'✅ 已登入（限額 600 次/小時）' if datasource.FINMIND_LOGGED_IN else '⚠️ 未登入（限額 300 次/小時）'}"
        )

        c1, c2 = st.columns([1, 3])
        code = c1.text_input("測試用代碼", "2330", key="h_code")
        if c2.button("執行資料來源健檢", key="h_ds", use_container_width=True):
            with st.spinner("測試中..."):
                st.session_state["h_ds_result"] = datasource.selftest(code or "2330")

        res = st.session_state.get("h_ds_result")
        if res is not None:
            st.dataframe(res, use_container_width=True, hide_index=True)
            ok = res["狀態"].str.contains("可用").sum()
            if ok >= 1:
                st.success(f"✅ 有 {ok} 個來源可用，系統可以正常運作。")
            else:
                st.error("❌ 沒有任何來源可用。請檢查網路，或查看下方系統紀錄。")
            st.caption("💡 `taiwan_stock_daily_adj` 顯示失敗是正常的 —— "
                       "還原股價 API 需要 FinMind 付費贊助帳號，程式已自動改用免費 API。")

    # ---------- 步驟二：情緒模型 ----------
    with st.container(border=True):
        st.markdown("### 步驟二　FinBERT 情緒模型")
        st.caption("首次執行需要下載模型（數百 MB），可能要等幾分鐘。")

        if st.button("執行情緒模型健檢", key="h_bert", use_container_width=True):
            with st.spinner("載入情緒模型中..."):
                st.session_state["h_bert_result"] = S.selftest()

        res = st.session_state.get("h_bert_result")
        if res is not None:
            name, table, errors = res
            if name is None:
                st.error("❌ 所有候選模型都載入失敗，TAB5 與情緒特徵無法運作。")
                for m in errors:
                    st.code(m)
            else:
                st.success(f"✅ 實際載入的模型：`{name}`")
                st.dataframe(table, use_container_width=True, hide_index=True)
                if table["模型標籤"].nunique() == 1:
                    st.warning("⚠️ 三句測試句被判為同一類，情緒引擎可能不正常。")
                elif (table["轉換分數"] == 0).all():
                    st.warning("⚠️ 所有標籤都對應到 0，請檢查 core/sentiment.py 的 label_to_score()。")
                else:
                    st.caption("正向 / 負向 / 中立三句都判斷正確，情緒引擎運作正常。")
                if errors:
                    with st.expander("以下候選模型載入失敗（已自動略過）"):
                        for m in errors:
                            st.code(m)

    # ---------- 步驟三：執行環境 ----------
    with st.container(border=True):
        st.markdown("### 步驟三　執行環境")
        st.info(_gpu_note())

        e1, e2 = st.columns([1, 2])
        with e1:
            st.metric("Python", platform.python_version())
            st.caption(f"作業系統：{platform.system()} {platform.release()}")
            st.caption(f"TAB4 訓練功能：{'✅ 可用' if M.TF_AVAILABLE else '❌ 未安裝 TensorFlow'}")
        with e2:
            st.dataframe(_versions(), use_container_width=True, hide_index=True, height=250)

    # ---------- 步驟四：資料存量 ----------
    with st.container(border=True):
        st.markdown("### 步驟四　本機資料存量")

        d1, d2 = st.columns(2)
        with d1:
            results = storage.list_results()
            st.metric("已存預測結果", f"{len(results)} 份")
            if results:
                for r in results:
                    st.caption(f"　• {r}")
            else:
                st.caption("　尚無，請到 TAB4 選「重新訓練」跑一次。")

        with d2:
            summary = storage.archive_summary()
            st.metric("新聞累積天數", f"{len(summary)} 天")
            if not summary.empty:
                st.metric("新聞總則數", f"{int(summary['則數'].clip(lower=0).sum())} 則")
            else:
                st.caption("　尚無，請執行 collect_news.py 或到新聞分頁存檔。")

            plog = storage.load_prediction_log()
            st.metric("預測紀錄", f"{len(plog)} 筆" if not plog.empty else "0 筆")
            if not plog.empty:
                st.caption(f"　共 {plog['run_id'].nunique()} 個批次，可在 AI 預測分頁做事後驗證")

        wl = storage.load_watchlist()
        st.caption(
            "每日蒐集清單："
            + "、".join(f"{k} {v}" for k, v in wl["stocks"].items())
            + f"　｜關鍵字 {len(wl['terms'])} 組（可在新聞分頁修改）"
        )

    # ---------- 系統紀錄 ----------
    with st.expander("📋 系統紀錄（任何東西壞掉時來這裡看真正的原因）"):
        if LOG_FILE.exists():
            try:
                lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
                if lines:
                    st.code("\n".join(lines[-60:]), language="text")
                    st.caption(f"顯示最後 60 行，完整檔案：{LOG_FILE}")
                else:
                    st.caption("紀錄檔是空的（代表目前沒有錯誤）。")
            except Exception as e:
                st.caption(f"讀取失敗：{e}")
        else:
            st.caption("尚未產生紀錄檔（代表目前沒有錯誤）。")

    st.divider()
    st.markdown(
        "#### 下一步\n"
        "1. 上面兩項健檢都通過後，到 **🤖 AI 多模態月度預測** 選「🔥 重新訓練」跑一次，結果會自動存檔\n"
        "2. 之後展示時改選「📂 讀取已存結果」，秒開\n"
        "3. 到 **📰 新聞情緒分析引擎** 搜尋並按「存入本機資料庫」，開始累積每日資料"
    )
