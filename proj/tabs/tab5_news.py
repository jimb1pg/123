# -*- coding: utf-8 -*-
"""分頁五：新聞情緒分析引擎（含本機資料累積）"""
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import NEWS_QUERY_TERMS
from core import news as N, sentiment as S, storage
from core.datasource import get_stock_name


def _inputs():
    n1, n2 = st.columns([2, 1])
    with n1:
        mode = st.radio("搜尋模式：", ["📈 依股票代碼（自動組合關鍵字）", "🔍 自由關鍵字搜尋"],
                        horizontal=True, key="t5_mode")
        if mode.startswith("📈"):
            code = st.text_input("股票 / ETF 代碼：", "2330", key="t5_code")
            code = code.strip().upper().replace(".TW", "")
            name = get_stock_name(code) if code else ""
            if name:
                st.caption(f"辨識名稱：**{name}**")
            terms = st.multiselect("要組合的主題關鍵字（可增減）", NEWS_QUERY_TERMS,
                                   default=NEWS_QUERY_TERMS[:8], key="t5_terms")
            return N.build_queries(name, terms), name, code
        raw = st.text_input("自由輸入關鍵字（逗號分隔，可多組）",
                            "台積電 法說會, 半導體 景氣, 輝達 訂單", key="t5_free")
        return tuple(q.strip() for q in raw.split(",") if q.strip()), "自訂搜尋", ""


def _overview(scored):
    pos = scored.loc[scored["Sentiment"] > 0, "Weight"].sum() * 100
    neg = scored.loc[scored["Sentiment"] < 0, "Weight"].sum() * 100
    neu = scored.loc[scored["Sentiment"] == 0, "Weight"].sum() * 100
    overall = float(np.average(scored["Sentiment"], weights=scored["Confidence"])) \
        if scored["Confidence"].sum() > 0 else 0.0
    direction = "🟢 偏正向" if overall > 0.05 else ("🔴 偏負向" if overall < -0.05 else "🟡 中立")

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("整體情緒分數", f"{overall:+.3f}")
    o2.metric("情緒方向", direction)
    o3.metric("正向權重", f"{pos:.1f}%")
    o4.metric("負向權重", f"{neg:.1f}%")
    return pos, neu, neg


def _charts(scored, smooth, pos, neu, neg):
    left, right = st.columns([1.3, 1])

    with left:
        with st.container(border=True):
            st.subheader("📈 每日情緒趨勢")
            daily = S.daily_sentiment(scored, smooth_window=smooth)
            if not daily.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=daily.index, y=daily["sentiment_score_raw"], name="每日情緒", opacity=0.55,
                    marker_color=np.where(daily["sentiment_score_raw"] >= 0, "#d62728", "#2ca02c")))
                fig.add_trace(go.Scatter(x=daily.index, y=daily["sentiment_score"],
                                         name=f"{smooth} 日平滑",
                                         line=dict(color="#1f77b4", width=2.5)))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=380, template="plotly_white", hovermode="x unified",
                                  yaxis_title="情緒分數 (-1 ~ +1)", yaxis_range=[-1.05, 1.05])
                st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            st.subheader("🥧 情緒權重分布")
            fig = go.Figure(go.Pie(labels=["正向", "中立", "負向"], values=[pos, neu, neg],
                                   marker_colors=["#d62728", "#c7c7c7", "#2ca02c"], hole=0.45))
            fig.update_layout(height=380, template="plotly_white", margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)


def _news_table(scored):
    with st.container(border=True):
        st.subheader("📋 新聞明細")
        f1, f2 = st.columns(2)
        filt = f1.selectbox("情緒篩選", ["全部", "只看正向", "只看負向", "只看中立"], key="t5_filter")
        topn = f2.slider("顯示則數", 5, max(len(scored), 5), min(30, len(scored)), key="t5_topn")

        view = scored.copy()
        if filt == "只看正向":
            view = view[view["Sentiment"] > 0]
        elif filt == "只看負向":
            view = view[view["Sentiment"] < 0]
        elif filt == "只看中立":
            view = view[view["Sentiment"] == 0]

        view = view.sort_values(["Date", "Weight"], ascending=[False, False]).head(topn)
        if view.empty:
            st.info("此篩選條件下沒有新聞。")
            return

        show = pd.DataFrame({
            "日期": view["Date"].dt.strftime("%Y-%m-%d"),
            "情緒": view["Sentiment"].apply(
                lambda s: "🟢 正向" if s > 0 else ("🔴 負向" if s < 0 else "🟡 中立")),
            "標題": view["Title"],
            "分數": view["Sentiment"].round(3),
            "信心": view["Confidence"].round(3),
            "權重(%)": view["Weight_Pct"].round(2),
            "命中關鍵字數": view["Keyword_Count"],
            "連結": view["Link"],
        })
        st.dataframe(show, use_container_width=True, hide_index=True, column_config={
            "連結": st.column_config.LinkColumn("原文", display_text="開啟"),
            "權重(%)": st.column_config.NumberColumn(format="%.2f")})

        st.download_button("⬇️ 下載本次結果 (CSV)",
                           data=scored.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"news_sentiment_{datetime.date.today()}.csv",
                           mime="text/csv", key="t5_dl")


def _archive_panel():
    with st.container(border=True):
        st.subheader("🗄️ 本機新聞資料庫")
        st.caption("每次分析都可以存檔累積。長期累積後就能觀察情緒與股價的關係。")

        summary = storage.archive_summary()
        if summary.empty:
            st.info("目前還沒有任何存檔。分析完成後按「存入本機資料庫」，或執行 collect_news.py。")
            return

        a1, a2 = st.columns([1, 2])
        a1.metric("已累積天數", f"{len(summary)} 天")
        a1.metric("總新聞則數", f"{int(summary['則數'].clip(lower=0).sum())} 則")

        with a2:
            fig = go.Figure(go.Bar(x=summary["日期"], y=summary["則數"], marker_color="#1f77b4"))
            fig.update_layout(height=220, template="plotly_white",
                              margin=dict(t=20, b=20), yaxis_title="則數")
            st.plotly_chart(fig, use_container_width=True)

        archive = storage.load_news_archive()
        if not archive.empty and "Sentiment" in archive.columns:
            daily = S.daily_sentiment(archive.dropna(subset=["Date"]), smooth_window=5)
            if not daily.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily.index, y=daily["sentiment_score"],
                                         name="5 日平滑情緒", line=dict(color="#ff7f0e", width=2.5)))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(title="歷史累積情緒趨勢", height=320,
                                  template="plotly_white", yaxis_range=[-1.05, 1.05])
                st.plotly_chart(fig, use_container_width=True)


def _watchlist_editor():
    """讓使用者在介面上改每日蒐集清單，不用去動程式碼。"""
    with st.expander("⚙️ 每日蒐集清單設定（collect_news.py 會讀這份設定）"):
        wl = storage.load_watchlist()

        st.caption("每行一筆，格式：`代碼,名稱`。名稱會用來組合搜尋詞，例如「營收 台積電」。")
        stock_text = st.text_area(
            "追蹤標的", value="\n".join(f"{k},{v}" for k, v in wl["stocks"].items()),
            height=120, key="t5_wl_stocks")

        term_text = st.text_input(
            "搜尋關鍵字（逗號分隔）", value="、".join(wl["terms"]).replace("、", ","),
            key="t5_wl_terms")

        c1, c2 = st.columns([1, 3])
        if c1.button("💾 儲存設定", key="t5_wl_save"):
            stocks = {}
            for line in stock_text.splitlines():
                if "," in line:
                    code, name = line.split(",", 1)
                    if code.strip():
                        stocks[code.strip().upper()] = name.strip() or code.strip()
                elif line.strip():
                    stocks[line.strip().upper()] = line.strip().upper()

            terms = [t.strip() for t in term_text.split(",") if t.strip()]
            if not stocks or not terms:
                c2.error("標的與關鍵字都不能空白。")
            else:
                path = storage.save_watchlist(stocks, terms)
                c2.success(f"✅ 已儲存至 {path}")

        st.caption("儲存後，下次執行 `python collect_news.py` 就會用新的設定。"
                   "也可以臨時指定：`python collect_news.py --stocks 2330,2317 --terms 營收,訂單`")


def render():
    with st.container(border=True):
        st.subheader("📰 新聞情緒分析引擎")
        st.caption("Google News RSS 即時搜尋 → 去重 → FinBERT 情緒判讀 → 權重與趨勢分析\n\n首次使用請先到「🩺 系統健檢」分頁確認情緒模型正常。")

        queries, subject, code = _inputs()

        _, n2 = st.columns([2, 1])
        with n2:
            per_query = st.slider("每組關鍵字抓幾則", 5, 20, 10, key="t5_per")
            smooth = st.slider("情緒平滑天數", 1, 30, 5, key="t5_smooth")
            st.caption(f"預計送出 {len(queries)} 組查詢")

        _watchlist_editor()
        go_btn = st.button("🚀 開始搜尋並分析", use_container_width=True, key="t5_btn")

    if go_btn:
        if not queries:
            st.warning("請先輸入有效的搜尋關鍵字或股票代碼。")
        else:
            with st.spinner(f"正在搜尋 {len(queries)} 組關鍵字並執行情緒分析..."):
                news_df, failures = N.fetch_news(queries, per_query)
                scored = S.score_news(news_df)
            st.session_state["t5_result"] = {
                "news": news_df, "scored": scored, "failures": failures,
                "subject": subject, "code": code, "smooth": smooth}

    res = st.session_state.get("t5_result")
    if res is None:
        _archive_panel()
        return

    news_df, scored = res["news"], res["scored"]

    if res["failures"]:
        with st.expander(f"⚠️ 有 {len(res['failures'])} 組查詢失敗"):
            for m in res["failures"]:
                st.code(m)

    if news_df.empty:
        st.error("❌ 沒有抓到任何新聞。可能是網路問題或關鍵字太冷門。")
        _archive_panel()
        return

    if scored.empty:
        st.error("❌ 抓到新聞但情緒分析失敗（FinBERT 未成功載入）。請先執行上方的模型健檢。")
        _archive_panel()
        return

    st.success(f"✅ 分析完成：{res['subject']} ｜共 {len(scored)} 則不重複新聞")
    pos, neu, neg = _overview(scored)

    with st.expander("🩺 模型輸出標籤分布（確認情緒引擎是否正常）"):
        dist = scored["Label"].value_counts().rename_axis("模型標籤").reset_index(name="則數")
        dist["對應分數"] = dist["模型標籤"].apply(S.label_to_score)
        st.dataframe(dist, use_container_width=True, hide_index=True)
        if len(dist) == 1:
            st.warning("⚠️ 所有新聞被判為同一類，情緒引擎可能沒有正常運作。")
        if (dist["對應分數"] == 0).all():
            st.warning("⚠️ 所有標籤都對應到 0（中立），請檢查 label_to_score() 的對應規則。")

    _charts(scored, res["smooth"], pos, neu, neg)

    with st.container(border=True):
        st.subheader("🔑 各搜尋關鍵字的權重占比")
        kw = S.keyword_weight_table(scored)
        if not kw.empty:
            st.dataframe(kw, use_container_width=True, hide_index=True, column_config={
                "權重占比": st.column_config.ProgressColumn(
                    "權重占比 (%)", format="%.2f%%", min_value=0,
                    max_value=float(kw["權重占比"].max()))})
            st.caption("一則新聞若同時被多組關鍵字命中，其權重會平均分配給各關鍵字。")

    _news_table(scored)

    # 存入本機資料庫
    with st.container(border=True):
        st.subheader("💾 存入本機資料庫")
        st.caption("按一下就會把這次結果存到 data/news/ 底下，"
                   "長期累積後可以在下方看到情緒趨勢。")
        if st.button("存入本機資料庫", key="t5_save"):
            to_save = scored.copy()
            to_save["StockID"] = res["code"] or "custom"
            path = storage.save_daily_news(to_save)
            st.success(f"✅ 已存至 {path}") if path else st.warning("沒有可儲存的資料。")

    _archive_panel()
