# -*- coding: utf-8 -*-
"""分頁四：AI 多模態月度預測"""
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import FORECAST_DAYS, LOOKBACK_DAYS, NEWS_QUERY_TERMS
from core import datasource, model as M, news as N, sentiment as S, storage
from core.indicators import make_rangebreaks

ICONS = {"上升型": "🟢", "下降型": "🔴", "震盪型": "🟡"}


def _train(target, source_key, source_label, years):
    status = st.status("正在建立 AI 預測模型...", expanded=True)

    status.write("1/6 下載歷史股價...")
    stock_df = datasource.get_stock_data(target, source_key, years)
    if stock_df.empty:
        raise ValueError("兩個資料來源都取不到資料。請展開「資料來源健檢」查看原因。")
    status.write(f"　　取得 {len(stock_df)} 筆（{stock_df.index.min().date()} ~ {stock_df.index.max().date()}）")

    status.write("2/6 抓取新聞並執行 FinBERT 情緒分析...")
    name = datasource.get_stock_name(target)
    news_df, _ = N.fetch_news(N.build_queries(name, NEWS_QUERY_TERMS))
    scored = S.score_news(news_df)
    senti = S.build_features(stock_df.index, scored)
    nonzero = int((senti["sentiment_score_raw"] != 0).sum())
    status.write(f"　　新聞 {len(news_df)} 則；有情緒值的交易日 {nonzero} / {len(senti)} 天")

    status.write("3/6 計算技術指標與 KMeans 市場狀態...")
    feat = M.prepare_features(stock_df, senti)
    feat, cluster_id, state, summary, km_err = M.attach_state(feat)
    if km_err:
        status.write(f"　　⚠️ KMeans 失敗：{km_err}（狀態特徵補 0）")

    status.write("4/6 建立資料集（purged split）...")
    X_tr, y_tr, X_va, y_va, clean, f_sc, t_sc, info = M.make_dataset(feat)
    status.write(f"　　{info}")

    status.write("5/6 訓練 TCN...")
    bar = st.progress(0.0)
    mdl, hist = M.train_model(
        X_tr, y_tr, X_va, y_va, LOOKBACK_DAYS, FORECAST_DAYS,
        progress_cb=lambda ep, tot, logs: bar.progress(
            min(ep / tot, 1.0), text=f"Epoch {ep}/{tot}  loss={logs.get('loss', 0):.5f}"))
    bar.empty()

    status.write("6/6 診斷與預測...")
    diag = M.diagnose(mdl, X_va, y_va, t_sc)
    forecast = M.predict_future(mdl, clean, f_sc, t_sc)

    result = {
        "target": target, "source_key": source_key, "source": source_label, "years": years,
        "stock_df": stock_df, "forecast_df": forecast, "sentiment_df": senti,
        "scored_news": scored, "latest_state": state, "latest_cluster": cluster_id,
        "cluster_summary": summary, "diag": diag, "split_info": info,
        "history": hist.history, "news_count": len(news_df), "sentiment_nonzero": nonzero,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    path = storage.save_result(result)
    n_logged = storage.append_prediction_log(result)     # 只增不刪，供日後比對
    status.update(label=f"✅ 完成：結果存至 {path.name}，並寫入 {n_logged} 筆預測紀錄",
                  state="complete", expanded=False)
    return result


def _show_health(diag, info):
    with st.container(border=True):
        st.subheader("🩺 模型健檢：與 naive baseline 的對照")
        st.caption("單看 MAE 沒有意義。要看的是：有沒有贏過「永遠預測 0」的基準，"
                   "以及模型輸出是不是接近常數。")
        d1, d2, d3 = st.columns(3)
        d1.metric("模型 MAE", f"{diag['mae_model']:.5f}",
                  f"{diag['mae_model'] - diag['mae_naive']:+.5f} vs 基準")
        d2.metric("基準 MAE（永遠猜 0）", f"{diag['mae_naive']:.5f}")
        d3.metric("預測/實際 標準差比", f"{diag['std_ratio']:.3f}",
                  help="低於 0.10 代表模型輸出幾乎是常數，等同沒有預測能力")
        st.info(diag["verdict"])

        with st.expander("完整診斷數值"):
            st.dataframe(pd.DataFrame([{
                "模型 MAE": diag["mae_model"], "基準 MAE": diag["mae_naive"],
                "模型 RMSE": diag["rmse_model"], "基準 RMSE": diag["rmse_naive"],
                "預測標準差": diag["std_pred"], "實際標準差": diag["std_actual"],
                "標準差比": diag["std_ratio"], "方向準確率": diag["direction_acc"],
            }]).T.rename(columns={0: "數值"}), use_container_width=True)
            st.write("資料切分：", info)


def _show_chart(res):
    stock_df, forecast = res["stock_df"], res["forecast_df"]
    last_close = float(stock_df["Close"].iloc[-1])

    with st.container(border=True):
        st.subheader("📊 歷史走勢 + 未來預測（含不確定性區間）")
        g1, g2 = st.columns(2)
        hist_days = g1.slider("歷史顯示天數", 20, 120, 60, key="t4_hist")
        fc_days = g2.slider("未來顯示天數", 1, FORECAST_DAYS, FORECAST_DAYS, key="t4_fc")

        hist, fc = stock_df.tail(hist_days), forecast.iloc[:fc_days]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name="歷史 K 線",
            increasing_line_color="#FF0000", increasing_fillcolor="#FF0000",
            decreasing_line_color="#00AA00", decreasing_fillcolor="#00AA00"))

        fig.add_trace(go.Scatter(x=fc.index, y=fc["Upper"], line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["Lower"], fill="tonexty",
                                 fillcolor="rgba(70,130,220,0.18)", line=dict(width=0),
                                 name="不確定性區間 (±1σ√t)", hoverinfo="skip"))

        fig.add_trace(go.Scatter(
            x=[stock_df.index[-1]] + list(fc.index), y=[last_close] + list(fc["Close"]),
            mode="lines+markers", name="AI 預測收盤價",
            line=dict(color="#1f77b4", width=2.5, dash="dash"), marker=dict(size=5),
            hovertemplate="%{x|%Y-%m-%d}<br>預測收盤: %{y:.2f}<extra></extra>"))

        fig.add_vline(x=stock_df.index[-1], line_dash="dot", line_color="gray")
        fig.update_layout(
            title=f"{res['target']} 歷史走勢與 AI 預測 ｜ KMeans：{res['latest_state']}",
            height=560, template="plotly_white", hovermode="x unified",
            xaxis_rangeslider_visible=False, yaxis_title="價格",
            xaxis=dict(rangebreaks=[dict(values=make_rangebreaks(hist.index.union(fc.index)))]))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ 陰影為以近 60 日波動率推估的不確定性範圍（±1σ√t），隨天數擴大。"
                   "本結果僅供學術展示，不構成投資建議。")

        st.subheader(f"🔮 未來 {fc_days} 交易日預測明細")
        out = fc.copy()
        out.insert(0, "日期", out.index.strftime("%Y-%m-%d"))
        out["預測漲跌幅(%)"] = out["Return"] * 100
        out = out.rename(columns={"Close": "預測收盤", "Upper": "區間上限", "Lower": "區間下限"})
        st.dataframe(out[["日期", "預測收盤", "區間下限", "區間上限", "預測漲跌幅(%)"]],
                     use_container_width=True, hide_index=True, column_config={
                         "預測收盤": st.column_config.NumberColumn(format="%.2f"),
                         "區間下限": st.column_config.NumberColumn(format="%.2f"),
                         "區間上限": st.column_config.NumberColumn(format="%.2f"),
                         "預測漲跌幅(%)": st.column_config.NumberColumn(format="%+.3f %%")})


def _show_verification(stock_id: str, source_key: str):
    """把過去的預測跟實際股價比對。這是整個專題最有說服力的一張圖。"""
    log = storage.load_prediction_log(stock_id)
    if log.empty:
        return

    today = pd.Timestamp(datetime.date.today())
    due = log[log["target_date"] < today].copy()

    with st.container(border=True):
        st.subheader("🔍 預測 vs 實際（事後驗證）")
        st.caption("每次訓練都會留下紀錄。時間過去後，這裡會自動比對當初的預測與真實走勢。")

        c1, c2, c3 = st.columns(3)
        c1.metric("累積預測筆數", f"{len(log)}")
        c2.metric("已到期可驗證", f"{len(due)}")
        c3.metric("預測批次", f"{log['run_id'].nunique()} 次")

        if due.empty:
            st.info("📅 目前還沒有到期的預測。等幾天後再回來看，就會出現比對結果。")
            return

        start = due["target_date"].min() - pd.Timedelta(days=7)
        end = today + pd.Timedelta(days=1)
        actual = datasource.load_yahoo(f"{stock_id}.TW",
                                       start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if actual.empty:
            actual = datasource.load_finmind(stock_id, start.strftime("%Y-%m-%d"),
                                             end.strftime("%Y-%m-%d"))
        if actual.empty:
            st.warning("抓不到實際股價，無法比對。")
            return

        due["actual_close"] = due["target_date"].map(actual["Close"])
        due = due.dropna(subset=["actual_close"])
        if due.empty:
            st.info("到期的預測日都不是交易日，暫無可比對資料。")
            return

        due["誤差"] = due["pred_close"] - due["actual_close"]
        due["誤差率(%)"] = due["誤差"] / due["actual_close"] * 100
        due["命中區間"] = (due["actual_close"] >= due["lower"]) & (due["actual_close"] <= due["upper"])
        due["方向正確"] = (np.sign(due["pred_close"] - due["base_close"])
                       == np.sign(due["actual_close"] - due["base_close"]))

        m1, m2, m3 = st.columns(3)
        m1.metric("平均絕對誤差率", f"{due['誤差率(%)'].abs().mean():.2f}%")
        m2.metric("落在預測區間內", f"{due['命中區間'].mean():.1%}",
                  help="理想上 ±1σ 區間約應涵蓋 68% 的實際值")
        m3.metric("方向正確率", f"{due['方向正確'].mean():.1%}",
                  help="相對於做預測當天的收盤價，漲跌方向是否猜對")

        latest_run = due.sort_values("predicted_at").iloc[-1]["run_id"]
        sub = due[due["run_id"] == latest_run].sort_values("target_date")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["target_date"], y=sub["upper"], line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=sub["target_date"], y=sub["lower"], fill="tonexty",
                                 fillcolor="rgba(70,130,220,0.15)", line=dict(width=0),
                                 name="當初的預測區間", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sub["target_date"], y=sub["pred_close"], name="當初的預測",
                                 line=dict(color="#1f77b4", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=sub["target_date"], y=sub["actual_close"], name="實際走勢",
                                 line=dict(color="#d62728", width=2.5)))
        fig.update_layout(
            title=f"{stock_id}　{sub['predicted_at'].iloc[0]} 做出的預測 vs 實際",
            height=420, template="plotly_white", hovermode="x unified", yaxis_title="價格")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("逐日比對明細"):
            show = due.sort_values("target_date", ascending=False).head(60)
            st.dataframe(pd.DataFrame({
                "預測日期": show["predicted_at"],
                "目標日": show["target_date"].dt.strftime("%Y-%m-%d"),
                "預測收盤": show["pred_close"].round(2),
                "實際收盤": show["actual_close"].round(2),
                "誤差率(%)": show["誤差率(%)"].round(2),
                "落在區間": show["命中區間"].map({True: "✅", False: "❌"}),
                "方向": show["方向正確"].map({True: "✅", False: "❌"}),
            }), use_container_width=True, hide_index=True)

        st.download_button("⬇️ 下載完整預測紀錄 (CSV)",
                           data=storage.load_prediction_log().to_csv(index=False).encode("utf-8-sig"),
                           file_name="prediction_log.csv", mime="text/csv", key="t4_dl_log")


def render():
    with st.container(border=True):
        st.subheader("🤖 AI 多模態月度預測")
        st.caption("KMeans 市場狀態（one-hot 進入模型）＋ FinBERT 新聞情緒 ＋ TCN 多步報酬率預測")

        a1, a2, a3 = st.columns([1.5, 1.2, 1.3])
        target = a1.text_input("輸入台股 / ETF 代碼：", "0050", key="t4_target")
        target = target.strip().upper().replace(".TW", "").replace(".TWO", "")
        label = a2.radio("模型資料來源：", ("Yahoo Finance", "FinMind"),
                         horizontal=True, key="t4_source")
        source_key = "yfinance" if label == "Yahoo Finance" else "finmind"
        years = a3.selectbox("模型歷史資料：", [3, 5, 7, 10], index=1,
                             format_func=lambda x: f"近 {x} 年", key="t4_years")

        if M.TF_AVAILABLE:
            modes = ["📂 讀取已存結果（展示建議）", "🔥 重新訓練（需數分鐘）"]
        else:
            modes = ["📂 讀取已存結果（展示建議）"]
            st.info("ℹ️ 本環境未安裝 TensorFlow，僅能讀取先前訓練好的結果。")
        mode = st.radio("執行模式：", modes, horizontal=True, key="t4_mode")

        saved = storage.list_results()
        st.caption(f"已存結果：{', '.join(saved)}" if saved
                   else "尚無已存結果，第一次請選「重新訓練」跑一次。")

        if not datasource.FINMIND_LOGGED_IN:
            st.caption("ℹ️ FinMind 未登入（限額 300 次/小時）。抓不到資料時請到「🩺 系統健檢」分頁診斷。")

        go_btn = st.button("🚀 執行", use_container_width=True, key="t4_btn")

    if go_btn:
        if not (target.isalnum() and 4 <= len(target) <= 6):
            st.error("❌ 代碼格式錯誤，請輸入 4～6 位英數代碼，例如 0050、2330。")
        elif mode.startswith("📂"):
            loaded = storage.load_result(target, source_key, years)
            if loaded is None:
                st.error(f"❌ 找不到 {target} / {label} / {years} 年 的已存結果，請先重新訓練。")
            else:
                st.session_state["t4_result"] = loaded
                st.success("✅ 已載入預存結果。")
        else:
            try:
                st.session_state["t4_result"] = _train(target, source_key, label, years)
            except Exception as e:
                st.error(f"❌ 執行失敗：{type(e).__name__} — {e}")
                st.info("先展開上方「資料來源健檢」確認 API 是否正常；"
                        "TCN 建議使用 5 年以上資料。詳細錯誤請看 data/app.log。")

    res = st.session_state.get("t4_result")
    if res is None:
        return

    diag, stock_df, forecast = res["diag"], res["stock_df"], res["forecast_df"]
    icon = ICONS.get(res["latest_state"], "⚪")
    last_close = float(stock_df["Close"].iloc[-1])
    final_close = float(forecast["Close"].iloc[-1])

    st.success(f"✅ 標的 **{res['target']}** ｜來源：{res['source']} ｜"
               f"歷史：{res['years']} 年 ｜產生時間：{res.get('created_at', '-')}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("目前收盤價", f"{last_close:.2f}")
    r2.metric("KMeans 市場狀態", f"{icon} {res['latest_state']}")
    r3.metric(f"{FORECAST_DAYS} 日後預測價", f"{final_close:.2f}",
              f"{(final_close / last_close - 1) * 100:+.2f}%")
    r4.metric("方向準確率", f"{diag['direction_acc']:.1%}")

    _show_health(diag, res.get("split_info", {}))
    _show_chart(res)

    k1, k2 = st.columns(2)
    with k1:
        with st.container(border=True):
            st.subheader("🧭 KMeans 市場狀態")
            st.metric("目前狀態", f"{icon} {res['latest_state']}")
            cs = res.get("cluster_summary")
            if cs is not None and not cs.empty:
                st.dataframe(cs.round(4), use_container_width=True)
            st.caption("此狀態已以 one-hot 形式加入 TCN 的輸入特徵。")

    with k2:
        with st.container(border=True):
            st.subheader("📉 TCN 訓練 / 驗證 Loss")
            h = res.get("history", {})
            if h and "loss" in h:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=h["loss"], name="Training Loss"))
                if "val_loss" in h:
                    fig.add_trace(go.Scatter(y=h["val_loss"], name="Validation Loss"))
                fig.update_layout(height=300, template="plotly_white",
                                  xaxis_title="Epoch", yaxis_title="MSE Loss")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("（無訓練曲線）")

    _show_verification(res["target"], res["source_key"])

    with st.container(border=True):
        st.subheader("📰 情緒特徵覆蓋率")
        sdf = res["sentiment_df"]
        nz = res.get("sentiment_nonzero", int((sdf["sentiment_score_raw"] != 0).sum()))
        cov = nz / len(sdf) * 100 if len(sdf) else 0

        f1, f2, f3 = st.columns(3)
        f1.metric("抓到新聞則數", res.get("news_count", 0))
        f2.metric("有情緒值的交易日", f"{nz} / {len(sdf)}")
        f3.metric("特徵覆蓋率", f"{cov:.1f}%")

        if cov < 20:
            st.warning(f"⚠️ 覆蓋率僅 {cov:.1f}%。Google News RSS 只能取得近期新聞，"
                       "因此多數歷史交易日的情緒值為 0，FinBERT 對模型的實際貢獻有限。"
                       "建議在書面報告誠實說明此限制。")
