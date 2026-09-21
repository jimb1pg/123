# -*- coding: utf-8 -*-
"""分頁一：個股歷史趨勢與技術分析"""
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import OHLCV
from core.datasource import get_exchange_rate, load_finmind, load_yahoo
from core.indicators import calc_rsi, make_rangebreaks

PRESETS = ["0050.TW (元大台灣50)", "0056.TW (元大高股息)",
           "2330.TW (台積電)", "AAPL (蘋果)", "🔍 自訂輸入代碼..."]


def _hex_to_rgba(hex_color, opacity=0.3):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{opacity})"


def _settings():
    """畫出設定區，回傳使用者的選擇。"""
    c1, c2, c3 = st.columns([1.5, 1.5, 2])

    with c1:
        sel = st.selectbox("選擇查詢標的：", PRESETS, key="t1_preset")
        ticker = st.text_input("請輸入代碼 (台股請加 .TW)：", "00940.TW", key="t1_custom") \
            if sel == "🔍 自訂輸入代碼..." else sel.split(" ")[0]
        source = st.radio("優先資料來源：", ("Yahoo Finance", "FinMind (僅限台股)"),
                          horizontal=True, key="t1_source")

    with c2:
        quick = st.radio("時間區間：",
                         ["近 1 個月", "近 3 個月", "今年以來", "全部歷史", "進階自訂..."],
                         horizontal=True, key="t1_time")
        custom = None
        today = datetime.date.today()

        if quick == "近 1 個月":
            days = 30
        elif quick == "近 3 個月":
            days = 90
        elif quick == "今年以來":
            days = (today - datetime.date(today.year, 1, 1)).days
        elif quick == "全部歷史":
            days = 365 * 20
        else:
            adv = st.selectbox("進階區間：", ["近一週", "近半年", "近兩年", "📅 自訂日曆區間"],
                               key="t1_adv")
            days = {"近一週": 7, "近半年": 180, "近兩年": 730}.get(adv, 30)
            if adv == "📅 自訂日曆區間":
                rng = st.date_input("選擇起訖日期",
                                    [today - datetime.timedelta(days=30), today], key="t1_date")
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    custom = rng
                else:
                    st.info("請選擇完整的起訖日期。")
                    st.stop()

    with c3:
        currency = st.radio("顯示計價幣別：", ("預設 (新台幣)", "美元 (USD)"),
                            horizontal=True, key="t1_currency")
        st.markdown("**主圖表顯示控制：**")
        st.radio("主圖表類型", ["📈 收盤價折線圖", "📊 專業 K 線與成交量圖"],
                 horizontal=True, key="t1_chart_type")

    return ticker, source, quick, days, custom, currency


def _load(ticker, source, quick, days, custom, currency, rate):
    today = datetime.date.today()
    if custom:
        start_show, end = custom[0], custom[1]
    else:
        end, start_show = today, today - datetime.timedelta(days=days)

    # 多抓 90 天，讓 MA60 在畫面第一天就有值
    s = (start_show - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    e = end.strftime("%Y-%m-%d")

    actual = source
    if source == "Yahoo Finance":
        df = load_yahoo(ticker, s, e)
        if (df.empty or len(df) <= 1) and ticker.endswith(".TW"):
            actual = "FinMind (備援)"
            df = load_finmind(ticker.replace(".TW", ""), s, e)
    else:
        df = load_finmind(ticker.replace(".TW", ""), s, e)

    if df.empty or len(df) <= 1:
        return None, None

    df = df.copy()
    is_tw = ticker.endswith(".TW") or "FinMind" in actual

    if currency == "預設 (新台幣)":
        if not is_tw:
            for c in ["Open", "High", "Low", "Close"]:
                if c in df.columns:
                    df[c] *= rate
        symbol = "NT$"
    else:
        if is_tw:
            for c in ["Open", "High", "Low", "Close"]:
                if c in df.columns:
                    df[c] /= rate
        symbol = "US$"

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["RSI"] = calc_rsi(df["Close"], 14)

    diff, pct = df["Close"].diff(), df["Close"].pct_change() * 100
    df["Hover"] = ("<b>價格差:</b> " + diff.apply(lambda x: f"{x:+.2f}" if pd.notnull(x) else "0")
                   + "<br><b>漲跌幅:</b> " + pct.apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "0%"))

    if quick != "全部歷史":
        df = df[df.index >= pd.Timestamp(start_show)]

    return df, {"ticker": ticker, "symbol": symbol, "source": actual}


def render():
    rate, _ = get_exchange_rate()

    with st.container(border=True):
        st.subheader("🛠️ 查詢與條件設定")
        ticker, source, quick, days, custom, currency = _settings()
        go_btn = st.button("🚀 載入歷史數據並繪圖", use_container_width=True, key="t1_btn")

    if go_btn:
        with st.spinner(f"正在撈取 {ticker} 歷史數據..."):
            df, meta = _load(ticker, source, quick, days, custom, currency, rate)
        if df is None:
            st.error("❌ 無法獲取資料。請確認代碼是否正確，或到「AI 預測」分頁執行資料來源健檢。")
        else:
            st.session_state["t1_data"], st.session_state["t1_meta"] = df, meta

    data = st.session_state.get("t1_data")
    if data is None:
        return

    meta = st.session_state["t1_meta"]
    st.success(f"✅ 成功自 {meta['source']} 載入 **{meta['ticker']}**！")

    m1, m2, m3 = st.columns(3)
    change = data["Close"].iloc[-1] - data["Close"].iloc[0]
    m1.metric("期間最後結算價格", f"{meta['symbol']}{data['Close'].iloc[-1]:.2f}")
    m2.metric("選定區間總漲跌幅", f"{change:+.2f}",
              f"{change / data['Close'].iloc[0] * 100:+.2f}%")
    m3.metric("顯示歷史天數", f"{len(data)} 天")

    with st.expander("⚙️ 進階圖表與指標設定 (MA均線、RSI)"):
        e1, e2, e3 = st.columns([1, 1, 1.5])
        with e1:
            st.markdown("**均線設定**")
            ma20 = st.toggle("顯示 MA20 (月線)", True, key="t1_ma20")
            ma60 = st.toggle("顯示 MA60 (季線)", True, key="t1_ma60")
            st.toggle("顯示折線圖資料點", False, key="t1_markers")
        with e2:
            st.markdown("**副圖設定**")
            show_rsi = st.toggle("顯示 RSI 技術指標圖", False, key="t1_rsi")
            st.toggle("顯示 RSI 資料點", False, key="t1_rsi_mk")
        with e3:
            st.markdown("**🎨 RSI 顏色自訂**")
            c1, c2, c3 = st.columns(3)
            col_ob = c1.color_picker("超買區 (>70)", "#FF0000", key="t1_ob")
            col_nm = c2.color_picker("RSI 主線", "#800080", key="t1_nm")
            col_os = c3.color_picker("超賣區 (<30)", "#008000", key="t1_os")

    breaks = make_rangebreaks(data.index)
    mode = "lines+markers" if st.session_state.get("t1_markers") else "lines"
    want_k = st.session_state.get("t1_chart_type") == "📊 專業 K 線與成交量圖"
    has_k = all(c in data.columns for c in OHLCV)

    with st.container(border=True):
        if want_k and has_k:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                row_heights=[0.7, 0.3], row_titles=["價格", "成交量"])
            fig.add_trace(go.Candlestick(
                x=data.index, open=data["Open"], high=data["High"],
                low=data["Low"], close=data["Close"], name="K線",
                increasing_line_color="red", increasing_fillcolor="red",
                decreasing_line_color="green", decreasing_fillcolor="green"), row=1, col=1)
            if ma20:
                fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20",
                                         line=dict(color="orange", width=1.5)), row=1, col=1)
            if ma60:
                fig.add_trace(go.Scatter(x=data.index, y=data["MA60"], name="MA60",
                                         line=dict(color="blue", width=1.5)), row=1, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="成交量", opacity=0.7,
                                 marker_color=np.where(data["Close"] >= data["Open"], "red", "green")),
                          row=2, col=1)
            fig.update_layout(title=f"{meta['ticker']} 歷史 K 線與量能圖", height=600,
                              template="plotly_white", hovermode="x unified",
                              xaxis_rangeslider_visible=False,
                              xaxis=dict(rangebreaks=[dict(values=breaks)]))
        else:
            if want_k and not has_k:
                st.warning(f"⚠️ {meta['ticker']} 缺乏完整 OHLCV，已自動降級為折線圖。")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index, y=data["Close"], mode=mode, name="收盤價", text=data["Hover"],
                hovertemplate="<b>時間</b>: %{x}<br><b>價格</b>: %{y:.2f}<br>%{text}<extra></extra>"))
            if ma20:
                fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20",
                                         line=dict(dash="dot", color="orange")))
            if ma60:
                fig.add_trace(go.Scatter(x=data.index, y=data["MA60"], name="MA60",
                                         line=dict(dash="dot", color="green")))
            fig.update_layout(title=f"{meta['ticker']} 歷史收盤價走勢", height=450,
                              xaxis_title="日期", yaxis_title=f"價格 ({meta['symbol']})",
                              template="plotly_white", hovermode="x unified",
                              xaxis=dict(rangebreaks=[dict(values=breaks)]))
        st.plotly_chart(fig, use_container_width=True)

    if not show_rsi:
        return

    with st.container(border=True):
        fig = go.Figure()
        valid = data.dropna(subset=["RSI"])
        if not valid.empty:
            xo, yo = valid.index, valid["RSI"].to_numpy()
            xs, ys = [], []
            for i in range(len(xo) - 1):
                x1, y1, x2, y2 = xo[i], yo[i], xo[i + 1], yo[i + 1]
                xs.append(x1)
                ys.append(y1)
                for lv in (70, 30):                 # 穿越 70/30 時插一個點，讓填色不外溢
                    if (y1 - lv) * (y2 - lv) < 0:
                        f = (lv - y1) / (y2 - y1)
                        xs.append(x1 + (x2 - x1) * f)
                        ys.append(float(lv))
            xs.append(xo[-1])
            ys.append(yo[-1])
            xs, ys = np.array(xs), np.array(ys)

            fig.add_trace(go.Scatter(x=xs, y=[70] * len(xs), line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=np.where(ys >= 70, ys, 70), fill="tonexty",
                                     fillcolor=_hex_to_rgba(col_ob), line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=[30] * len(xs), line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=np.where(ys <= 30, ys, 30), fill="tonexty",
                                     fillcolor=_hex_to_rgba(col_os), line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=data.index, y=data["RSI"], name="RSI",
                mode="lines+markers" if st.session_state.get("t1_rsi_mk") else "lines",
                line=dict(color=col_nm, width=2)))

        fig.add_hline(y=70, line_dash="dash", line_color="gray",
                      annotation_text="超買區 (70)", annotation_position="top left")
        fig.add_hline(y=30, line_dash="dash", line_color="gray",
                      annotation_text="超賣區 (30)", annotation_position="bottom left")
        fig.update_layout(title=f"{meta['ticker']} RSI 相對強弱指標", height=350,
                          template="plotly_white", yaxis_range=[0, 100], hovermode="x unified",
                          xaxis=dict(rangebreaks=[dict(values=breaks)]))
        st.plotly_chart(fig, use_container_width=True)
