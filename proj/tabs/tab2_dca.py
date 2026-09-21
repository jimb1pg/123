# -*- coding: utf-8 -*-
"""分頁二：定期定額回測（含年化報酬率 XIRR 與最大回撤）"""
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.datasource import load_finmind, load_yahoo, FINMIND_LOGGED_IN
from core.finance import max_drawdown, xirr
from core.indicators import make_rangebreaks


def render():
    with st.container(border=True):
        st.subheader("💵 定期定額投資試算")
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
        target = c1.text_input("輸入回測標的代碼：", "0050.TW", key="t2_target")
        amount = c2.number_input("每月固定投入金額 (NT$)", min_value=1000,
                                 value=10000, step=1000, key="t2_amt")
        start = c3.date_input("回測起始日期", datetime.date(2020, 1, 1), key="t2_date")
        source = c4.radio("優先資料來源：", ("Yahoo Finance", "FinMind (僅限台股)"),
                          horizontal=True, key="t2_source")

        if FINMIND_LOGGED_IN:
            st.caption("ℹ️ Yahoo 與 FinMind 皆使用還原股價（已計入除權息），長期報酬率較貼近實際。")
        else:
            st.caption("ℹ️ Yahoo Finance 使用還原股價（已計入除權息）。"
                       "FinMind 未登入時只能取得未還原原價，長期回測建議用 Yahoo Finance。")

        go_btn = st.button("🚀 執行定期定額回測", use_container_width=True, key="t2_btn")

    if not go_btn:
        return

    with st.spinner(f"正在抓取 {target} 歷史數據..."):
        s = start.strftime("%Y-%m-%d")
        e = datetime.date.today().strftime("%Y-%m-%d")

        actual = source
        if source == "Yahoo Finance":
            bt = load_yahoo(target, s, e)
            if (bt.empty or len(bt) <= 1) and target.endswith(".TW"):
                actual = "FinMind (備援)"
                bt = load_finmind(target.replace(".TW", ""), s, e)
        else:
            bt = load_finmind(target.replace(".TW", ""), s, e)

    if bt.empty or len(bt) <= 1:
        st.error("❌ 獲取資料失敗，請確認該標的在指定日期已上市。")
        return

    periods = bt.index.to_period("M")
    monthly = bt.groupby(periods).first()
    monthly_dates = bt.index.to_series().groupby(periods).first()

    months = len(monthly)
    total_cost = months * amount
    shares_each = amount / monthly["Close"]
    total_shares = shares_each.sum()
    final_value = total_shares * bt["Close"].iloc[-1]
    roi = (final_value - total_cost) / total_cost * 100

    # 年化報酬率：每月投入記為負、期末市值記為正
    flows = [-float(amount)] * months + [float(final_value)]
    dates = list(monthly_dates.values) + [bt.index[-1]]
    annual = xirr(flows, dates) * 100

    # 資產走勢
    curve = pd.DataFrame(index=bt.index)
    curve["資產價值"] = bt["Close"].to_numpy() * shares_each.cumsum().reindex(periods).to_numpy()
    curve["累積投入成本"] = pd.Series(np.arange(1, months + 1) * float(amount),
                                index=monthly.index).reindex(periods).to_numpy()
    mdd = max_drawdown(curve["資產價值"])

    st.success(f"✅ 回測完成！(來源: {actual}) | "
               f"{monthly.index[0]} ~ {monthly.index[-1]}（共 {months} 個月）")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("總投入成本", f"NT$ {total_cost:,.0f}")
    r2.metric("期末總價值", f"NT$ {final_value:,.0f}", f"{final_value - total_cost:+,.0f}")
    r3.metric("總報酬率 (ROI)", f"{roi:.2f}%",
              help="單純的(期末-成本)/成本，沒有考慮每筆錢投入的時間長短")
    r4.metric("年化報酬率 (XIRR)", f"{annual:.2f}%" if np.isfinite(annual) else "無法計算",
              help="定期定額的正確報酬率算法，已考慮每筆資金投入的時間長短")

    st.metric("期間最大回撤 (MDD)", f"{mdd:.2f}%",
              help="從歷史高點跌下來最慘的一次，用來衡量這筆投資的風險")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve["資產價值"], name="資產價值",
                             line=dict(color="#d62728", width=2)))
    fig.add_trace(go.Scatter(x=curve.index, y=curve["累積投入成本"], name="累積投入成本",
                             line=dict(color="gray", dash="dot")))
    fig.update_layout(title=f"{target} 定期定額資產累積走勢", height=430,
                      template="plotly_white", hovermode="x unified", yaxis_title="金額 (NT$)",
                      xaxis=dict(rangebreaks=[dict(values=make_rangebreaks(bt.index))]))
    st.plotly_chart(fig, use_container_width=True)
