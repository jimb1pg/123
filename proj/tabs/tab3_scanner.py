# -*- coding: utf-8 -*-
"""分頁三：市場趨勢即時掃描"""
import datetime

import pandas as pd
import streamlit as st

from core.datasource import load_finmind, load_yahoo
from core.indicators import calc_rsi

ACTIVE_ETF = [f"00{970 + i}A.TW" for i in range(28)]

POOLS = {
    "🇹🇼 0050 前十大權值股": ["2330.TW", "2454.TW", "2308.TW", "2317.TW", "3711.TW",
                        "2383.TW", "2303.TW", "3037.TW", "2345.TW", "2891.TW"],
    "🇹🇼 熱門高股息 ETF": ["0056.TW", "00878.TW", "00713.TW", "00919.TW",
                      "00929.TW", "00939.TW", "00940.TW"],
    "🇹🇼 半導體與科技主題 ETF": ["00891.TW", "00892.TW", "00881.TW", "00904.TW",
                         "00927.TW", "0052.TW"],
    "🇹🇼 2026 全市場主動式 ETF (28檔)": ACTIVE_ETF,
    "🇺🇸 美股科技巨頭與半導體": ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA",
                      "AMD", "AMZN", "META", "AVGO"],
    "🌍 全球宏觀指標 (金/油/大盤)": ["GC=F", "CL=F", "^TWII", "^GSPC", "^TNX"],
    "✏️ 自訂掃描清單": [],
}

NAMES = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "2308.TW": "台達電", "2317.TW": "鴻海",
    "3711.TW": "日月光投控", "2383.TW": "台光電", "2303.TW": "聯電", "3037.TW": "欣興",
    "2345.TW": "智邦", "2891.TW": "中信金",
    "0056.TW": "元大高股息", "00878.TW": "國泰永續高股息", "00713.TW": "元大台灣高息低波",
    "00919.TW": "群益台灣精選高息", "00929.TW": "復華台灣科技優息",
    "00939.TW": "統一台灣高息動能", "00940.TW": "元大台灣價值高息",
    "00891.TW": "中信關鍵半導體", "00892.TW": "富邦台灣半導體", "00881.TW": "國泰台灣5G+",
    "00904.TW": "新光臺灣半導體30", "00927.TW": "群益半導體收益", "0052.TW": "富邦科技",
    "AAPL": "蘋果", "MSFT": "微軟", "NVDA": "輝達", "GOOGL": "Google", "TSLA": "特斯拉",
    "AMD": "超微", "AMZN": "亞馬遜", "META": "Meta", "AVGO": "博通",
    "GC=F": "黃金期貨", "CL=F": "原油期貨 (WTI)", "^TWII": "台灣加權指數",
    "^GSPC": "標普500指數", "^TNX": "美國10年期公債殖利率",
}

HOLDINGS = {
    "0056.TW": "聯發科, 鴻海, 廣達", "00878.TW": "華碩, 大聯大, 聯發科",
    "00713.TW": "統一, 台灣大, 遠傳", "00919.TW": "長榮, 聯發科, 瑞昱",
    "00929.TW": "聯發科, 瑞昱, 聯詠", "00939.TW": "聯發科, 緯創, 大聯大",
    "00940.TW": "長榮, 聯電, 中美晶", "00891.TW": "聯發科, 台積電, 日月光",
    "00892.TW": "台積電, 聯發科, 聯電", "00881.TW": "台積電, 鴻海, 聯發科",
    "00904.TW": "台積電, 聯發科, 聯詠", "00927.TW": "聯發科, 台積電, 瑞昱",
    "0052.TW": "台積電, 聯發科, 鴻海",
}

for _t in ACTIVE_ETF:
    NAMES[_t] = f"主動型 {_t[:6]}"
    HOLDINGS[_t] = "機密 (經理人動態選股)"


def _scan_one(ticker, use_fm, start, end, year):
    df = load_finmind(ticker.replace(".TW", "").replace(".TWO", ""), start, end) \
        if use_fm else load_yahoo(ticker, start, end)

    if df.empty or len(df) <= 15:
        return None

    price = float(df["Close"].iloc[-1])
    ytd = df[df.index.year == year]
    ytd_ret = (price - ytd["Close"].iloc[0]) / ytd["Close"].iloc[0] * 100 if not ytd.empty else 0.0
    rsi = float(calc_rsi(df["Close"], 14).iloc[-1])

    is_tw = ticker.endswith(".TW") or ticker.endswith(".TWO")
    flag = "🇹🇼 " if is_tw else ("🌍 " if ticker in ["GC=F", "CL=F", "^GSPC", "^TNX"] else "🇺🇸 ")

    return {
        "代號": ticker, "標的名稱": flag + NAMES.get(ticker, ticker),
        "前三大持股 (透視)": HOLDINGS.get(ticker, "無 (個股/指數)"),
        "最新收盤價": round(price, 2), "今年來漲幅(%)": round(ytd_ret, 2),
        "當前 RSI": round(rsi, 1), "來源": "FinMind" if use_fm else "Yahoo",
    }


def render():
    with st.container(border=True):
        st.subheader("📺 市場趨勢即時掃描")
        c1, c2, c3 = st.columns([1.5, 1, 1.5])

        with c1:
            pool = st.selectbox("選擇實時運算模組", list(POOLS.keys()), key="t3_pool")
            if pool == "✏️ 自訂掃描清單":
                raw = st.text_input("輸入代碼 (逗號分隔，台股建議加 .TW)",
                                    "2330.TW, GC=F, AAPL, NVDA", key="t3_custom")
                tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            else:
                tickers = POOLS[pool]

        with c2:
            route = st.radio("底層 API 路由策略",
                             ["自動 (台股FinMind/其他Yahoo)", "強制 Yahoo", "強制 FinMind"],
                             key="t3_route")
            # 選單只放字串，避免混用型別造成錯誤
            total = len(tickers)
            count_map = {f"顯示全部 ({total} 檔)": max(total, 1)}
            for n in (3, 5, 10, 20):
                if n < total:
                    count_map[f"前 {n} 檔"] = n
            pick = st.selectbox("顯示結果數量", list(count_map.keys()), key="t3_cnt")

        with c3:
            strategy = st.selectbox("即時排序策略",
                                    ["依 今年來漲幅(%) 由高到低",
                                     "依 當前 RSI 由高到低 (動能強)",
                                     "依 當前 RSI 由低到高 (超跌區)"], key="t3_strat")

        go_btn = st.button("🚀 啟動全網大掃描", use_container_width=True, key="t3_btn")

    if not go_btn:
        return

    if not tickers:
        st.warning("請輸入至少一檔股票或指標代碼。")
        return

    bar = st.progress(0, text="準備連線抓取資料...")
    year = datetime.date.today().year
    end = datetime.date.today().strftime("%Y-%m-%d")
    start = (datetime.date(year, 1, 1) - datetime.timedelta(days=45)).strftime("%Y-%m-%d")

    results = []
    for i, tk in enumerate(tickers):
        bar.progress((i + 1) / len(tickers), text=f"正在分析 {NAMES.get(tk, tk)}...")
        is_tw = tk.endswith(".TW") or tk.endswith(".TWO") or (tk.isdigit() and len(tk) >= 4)
        use_fm = is_tw if route.startswith("自動") else (route == "強制 FinMind")
        row = _scan_one(tk, use_fm, start, end, year)
        if row:
            results.append(row)
    bar.empty()

    if not results:
        st.error("❌ 獲取資料失敗。可能是 API 限流或代碼無效，請到「AI 預測」分頁執行資料來源健檢。")
        return

    df = pd.DataFrame(results)
    if "今年來漲幅" in strategy:
        df = df.sort_values("今年來漲幅(%)", ascending=False)
    elif "由高到低" in strategy:
        df = df.sort_values("當前 RSI", ascending=False)
    else:
        df = df.sort_values("當前 RSI", ascending=True)

    df = df.head(count_map[pick]).reset_index(drop=True)
    df.index += 1

    st.success(f"✅ 掃描完成！共成功抓取 {len(results)} 筆市場數據。")
    st.dataframe(df, use_container_width=True, column_config={
        "最新收盤價": st.column_config.NumberColumn("收盤報價", format="%.2f"),
        "今年來漲幅(%)": st.column_config.NumberColumn("今年漲幅(%)", format="%.2f %%"),
        "當前 RSI": st.column_config.ProgressColumn("短期動能 (RSI)", format="%.1f",
                                                  min_value=0, max_value=100),
    })
