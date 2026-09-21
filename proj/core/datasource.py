# -*- coding: utf-8 -*-
"""
股價資料來源。所有抓股價的動作都走這裡，不要在別的地方直接呼叫 API。

重點：
  * FinMind 的 taiwan_stock_daily_adj（還原股價）需要「付費贊助帳號」，
    免費帳號呼叫會失敗。這裡會自動退回免費的 taiwan_stock_daily。
  * 沒有 token 也能跑，只是限額較低（300 次/小時 vs 600 次/小時）。
"""
import datetime
import logging
import os

import pandas as pd
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from FinMind.data import DataLoader

from config import CACHE_TTL, FALLBACK_NAMES, OHLCV

log = logging.getLogger(__name__)

fm_api = DataLoader()

FINMIND_RENAME = {
    "date": "Date", "open": "Open", "max": "High", "min": "Low",
    "close": "Close", "Trading_Volume": "Volume", "volume": "Volume",
}


# ------------------------------------------------------------
# FinMind 登入（可選）
# ------------------------------------------------------------

def _read_token() -> str:
    """token 找三個地方：secrets.toml → 環境變數 → data/finmind_token.txt"""
    try:
        t = st.secrets.get("FINMIND_TOKEN", "")
        if t:
            return str(t).strip()
    except Exception:
        pass

    t = os.environ.get("FINMIND_TOKEN", "")
    if t:
        return t.strip()

    from config import DATA_DIR
    f = DATA_DIR / "finmind_token.txt"
    if f.exists():
        try:
            return f.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


FINMIND_TOKEN = _read_token()
FINMIND_LOGGED_IN = False

if FINMIND_TOKEN:
    try:
        fm_api.login_by_token(api_token=FINMIND_TOKEN)
        FINMIND_LOGGED_IN = True
    except Exception as e:
        log.warning(f"FinMind 登入失敗：{e}")


# ------------------------------------------------------------
# 抓資料
# ------------------------------------------------------------

def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """把 FinMind 回傳的表整理成統一的 OHLCV 格式。"""
    df = df.rename(columns=FINMIND_RENAME)
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if "Volume" not in df.columns:
        df["Volume"] = 0
    for c in OHLCV:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[[c for c in OHLCV if c in df.columns]].dropna(subset=["Close"])


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_finmind(stock_id: str, start: str, end: str) -> pd.DataFrame:
    """已登入時先試還原股價，失敗或未登入則用免費的原始股價。"""
    tries = []
    if FINMIND_LOGGED_IN:
        tries.append(("adj", fm_api.taiwan_stock_daily_adj))
    tries.append(("raw", fm_api.taiwan_stock_daily))

    for kind, fn in tries:
        try:
            df = fn(stock_id=stock_id, start_date=start, end_date=end)
        except Exception as e:
            log.warning(f"FinMind[{kind}] {stock_id}：{e}")
            continue
        if df is not None and not df.empty:
            return _tidy(df)

    log.warning(f"FinMind 全部來源皆無資料：{stock_id} {start}~{end}")
    return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Yahoo Finance，auto_adjust=True 已還原除權息。"""
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    except Exception as e:
        log.warning(f"Yahoo {ticker}：{e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if "Volume" not in df.columns:
        df["Volume"] = 0

    return df[[c for c in OHLCV if c in df.columns]]


def date_range(years: int = 5):
    today = datetime.date.today()
    return (today - relativedelta(years=years)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_stock_data(stock_id: str, source: str = "yfinance", years: int = 5) -> pd.DataFrame:
    """
    AI 模型專用：抓指定年數的台股 OHLCV。
    任一來源失敗時自動嘗試另一個，盡量不要讓使用者看到空白。
    """
    start, end = date_range(years)
    stock_id = stock_id.strip().upper()

    order = ["yfinance", "finmind"] if source == "yfinance" else ["finmind", "yfinance"]

    for src in order:
        df = load_yahoo(f"{stock_id}.TW", start, end) if src == "yfinance" \
            else load_finmind(stock_id, start, end)
        if not df.empty and len(df) > 100:
            for c in OHLCV:
                if c not in df.columns:
                    df[c] = 0.0
            return df[OHLCV].dropna(subset=["Close"])

    return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_name(stock_id: str) -> str:
    """取中文名稱；查不到就用內建表，再不行就回傳代碼。"""
    try:
        info = fm_api.taiwan_stock_info()
        if info is not None and not info.empty:
            id_col = next((c for c in ["stock_id", "StockID"] if c in info.columns), None)
            nm_col = next((c for c in ["stock_name", "StockName", "name"] if c in info.columns), None)
            if id_col and nm_col:
                hit = info[info[id_col].astype(str).str.upper() == stock_id.upper()]
                if not hit.empty:
                    return str(hit.iloc[0][nm_col]).strip()
    except Exception as e:
        log.warning(f"取股票名稱失敗 {stock_id}：{e}")
    return FALLBACK_NAMES.get(stock_id.upper(), stock_id.upper())


# ------------------------------------------------------------
# 健檢
# ------------------------------------------------------------

def selftest(stock_id: str = "2330") -> pd.DataFrame:
    """逐一測試各資料來源，把真正的錯誤訊息顯示出來。"""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30)
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    rows = []
    for name, fn in [
        ("FinMind taiwan_stock_daily（免費）", fm_api.taiwan_stock_daily),
        ("FinMind taiwan_stock_daily_adj（需贊助帳號）", fm_api.taiwan_stock_daily_adj),
    ]:
        try:
            df = fn(stock_id=stock_id, start_date=s, end_date=e)
            ok = df is not None and not df.empty
            rows.append({"資料來源": name, "狀態": "✅ 可用" if ok else "⚠️ 回傳空資料",
                         "筆數": 0 if df is None else len(df), "訊息": ""})
        except Exception as ex:
            rows.append({"資料來源": name, "狀態": "❌ 失敗", "筆數": 0,
                         "訊息": str(ex)[:220]})

    try:
        y = yf.Ticker(f"{stock_id}.TW").history(start=s, end=e, auto_adjust=True)
        rows.append({"資料來源": "Yahoo Finance", "狀態": "✅ 可用" if not y.empty else "⚠️ 回傳空資料",
                     "筆數": len(y), "訊息": ""})
    except Exception as ex:
        rows.append({"資料來源": "Yahoo Finance", "狀態": "❌ 失敗", "筆數": 0,
                     "訊息": str(ex)[:220]})

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_exchange_rate():
    """美元兌台幣；取不到回傳預設 32.0。"""
    end = datetime.date.today()
    df = load_yahoo("USDTWD=X", (end - datetime.timedelta(days=10)).strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"))
    if not df.empty:
        v = df["Close"].iloc[-1]
        if pd.notna(v) and v > 0:
            return float(v), False
    return 32.0, True
