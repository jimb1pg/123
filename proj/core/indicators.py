# -*- coding: utf-8 -*-
"""技術指標。全系統只用這裡的定義，避免各分頁算法不一致。"""
import numpy as np
import pandas as pd


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(np.where(avg_loss == 0, 100.0, rsi), index=close.index)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次算齊 MA / RSI / MACD / 波動率 / ATR / 量能。"""
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["RSI14"] = calc_rsi(df["Close"], 14)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Volatility20"] = df["Return"].rolling(20).std()

    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    return df.replace([np.inf, -np.inf], np.nan)


def make_rangebreaks(index) -> list:
    """找出要從 X 軸隱藏的非交易日，讓圖上不要出現週末空白。"""
    if index is None or len(index) == 0:
        return []
    idx = pd.DatetimeIndex(index)
    actual = set(idx.strftime("%Y-%m-%d"))
    full = set(pd.date_range(idx.min(), idx.max(), freq="D").strftime("%Y-%m-%d"))
    return list(full - actual)
