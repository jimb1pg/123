# -*- coding: utf-8 -*-
"""FinBERT 情緒分析。只負責「判斷情緒」，不負責抓新聞。"""
import logging

import numpy as np
import pandas as pd
import streamlit as st

from config import CACHE_TTL, NEWS_RECENCY_DECAY, SENTIMENT_MODEL_CANDIDATES

log = logging.getLogger(__name__)

# 無語意標籤的備援對應（finbert-tone 慣例：0 中立 / 1 正向 / 2 負向）
GENERIC_LABEL_MAP = {"LABEL_0": 0.0, "LABEL_1": 1.0, "LABEL_2": -1.0}


@st.cache_resource(show_spinner=False)
def get_classifier():
    """回傳 (pipeline, 模型名稱, 失敗訊息清單)。全失敗時 pipeline 為 None。"""
    from transformers import pipeline

    errors = []
    for name in SENTIMENT_MODEL_CANDIDATES:
        try:
            clf = pipeline("text-classification", model=name, tokenizer=name,
                           truncation=True, max_length=128)
            return clf, name, errors
        except Exception as e:
            msg = f"{name}：{type(e).__name__} {e}"
            log.warning(msg)
            errors.append(msg)
    return None, None, errors


def load_classifier_plain():
    """給 collect_news.py 用（不依賴 Streamlit）。"""
    from transformers import pipeline
    for name in SENTIMENT_MODEL_CANDIDATES:
        try:
            return pipeline("text-classification", model=name, tokenizer=name,
                            truncation=True, max_length=128), name
        except Exception:
            continue
    return None, None


def label_to_score(label: str) -> float:
    raw = str(label)
    if raw.upper() in GENERIC_LABEL_MAP:
        return GENERIC_LABEL_MAP[raw.upper()]
    low = raw.lower()
    if any(w in low for w in ["positive", "pos", "bullish", "利多", "正向", "正面"]):
        return 1.0
    if any(w in low for w in ["negative", "neg", "bearish", "利空", "負向", "負面"]):
        return -1.0
    return 0.0


def _apply(clf, news_df: pd.DataFrame) -> pd.DataFrame:
    texts = news_df["Title"].astype(str).str.slice(0, 500).tolist()
    try:
        preds = clf(texts, batch_size=32)          # 批次推論，比逐則快很多
    except Exception as e:
        log.warning(f"批次推論失敗，改逐則：{e}")
        preds = []
        for t in texts:
            try:
                preds.append(clf(t)[0])
            except Exception:
                preds.append({"label": "neutral", "score": 0.0})

    out = news_df.copy().reset_index(drop=True)
    out["Label"] = [str(p.get("label", "neutral")) for p in preds]
    out["Confidence"] = [float(p.get("score", 0.0)) for p in preds]
    out["Sentiment"] = [label_to_score(p.get("label", "neutral")) * float(p.get("score", 0.0))
                        for p in preds]

    latest = out["Date"].max()
    days_old = (latest - out["Date"]).dt.days.clip(lower=0)
    out["Raw_Weight"] = np.exp(-NEWS_RECENCY_DECAY * days_old) * out["Confidence"]

    total = out["Raw_Weight"].sum()
    out["Weight"] = out["Raw_Weight"] / total if total > 0 else 1.0 / max(len(out), 1)
    out["Weight_Pct"] = out["Weight"] * 100
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return pd.DataFrame()
    clf, _, _ = get_classifier()
    if clf is None:
        return pd.DataFrame()
    return _apply(clf, news_df)


def score_news_plain(clf, news_df: pd.DataFrame) -> pd.DataFrame:
    if clf is None or news_df is None or news_df.empty:
        return pd.DataFrame()
    return _apply(clf, news_df)


# ------------------------------------------------------------
# 聚合
# ------------------------------------------------------------

def daily_sentiment(scored: pd.DataFrame, smooth_window: int = 20) -> pd.DataFrame:
    """把逐則新聞聚合成每日情緒（信心度加權平均）。"""
    if scored is None or scored.empty:
        return pd.DataFrame(columns=["sentiment_score_raw", "sentiment_score"])

    tmp = scored[["Date", "Sentiment", "Confidence"]].copy()
    tmp["Num"] = tmp["Sentiment"] * tmp["Confidence"]

    g = tmp.groupby("Date")[["Num", "Confidence"]].sum()
    daily = (g["Num"] / g["Confidence"].replace(0, np.nan)).fillna(0.0)
    daily.name = "sentiment_score_raw"

    out = daily.to_frame()
    out["sentiment_score"] = out["sentiment_score_raw"].rolling(
        smooth_window, min_periods=1).mean()
    return out


def keyword_weight_table(scored: pd.DataFrame) -> pd.DataFrame:
    """各搜尋關鍵字的權重占比。一則新聞命中多個關鍵字時，權重平均分配。"""
    if scored is None or scored.empty or "Matched_Queries" not in scored.columns:
        return pd.DataFrame()

    rows = []
    for _, item in scored.iterrows():
        matched = [x.strip() for x in str(item["Matched_Queries"]).split("|") if x.strip()]
        if not matched:
            continue
        share = item["Weight"] / len(matched)
        for q in matched:
            rows.append({"搜尋關鍵字": q, "權重": share, "命中": 1})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).groupby("搜尋關鍵字").agg(
        新聞則數=("命中", "sum"), 權重占比=("權重", "sum"))
    df["權重占比"] *= 100
    return df.sort_values("權重占比", ascending=False).reset_index()


def build_features(stock_index: pd.DatetimeIndex, scored: pd.DataFrame) -> pd.DataFrame:
    """把每日情緒對齊到股票交易日；沒有新聞的日子補 0。"""
    result = pd.DataFrame(index=stock_index)
    result["sentiment_score_raw"] = 0.0
    result["sentiment_score"] = 0.0

    if scored is None or scored.empty:
        return result

    lo, hi = stock_index.min().normalize(), stock_index.max().normalize()
    sub = scored[(scored["Date"] >= lo) & (scored["Date"] <= hi)]
    if sub.empty:
        return result

    daily = daily_sentiment(sub)
    result["sentiment_score_raw"] = daily["sentiment_score_raw"].reindex(result.index).fillna(0.0)
    result["sentiment_score"] = result["sentiment_score_raw"].rolling(20, min_periods=1).mean()
    return result


def selftest():
    """三句測試：正向 / 負向 / 中立。用來確認模型和標籤對應是否正常。"""
    clf, name, errors = get_classifier()
    if clf is None:
        return None, None, errors

    samples = [
        "台積電第三季營收創新高，法人看好後市",
        "公司下修全年財測，股價重挫",
        "公司今日召開股東常會",
    ]
    preds = clf(samples)
    table = pd.DataFrame({
        "測試句": samples,
        "模型標籤": [p["label"] for p in preds],
        "信心度": [round(p["score"], 4) for p in preds],
        "轉換分數": [label_to_score(p["label"]) for p in preds],
    })
    return name, table, errors
