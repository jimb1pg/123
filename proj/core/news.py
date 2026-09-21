# -*- coding: utf-8 -*-
"""新聞抓取。只負責「拿到新聞」，不做情緒分析。"""
import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from config import CACHE_TTL, NEWS_QUERY_TERMS, NEWS_RESULTS_PER_QUERY

log = logging.getLogger(__name__)

EMPTY = pd.DataFrame(columns=["Title", "Link", "Date", "Source",
                              "Matched_Queries", "Keyword_Count"])


def build_queries(stock_name: str, terms=None) -> tuple:
    """組出「關鍵字 + 股票名稱」的搜尋詞，例如「營收 台積電」。"""
    terms = terms or NEWS_QUERY_TERMS
    if not stock_name:
        return ()
    return tuple(f"{t} {stock_name}" for t in terms)


def search_google_news(query: str, max_results: int = NEWS_RESULTS_PER_QUERY) -> list:
    """Google News RSS，不需要 API Key。"""
    url = ("https://news.google.com/rss/search?q=" + quote_plus(query)
           + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=12) as resp:
        root = ET.fromstring(resp.read())

    items = []
    for it in root.findall("./channel/item")[:max_results]:
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        items.append({
            "Query": query,
            "Title": title,
            "Link": (it.findtext("link") or "").strip(),
            "PubDate": (it.findtext("pubDate") or "").strip(),
            "Source": (it.findtext("source") or "").strip(),
        })
    return items


def _dedup(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["PubDate"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).copy()
    if df.empty:
        return EMPTY.copy()

    df["Date"] = df["Date"].dt.tz_convert(None).dt.normalize()
    df["Key"] = df["Title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    agg = df.groupby("Key", sort=False).agg(
        Title=("Title", "first"),
        Link=("Link", "first"),
        Date=("Date", "first"),
        Source=("Source", "first"),
        Matched_Queries=("Query", lambda s: " | ".join(dict.fromkeys(s))),
        Keyword_Count=("Query", "nunique"),
    )
    return agg.reset_index(drop=True).sort_values("Date", ascending=False)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_news(queries: tuple, max_per_query: int = NEWS_RESULTS_PER_QUERY):
    """給 Streamlit 用（有快取）。回傳 (新聞表, 失敗清單)。"""
    rows, failures = [], []
    for q in queries:
        try:
            rows.extend(search_google_news(q, max_per_query))
        except Exception as e:
            log.warning(f"新聞搜尋失敗 {q}：{e}")
            failures.append(f"{q}：{e}")

    if not rows:
        return EMPTY.copy(), failures
    return _dedup(rows), failures


def fetch_news_plain(queries, max_per_query: int = NEWS_RESULTS_PER_QUERY):
    """給 collect_news.py 用（不依賴 Streamlit）。"""
    rows, failures = [], []
    for q in queries:
        try:
            rows.extend(search_google_news(q, max_per_query))
        except Exception as e:
            failures.append(f"{q}：{e}")
    if not rows:
        return EMPTY.copy(), failures
    return _dedup(rows), failures
