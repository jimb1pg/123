# -*- coding: utf-8 -*-
"""
每日新聞蒐集腳本（本機執行，不需要 Streamlit）

基本用法：
    python collect_news.py

臨時指定標的與關鍵字（不改設定檔）：
    python collect_news.py --stocks 2330,2317 --terms 營收,訂單
    python collect_news.py --per-query 15

追蹤清單平常存在 data/watchlist.json，
可以直接編輯該檔案，或在程式的「新聞情緒分析引擎」分頁上修改。

結果存到 data/news/YYYY-MM-DD.csv，同一天重複執行會自動合併去重。
"""
import argparse
import datetime
import sys

import pandas as pd

from config import NEWS_DIR, NEWS_RESULTS_PER_QUERY
from core.news import fetch_news_plain
from core.sentiment import load_classifier_plain, score_news_plain
from core.storage import load_watchlist


def parse_args():
    ap = argparse.ArgumentParser(description="每日新聞情緒蒐集")
    ap.add_argument("--stocks", help="股票代碼，逗號分隔。例：2330,0050")
    ap.add_argument("--terms", help="搜尋關鍵字，逗號分隔。例：營收,訂單")
    ap.add_argument("--per-query", type=int, default=NEWS_RESULTS_PER_QUERY,
                    help=f"每組關鍵字抓幾則（預設 {NEWS_RESULTS_PER_QUERY}）")
    return ap.parse_args()


def resolve_targets(args):
    """指令參數優先，其次 data/watchlist.json，最後 config.py 預設。"""
    wl = load_watchlist()
    stocks, terms = wl["stocks"], wl["terms"]

    if args.stocks:
        codes = [c.strip().upper() for c in args.stocks.split(",") if c.strip()]
        stocks = {c: stocks.get(c, c) for c in codes}       # 沒有中文名就用代碼

    if args.terms:
        terms = [t.strip() for t in args.terms.split(",") if t.strip()]

    return stocks, terms


def main():
    args = parse_args()
    stocks, terms = resolve_targets(args)

    print("=" * 60)
    print(f"每日新聞蒐集　{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"追蹤標的：{'、'.join(f'{k} {v}' for k, v in stocks.items())}")
    print(f"搜尋關鍵字：{'、'.join(terms)}")
    print(f"每組抓取則數：{args.per_query}")
    print("=" * 60)

    if not stocks or not terms:
        print("❌ 追蹤清單或關鍵字是空的，請檢查 data/watchlist.json。")
        return 1

    print("\n載入 FinBERT 模型（首次執行需下載，請耐心等待）...")
    clf, model_name = load_classifier_plain()
    if clf is None:
        print("❌ 情緒模型載入失敗，中止。")
        return 1
    print(f"✅ 使用模型：{model_name}\n")

    all_rows = []
    for code, name in stocks.items():
        queries = tuple(f"{t} {name}" for t in terms)
        print(f"[{code} {name}] 搜尋 {len(queries)} 組關鍵字...")

        news_df, failures = fetch_news_plain(queries, args.per_query)
        for f in failures:
            print(f"   ⚠️ 搜尋失敗：{f}")

        if news_df.empty:
            print("   ⚠️ 沒有抓到任何新聞\n")
            continue

        scored = score_news_plain(clf, news_df)
        if scored.empty:
            print("   ⚠️ 情緒分析失敗\n")
            continue

        scored["StockID"] = code
        scored["StockName"] = name
        all_rows.append(scored)

        pos = int((scored["Sentiment"] > 0).sum())
        neg = int((scored["Sentiment"] < 0).sum())
        neu = len(scored) - pos - neg
        print(f"   ✅ 共 {len(scored)} 則　正向 {pos} / 中立 {neu} / 負向 {neg}\n")

    if not all_rows:
        print("沒有任何資料可存，結束。")
        return 1

    df = pd.concat(all_rows, ignore_index=True)
    path = NEWS_DIR / f"{datetime.date.today()}.csv"

    if path.exists():
        try:
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        except Exception as e:
            print(f"⚠️ 合併舊檔失敗：{e}")

    key = df["Title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.assign(_k=key).drop_duplicates(subset=["StockID", "_k"]).drop(columns=["_k"])
    df["SavedAt"] = datetime.datetime.now().isoformat(timespec="seconds")
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"✅ 已寫入 {path}")
    print(f"　 當日累計 {len(df)} 則（已去重）")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
