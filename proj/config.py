# -*- coding: utf-8 -*-
"""
所有「設定值」都集中在這裡。
想調參數（預測天數、關鍵字、模型名稱…）只要改這個檔案，不用去翻程式碼。
"""
import logging
from pathlib import Path

APP_TITLE = "ETF 損益分析與預測系統"

# ---------- 資料夾（程式會自動建立） ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
NEWS_DIR = DATA_DIR / "news"          # 每日新聞 CSV 存這裡
RESULT_DIR = DATA_DIR / "results"     # 預測結果 .pkl 存這裡
LOG_FILE = DATA_DIR / "app.log"       # 出錯時來這裡看原因

for _d in (DATA_DIR, NEWS_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    encoding="utf-8",
)

# ---------- 快取 ----------
CACHE_TTL = 1800        # 秒。同樣的查詢 30 分鐘內直接用上次結果

# ---------- 模型 ----------
RANDOM_SEED = 42
LOOKBACK_DAYS = 60      # 模型看過去幾個交易日
FORECAST_DAYS = 22      # 預測未來幾個交易日（約一個月）
TRAIN_EPOCHS = 80
BATCH_SIZE = 32
KMEANS_CLUSTERS = 3
TRAIN_RATIO = 0.8

# ---------- 新聞 ----------
NEWS_RESULTS_PER_QUERY = 10
NEWS_RECENCY_DECAY = 0.05
NEWS_QUERY_TERMS = [
    "AI", "營收", "財報", "獲利", "法說會", "半導體", "產能", "訂單",
    "展望", "投資", "產品", "需求", "景氣", "政策", "利率",
]

# 依序嘗試，第一個載得起來的就用
SENTIMENT_MODEL_CANDIDATES = [
    "yiyanghkust/finbert-tone-chinese",
    "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
    "uer/roberta-base-finetuned-jd-binary-chinese",
]

# ---------- 欄位定義 ----------
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

STATE_NAMES = ["上升型", "下降型", "震盪型"]
STATE_COLUMNS = [f"State_{s}" for s in STATE_NAMES]

BASE_FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume", "Return",
    "MA5", "MA20", "MA60", "RSI14", "MACD", "MACD_Signal",
    "Volatility20", "ATR14", "Volume_Change", "sentiment_score",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + STATE_COLUMNS   # KMeans 狀態也進模型
TARGET_COLUMNS = ["Close_Return"]

# ---------- 常見台股名稱（FinMind 查不到時的備援） ----------
FALLBACK_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電",
    "2382": "廣達", "2303": "聯電", "2881": "富邦金", "2882": "國泰金",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息",
}

# ---------- 每日蒐集腳本要追蹤的標的 ----------
WATCHLIST = {"2330": "台積電", "0050": "元大台灣50", "2454": "聯發科"}
COLLECT_TERMS = ["營收", "財報", "法說會", "訂單", "展望"]
