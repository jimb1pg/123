# -*- coding: utf-8 -*-
"""本機檔案存取：預測結果 .pkl、每日新聞 CSV。全部存在 data/ 資料夾。"""
import datetime
import json
import logging
import pickle

import pandas as pd

from config import COLLECT_TERMS, DATA_DIR, NEWS_DIR, RESULT_DIR, WATCHLIST

log = logging.getLogger(__name__)


# ---------- 預測結果 ----------

def result_path(stock_id: str, source: str, years: int):
    return RESULT_DIR / f"{stock_id}_{source}_{years}y.pkl"


def save_result(result: dict):
    path = result_path(result["target"], result["source_key"], result["years"])
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return path


def load_result(stock_id: str, source: str, years: int):
    path = result_path(stock_id, source, years)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log.warning(f"讀取預測結果失敗 {path}：{e}")
        return None


def list_results():
    return sorted(p.name for p in RESULT_DIR.glob("*.pkl"))


# ---------- 每日新聞存檔 ----------

def save_daily_news(df: pd.DataFrame, day=None) -> str:
    """把當天的新聞情緒結果存成一個 CSV。同一天重複執行會合併去重。"""
    if df is None or df.empty:
        return ""

    day = day or datetime.date.today()
    path = NEWS_DIR / f"{day}.csv"

    out = df.copy()
    out["SavedAt"] = datetime.datetime.now().isoformat(timespec="seconds")

    if path.exists():
        try:
            old = pd.read_csv(path)
            out = pd.concat([old, out], ignore_index=True)
        except Exception as e:
            log.warning(f"合併舊檔失敗 {path}：{e}")

    if "Title" in out.columns:
        key = out["Title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        subset = ["StockID"] if "StockID" in out.columns else []
        out = out.assign(_k=key).drop_duplicates(subset=subset + ["_k"]).drop(columns=["_k"])

    out.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def load_news_archive(stock_id: str = None) -> pd.DataFrame:
    """把 data/news/ 底下所有 CSV 讀成一張表，用來看歷史情緒趨勢。"""
    files = sorted(NEWS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            log.warning(f"讀取 {f} 失敗：{e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
    if stock_id and "StockID" in df.columns:
        df = df[df["StockID"].astype(str) == str(stock_id)]
    return df


def archive_summary() -> pd.DataFrame:
    """每個檔案幾則新聞，用來檢查資料累積狀況。"""
    rows = []
    for f in sorted(NEWS_DIR.glob("*.csv")):
        try:
            rows.append({"日期": f.stem, "則數": len(pd.read_csv(f))})
        except Exception:
            rows.append({"日期": f.stem, "則數": -1})
    return pd.DataFrame(rows)


# ---------- 蒐集清單（可在介面上編輯，collect_news.py 也讀這個） ----------

WATCHLIST_FILE = DATA_DIR / "watchlist.json"


def load_watchlist() -> dict:
    """讀取蒐集清單。檔案不存在時用 config.py 的預設值。"""
    if WATCHLIST_FILE.exists():
        try:
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            stocks = data.get("stocks") or {}
            terms = data.get("terms") or []
            if stocks and terms:
                return {"stocks": {str(k): str(v) for k, v in stocks.items()},
                        "terms": [str(t) for t in terms]}
        except Exception as e:
            log.warning(f"讀取 watchlist 失敗：{e}")
    return {"stocks": dict(WATCHLIST), "terms": list(COLLECT_TERMS)}


def save_watchlist(stocks: dict, terms: list) -> str:
    payload = {
        "stocks": {str(k).strip(): str(v).strip() for k, v in stocks.items() if str(k).strip()},
        "terms": [str(t).strip() for t in terms if str(t).strip()],
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    WATCHLIST_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(WATCHLIST_FILE)


# ---------- 預測紀錄（只增不刪，用來做事後驗證） ----------

PRED_LOG = DATA_DIR / "prediction_log.csv"


def append_prediction_log(result: dict) -> int:
    """
    把一次預測的每一天都記成一列。
    .pkl 會被覆蓋，但這份紀錄只會累積，所以未來可以回頭比對預測準不準。
    """
    fc = result.get("forecast_df")
    if fc is None or fc.empty:
        return 0

    predicted_at = result.get("created_at") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    run_id = f"{result['target']}_{result['source_key']}_{predicted_at.replace(' ', '_').replace(':', '')}"
    last_close = float(result["stock_df"]["Close"].iloc[-1])
    diag = result.get("diag", {})

    rows = pd.DataFrame({
        "run_id": run_id,
        "stock_id": result["target"],
        "source": result["source_key"],
        "years": result["years"],
        "predicted_at": predicted_at,
        "base_close": round(last_close, 4),
        "target_date": fc.index.strftime("%Y-%m-%d"),
        "pred_close": fc["Close"].round(4).to_numpy(),
        "pred_return": fc["Return"].round(6).to_numpy(),
        "upper": fc["Upper"].round(4).to_numpy(),
        "lower": fc["Lower"].round(4).to_numpy(),
        "direction_acc": round(float(diag.get("direction_acc", 0)), 4),
        "std_ratio": round(float(diag.get("std_ratio", 0)), 4),
    })

    header = not PRED_LOG.exists()
    rows.to_csv(PRED_LOG, mode="a", header=header, index=False, encoding="utf-8-sig")
    return len(rows)


def load_prediction_log(stock_id: str = None) -> pd.DataFrame:
    if not PRED_LOG.exists():
        return pd.DataFrame()
    try:
        # stock_id 必須當字串讀，否則 "0050" 會變成數字 50
        df = pd.read_csv(PRED_LOG, dtype={"stock_id": str, "source": str})
    except Exception as e:
        log.warning(f"讀取預測紀錄失敗：{e}")
        return pd.DataFrame()

    if df.empty:
        return df
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
    df = df.dropna(subset=["target_date"])
    if stock_id:
        df = df[df["stock_id"].astype(str) == str(stock_id)]
    return df
