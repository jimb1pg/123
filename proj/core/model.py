# -*- coding: utf-8 -*-
"""
KMeans 市場狀態 + TCN 預測模型。

TensorFlow 是「用到才 import」，所以沒裝 TF 的環境
（例如只想看已存結果）仍然可以正常啟動程式。
"""
import importlib.util
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import (BATCH_SIZE, FEATURE_COLUMNS, FORECAST_DAYS, KMEANS_CLUSTERS,
                    LOOKBACK_DAYS, RANDOM_SEED, STATE_COLUMNS, STATE_NAMES,
                    TARGET_COLUMNS, TRAIN_EPOCHS, TRAIN_RATIO)
from core.indicators import add_technical_indicators

log = logging.getLogger(__name__)

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


# ============================================================
# 特徵組裝
# ============================================================

def prepare_features(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    df = add_technical_indicators(stock_df)
    df = df.join(sentiment_df[["sentiment_score_raw", "sentiment_score"]], how="left")
    df["sentiment_score_raw"] = df["sentiment_score_raw"].fillna(0.0)
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    return df.replace([np.inf, -np.inf], np.nan)


# ============================================================
# KMeans 市場狀態
# ============================================================

def _kmeans_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["Trend20"] = df["Close"].pct_change(20)
    x["Trend60"] = df["Close"].pct_change(60)
    x["Volatility20"] = df["Return"].rolling(20).std()
    x["RSI14"] = df["RSI14"]
    x["MACD"] = df["MACD"] / df["Close"]
    x["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
    return x.replace([np.inf, -np.inf], np.nan).dropna()


def market_state(df: pd.DataFrame, n_clusters: int = KMEANS_CLUSTERS):
    feats = _kmeans_features(df)
    if len(feats) < 100:
        raise ValueError("KMeans 資料不足，建議至少 1 年以上資料。")

    x = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    labels = km.fit_predict(x)

    feats = feats.copy()
    feats["Cluster"] = labels

    summary = feats.groupby("Cluster").agg(
        Trend20=("Trend20", "mean"), Trend60=("Trend60", "mean"),
        Volatility20=("Volatility20", "mean"))

    score = summary["Trend20"] + summary["Trend60"]
    up_id, down_id = score.idxmax(), score.idxmin()
    mapping = {c: ("上升型" if c == up_id else "下降型" if c == down_id else "震盪型")
               for c in summary.index}
    summary["狀態"] = [mapping[c] for c in summary.index]

    series = pd.Series(labels, index=feats.index).map(mapping)
    latest = int(labels[-1])
    return series, latest, mapping[latest], summary


def attach_state(df: pd.DataFrame):
    """把 KMeans 狀態接成 one-hot 欄位。失敗時補 0，模型仍可運作。"""
    for c in STATE_COLUMNS:
        df[c] = 0.0
    try:
        series, latest_id, latest_name, summary = market_state(df)
        aligned = series.reindex(df.index)
        for name in STATE_NAMES:
            df[f"State_{name}"] = (aligned == name).astype(float)
        df["Market_State"] = aligned
        return df, latest_id, latest_name, summary, None
    except Exception as e:
        log.warning(f"KMeans 失敗：{e}")
        df["Market_State"] = None
        return df, -1, "未知", pd.DataFrame(), str(e)


# ============================================================
# 資料集：purged split + 只用訓練集 fit scaler
# ============================================================

def make_dataset(df, lookback=LOOKBACK_DAYS, horizon=FORECAST_DAYS, train_ratio=TRAIN_RATIO):
    """
    兩個關鍵設計：
      1. scaler 只用訓練區間 fit，避免驗證集資訊洩漏。
      2. 訓練/驗證中間挖掉一段，避免訓練樣本的「答案」
         落在驗證樣本的「題目」裡（purged split）。
    """
    clean = df.copy()
    clean["Close_Return"] = clean["Close"] / clean["Close"].shift(1) - 1
    clean = clean.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).copy()

    need = lookback + horizon + 200
    if len(clean) < need:
        raise ValueError(f"有效資料僅 {len(clean)} 筆，至少需要 {need} 筆，請拉長歷史年數。")

    split_row = int(len(clean) * train_ratio)

    f_scaler = StandardScaler().fit(clean[FEATURE_COLUMNS].iloc[:split_row])
    t_scaler = StandardScaler().fit(clean[TARGET_COLUMNS].iloc[:split_row])

    fv = f_scaler.transform(clean[FEATURE_COLUMNS])
    tv = t_scaler.transform(clean[TARGET_COLUMNS])

    xs, ys = [], []
    for end in range(lookback, len(clean) - horizon + 1):
        xs.append(fv[end - lookback:end])
        ys.append(tv[end:end + horizon].reshape(-1))

    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    n_train = split_row - lookback - horizon + 1   # 目標完全落在訓練區
    val_start = split_row                          # 輸入完全落在驗證區

    if n_train < 50 or len(X) - val_start < 10:
        raise ValueError("purged split 後樣本不足，請增加歷史年數。")

    info = {
        "總樣本": len(X), "訓練樣本": int(n_train),
        "驗證樣本": int(len(X) - val_start), "挖空樣本": int(val_start - n_train),
        "切分日": str(clean.index[split_row].date()),
    }
    return X[:n_train], y[:n_train], X[val_start:], y[val_start:], clean, f_scaler, t_scaler, info


# ============================================================
# TCN
# ============================================================

def build_model(lookback: int, n_features: int, horizon: int):
    import tensorflow as tf
    from tensorflow.keras.layers import (Add, BatchNormalization, Conv1D, Dense,
                                         Dropout, GlobalAveragePooling1D)
    tf.random.set_seed(RANDOM_SEED)

    def block(x, filters, k, dilation, drop=0.15):
        res = x
        y = Conv1D(filters, k, padding="causal", dilation_rate=dilation, activation="relu")(x)
        y = BatchNormalization()(y)
        y = Dropout(drop)(y)
        y = Conv1D(filters, k, padding="causal", dilation_rate=dilation, activation="relu")(y)
        y = BatchNormalization()(y)
        y = Dropout(drop)(y)
        if res.shape[-1] != filters:
            res = Conv1D(filters, 1, padding="same")(res)
        return Add()([res, y])

    inputs = tf.keras.Input(shape=(lookback, n_features))
    x = Conv1D(64, 3, padding="causal", activation="relu")(inputs)
    for d in [1, 2, 4, 8, 16]:
        x = block(x, 64, 3, d)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.20)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(horizon, name="future_return")(x)

    model = tf.keras.Model(inputs, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model


def train_model(X_tr, y_tr, X_va, y_va, lookback, horizon, progress_cb=None):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    model = build_model(lookback, X_tr.shape[2], horizon)
    callbacks = [EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]

    if progress_cb is not None:
        class _P(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_cb(epoch + 1, TRAIN_EPOCHS, logs or {})
        callbacks.append(_P())

    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                        epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE, shuffle=False,
                        callbacks=callbacks, verbose=0)
    return model, history


# ============================================================
# 診斷（含 naive baseline 對照）
# ============================================================

def diagnose(model, X_va, y_va, t_scaler) -> dict:
    pred = t_scaler.inverse_transform(model.predict(X_va, verbose=0).reshape(-1, 1)).flatten()
    actual = t_scaler.inverse_transform(y_va.reshape(-1, 1)).flatten()

    mae_model = float(np.mean(np.abs(actual - pred)))
    rmse_model = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae_naive = float(np.mean(np.abs(actual)))          # baseline：永遠猜 0
    rmse_naive = float(np.sqrt(np.mean(actual ** 2)))

    std_pred, std_actual = float(pred.std()), float(actual.std())
    ratio = std_pred / std_actual if std_actual > 0 else 0.0

    mask = actual != 0
    direction = float((np.sign(pred[mask]) == np.sign(actual[mask])).mean()) if mask.any() else 0.0

    if ratio < 0.10:
        verdict = "⚠️ 模型輸出幾乎是常數，等同沒有預測能力（典型的 MSE 塌陷）"
    elif mae_model >= mae_naive:
        verdict = "⚠️ 模型未優於 naive baseline"
    elif direction < 0.52:
        verdict = "🟡 誤差略優於 baseline，但方向準確率接近隨機"
    else:
        verdict = "🟢 模型在誤差與方向上都優於 baseline"

    return {"mae_model": mae_model, "rmse_model": rmse_model,
            "mae_naive": mae_naive, "rmse_naive": rmse_naive,
            "std_pred": std_pred, "std_actual": std_actual, "std_ratio": ratio,
            "direction_acc": direction, "verdict": verdict}


# ============================================================
# 預測未來（收盤價折線 + 不確定性區間）
# ============================================================

def predict_future(model, clean_df, f_scaler, t_scaler,
                   lookback=LOOKBACK_DAYS, horizon=FORECAST_DAYS) -> pd.DataFrame:
    latest = clean_df[FEATURE_COLUMNS].iloc[-lookback:]
    if len(latest) != lookback:
        raise ValueError(f"最新資料不足 {lookback} 個交易日。")

    x = f_scaler.transform(latest).reshape(1, lookback, len(FEATURE_COLUMNS))
    scaled = model.predict(x, verbose=0)[0].reshape(horizon, 1)
    returns = t_scaler.inverse_transform(scaled).flatten()

    recent = clean_df["Close"].pct_change().dropna().tail(60)
    sigma = float(recent.std()) if len(recent) > 1 else 0.01

    limit = float(np.clip(sigma * 3, 0.01, 0.10))
    returns = np.clip(returns, -limit, limit)

    last_close = float(clean_df["Close"].iloc[-1])
    closes = last_close * np.cumprod(1 + returns)

    steps = np.arange(1, horizon + 1)
    band = closes * sigma * np.sqrt(steps)     # 預測愈遠，不確定性愈大

    dates = pd.bdate_range(start=clean_df.index[-1] + pd.Timedelta(days=1), periods=horizon)

    return pd.DataFrame({
        "Close": closes, "Return": returns,
        "Upper": closes + band, "Lower": np.maximum(closes - band, 0.01),
    }, index=dates)
