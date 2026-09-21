# -*- coding: utf-8 -*-
"""金融計算：年化報酬率 XIRR、最大回撤 MDD。"""
import numpy as np
import pandas as pd


def xirr(cashflows, dates, lo=-0.9999, hi=10.0, tol=1e-7, max_iter=300) -> float:
    """
    年化內部報酬率。用二分法解，不需要額外套件。

    cashflows : 投入記為「負數」，期末市值記為「正數」
    dates     : 對應的日期清單
    回傳      : 年化報酬率（0.12 代表 12%）
    """
    if not cashflows or len(cashflows) != len(dates):
        return float("nan")

    d0 = pd.Timestamp(dates[0])

    def npv(rate):
        total = 0.0
        for cf, d in zip(cashflows, dates):
            years = (pd.Timestamp(d) - d0).days / 365.0
            total += cf / ((1 + rate) ** years)
        return total

    if npv(lo) * npv(hi) > 0:      # 區間內無解
        return float("nan")

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return (lo + hi) / 2


def max_drawdown(series: pd.Series) -> float:
    """最大回撤（負數百分比）。從歷史高點跌下來最慘的一次。"""
    s = pd.Series(series).dropna()
    if s.empty:
        return 0.0
    peak = s.cummax()
    dd = (s - peak) / peak.replace(0, np.nan)
    return float(dd.min() * 100) if dd.notna().any() else 0.0
