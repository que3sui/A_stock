"""
显式时序滤波器因子: EMA, 趋势强度, 趋势加速度

基于 attention 分析结论 (HHI≈0.05 = 几乎无时序选择性):
  在现有因子基础上, 加入显式滤波器结果作为新信息。

因子:
  ema_mom_5    — 5日EMA收益率
  ema_mom_20   — 20日EMA收益率
  ema_cross    — ema_5/ema_20 - 1 (MACD思路, 快慢线交叉)
  trend_r2     — 20日线性回归 R² (趋势稳定性, 0=乱, 1=直线)
  trend_slope  — 20日线性回归斜率 (趋势方向+强度, 年化)
  vol_trend    — 波动率/趋势斜率绝对值 (噪声比, 低=趋势清晰)

Usage:
  python -m code.features.filter_factors
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"

FILTER_COLS = ["ema_mom_5", "ema_mom_20", "ema_cross",
               "trend_r2", "trend_slope", "vol_trend"]


def _g_ema(series, group, span):
    """groupby EMA"""
    return series.groupby(group, sort=False).transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )


def _g_rolling_slope_r2(series, group, window=20):
    """每只股票滚动20日 OLS: 斜率 + R²"""
    def _ols(y):
        y = y.values
        x = np.arange(len(y), dtype=np.float64)
        x = x - x.mean()
        if (x_ss := (x**2).sum()) < 1e-12:
            return 0.0, 0.0
        slope = (x * y).sum() / x_ss
        y_pred = slope * x + y.mean()
        ss_res = ((y - y_pred)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        return slope, max(0.0, min(1.0, r2))

    result = series.groupby(group, sort=False).transform(
        lambda x: x.rolling(window, min_periods=10).apply(
            lambda y: _ols(y)[0], raw=False
        )
    )
    return result


def build():
    t0 = time.time()
    print("Loading panel ...")
    panel = pd.read_parquet(CACHE / "panel.parquet",
                            columns=["ts_code", "trade_date", "close", "pct_chg", "ret_1"
                                     ] if "ret_1" in pd.read_parquet(CACHE / "panel.parquet").columns
                            else ["ts_code", "trade_date", "close", "pct_chg"])
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # Ensure ret_1 exists
    if "ret_1" not in panel.columns:
        panel["ret_1"] = panel.groupby("ts_code", sort=False)["close"].pct_change(1)

    g_code = panel["ts_code"]
    close = panel["close"]

    # 1) EMA 因子
    print("[1/4] EMA factors ...")
    panel["ema_5"] = _g_ema(close, g_code, 5)
    panel["ema_20_val"] = _g_ema(close, g_code, 20)
    panel["ema_mom_5"] = panel["ema_5"] / panel["ema_5"].groupby(g_code, sort=False).shift(5) - 1
    panel["ema_mom_20"] = panel["ema_20_val"] / panel["ema_20_val"].groupby(g_code, sort=False).shift(20) - 1
    panel["ema_cross"] = panel["ema_5"] / (panel["ema_20_val"] + 1e-6) - 1

    # 2) 趋势强度 (20日 OLS)
    print("[2/4] Trend strength (20-day OLS) ...")
    # Use log price for better linearity
    log_close = np.log(panel["close"].clip(lower=0.01))

    def _ols_slope_r2(y_series, group):
        slopes = np.full(len(y_series), np.nan, dtype=np.float32)
        r2s = np.full(len(y_series), np.nan, dtype=np.float32)
        for code, idx in y_series.groupby(group, sort=False).indices.items():
            y = y_series.iloc[idx].values
            for i in range(10, len(y) + 1):  # min 10 points
                yi = y[max(0, i-20):i]
                x = np.arange(len(yi), dtype=np.float64)
                x = x - x.mean()
                x_ss = (x**2).sum()
                if x_ss < 1e-12:
                    continue
                slope = (x * yi).sum() / x_ss
                y_pred = slope * x + yi.mean()
                ss_res = ((yi - y_pred)**2).sum()
                ss_tot = ((yi - yi.mean())**2).sum()
                r2 = 1.0 - ss_res / (ss_tot + 1e-12)
                slopes[idx[i-1]] = slope * 252  # annualize
                r2s[idx[i-1]] = max(0.0, min(1.0, r2))
        return slopes, r2s

    slopes_arr, r2_arr = _ols_slope_r2(log_close, g_code)
    panel["trend_slope"] = slopes_arr
    panel["trend_r2"] = r2_arr

    # 3) 噪声比
    print("[3/4] Volatility-to-trend ratio ...")
    ret_1 = panel["ret_1"]
    vol_20 = ret_1.groupby(g_code, sort=False).transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    panel["vol_trend"] = vol_20 / (np.abs(panel["trend_slope"]) + 1e-6)

    # 4) 清理 NaN
    print("[4/4] Cleaning ...")
    for c in FILTER_COLS:
        panel[c] = panel[c].fillna(0.0).astype("float32")

    # 统计
    out = panel[["trade_date", "ts_code"] + FILTER_COLS]
    out.to_parquet(CACHE / "filter_factors.parquet", index=False, compression="snappy")

    print(f"\nFilter factors saved: {CACHE / 'filter_factors.parquet'}")
    print(f"  rows: {len(out):,}  dates: {out['trade_date'].nunique()}")
    for c in FILTER_COLS:
        na = out[c].isna().mean() * 100
        q01, q50, q99 = out[c].quantile([0.01, 0.50, 0.99])
        print(f"  {c:16s}  na={na:.1f}%  q01={q01:+.4f}  q50={q50:+.4f}  q99={q99:+.4f}")
    print(f"  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    build()
