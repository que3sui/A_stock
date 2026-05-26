"""
横截面中性化:
  1. universe 过滤 (只保留每月中证800近似池)
  2. MAD 去极值 (按日)
  3. 行业 + log市值 OLS 残差化 (按日 batch over 所有因子)
  4. 横截面 Z-score + clip(-5,5)
Output: cache/features.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"

FACTOR_COLS = [
    "mom_5", "mom_20", "mom_60", "mom_120",
    "rev_1", "rev_5",
    "vol_20", "vol_60",
    "turnover_20", "amihud_20",
    "mf_net_5", "mf_lg_strength", "mf_elg_strength",
    "pe_ttm_rank", "pb_rank", "circ_mv_log",
    "rsi_14", "bias_20", "vwap_dev", "vol_zscore",
]


def mad_clip(arr, k=5):
    """MAD 去极值: 中位数 ± k*MAD (列方向)"""
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0) + 1e-12
    lower = med - k * mad
    upper = med + k * mad
    return np.clip(arr, lower, upper)


def neutralize_one_day(day_df, factor_cols):
    """对一天的所有因子做: MAD去极值 → 行业+市值OLS残差 → Z-score"""
    Y = day_df[factor_cols].values.astype(np.float64).copy()
    Y = mad_clip(Y)

    industry = day_df["industry"]
    log_mv = day_df["circ_mv_log"].values.astype(np.float64)

    valid_base = (~np.isnan(log_mv)) & industry.notna().values
    if valid_base.sum() >= 50:
        ind_dummies = pd.get_dummies(industry, drop_first=True).values.astype(np.float64)
        X_full = np.column_stack([np.ones(len(day_df)), ind_dummies, log_mv])
        X_use = X_full[valid_base]
        Y_use = Y[valid_base]

        # NaN -> 列均值, 用于 OLS 拟合; 残差计算时仍用原始 Y_use (NaN 自动传播)
        col_means = np.nanmean(Y_use, axis=0)
        Y_fill = np.where(np.isnan(Y_use), col_means, Y_use)

        beta = np.linalg.lstsq(X_use, Y_fill, rcond=None)[0]  # (p, n_factors)
        Y[valid_base] = Y_use - X_use @ beta

    # Z-score
    mean = np.nanmean(Y, axis=0)
    std = np.nanstd(Y, axis=0) + 1e-6
    Y_z = (Y - mean) / std
    return np.clip(Y_z, -5, 5).astype(np.float32)


def build():
    t0 = time.time()
    print("Loading factors_raw ...")
    df = pd.read_parquet(CACHE / "factors_raw.parquet")
    print(f"  shape: {df.shape} ({time.time()-t0:.1f}s)")

    print("Loading universe & filtering ...")
    uni = pd.read_parquet(CACHE / "universe.parquet")
    uni_df = uni[["yyyymm", "ts_code"]].copy()
    uni_df["in_uni"] = True
    df["yyyymm"] = df["trade_date"] // 100
    df = df.merge(uni_df, on=["yyyymm", "ts_code"], how="left")
    df = df[df["in_uni"].fillna(False)].copy()
    df = df.drop(columns=["yyyymm", "in_uni"]).reset_index(drop=True)
    print(f"  after universe: {len(df):,} rows ({time.time()-t0:.1f}s)")

    print("Neutralizing by day ...")
    dates = np.sort(df["trade_date"].unique())
    Y_out = np.full((len(df), len(FACTOR_COLS)), np.nan, dtype=np.float32)
    date_idx = df.groupby("trade_date").indices

    for date in tqdm(dates, desc="neutralize"):
        idx = date_idx[date]
        if len(idx) < 50:
            continue
        Y_out[idx] = neutralize_one_day(df.iloc[idx], FACTOR_COLS)

    for i, c in enumerate(FACTOR_COLS):
        df[c] = Y_out[:, i]

    print(f"\n[Stats after neutralization] (time={time.time()-t0:.1f}s)")
    for c in FACTOR_COLS:
        na = df[c].isna().mean() * 100
        q01, q50, q99 = df[c].quantile([0.01, 0.50, 0.99])
        print(f"  {c:18s}  na={na:5.1f}%  q01={q01:7.3f}  q50={q50:7.3f}  q99={q99:7.3f}")

    keep = (
        ["trade_date", "ts_code", "industry", "circ_mv", "close", "open", "vwap"]
        + FACTOR_COLS
    )
    out = CACHE / "features.parquet"
    df[keep].to_parquet(out, index=False, compression="snappy")
    print(f"\nOK saved: {out} ({out.stat().st_size/1024/1024:.1f} MB)  total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    build()
