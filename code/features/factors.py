"""
计算 20 个因子, 输出 cache/factors_raw.parquet
按 ts_code 分组的时序因子 + 按 trade_date 横截面 rank 因子
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"

FACTOR_COLS = [
    # 动量 (4)
    "mom_5", "mom_20", "mom_60", "mom_120",
    # 反转 (2)
    "rev_1", "rev_5",
    # 波动率 (2)
    "vol_20", "vol_60",
    # 流动性 (2)
    "turnover_20", "amihud_20",
    # 资金流 (3)
    "mf_net_5", "mf_lg_strength", "mf_elg_strength",
    # 基本面 (3)
    "pe_ttm_rank", "pb_rank", "circ_mv_log",
    # 技术 (4)
    "rsi_14", "bias_20", "vwap_dev", "vol_zscore",
]


def _g_rolling(series, group, window, min_periods, op):
    """高效 groupby rolling: 一次性 groupby + transform"""
    if op == "mean":
        return series.groupby(group, sort=False).transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
    if op == "sum":
        return series.groupby(group, sort=False).transform(
            lambda x: x.rolling(window, min_periods=min_periods).sum()
        )
    if op == "std":
        return series.groupby(group, sort=False).transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )
    raise ValueError(op)


def build():
    t0 = time.time()
    print("Loading panel ...")
    df = pd.read_parquet(CACHE / "panel.parquet")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"  shape: {df.shape}  ({time.time()-t0:.1f}s)")

    g_code = df["ts_code"]

    # 基础: 日收益
    df["ret_1"] = df.groupby("ts_code", sort=False)["close"].pct_change(1)

    # --- 动量 ---
    print("[1/7] Momentum ...")
    for n in [5, 20, 60, 120]:
        df[f"mom_{n}"] = df.groupby("ts_code", sort=False)["close"].pct_change(n)

    # --- 反转 ---
    print("[2/7] Reversal ...")
    df["rev_1"] = -df["ret_1"]
    df["rev_5"] = -df["mom_5"]

    # --- 波动率 ---
    print("[3/7] Volatility ...")
    df["vol_20"] = _g_rolling(df["ret_1"], g_code, 20, 10, "std")
    df["vol_60"] = _g_rolling(df["ret_1"], g_code, 60, 30, "std")

    # --- 流动性 ---
    print("[4/7] Liquidity ...")
    df["turnover_20"] = _g_rolling(df["turnover_rate"], g_code, 20, 10, "mean")
    # Amihud 用 log 形式避免 float32 精度损失 (|ret|~0.01, amount~1e5, 比值~1e-7)
    log_amihud = np.log(df["ret_1"].abs() + 1e-6) - np.log(df["amount"] + 1e-6)
    df["amihud_20"] = _g_rolling(log_amihud, g_code, 20, 10, "mean")

    # --- 资金流 ---
    print("[5/7] Money flow ...")
    df["mf_net_5"] = _g_rolling(df["net_mf_amount"], g_code, 5, 3, "sum")
    lg = (df["buy_lg_amount"] - df["sell_lg_amount"]) / (
        df["buy_lg_amount"] + df["sell_lg_amount"] + 1e-6
    )
    elg = (df["buy_elg_amount"] - df["sell_elg_amount"]) / (
        df["buy_elg_amount"] + df["sell_elg_amount"] + 1e-6
    )
    df["mf_lg_strength"] = _g_rolling(lg, g_code, 5, 3, "mean")
    df["mf_elg_strength"] = _g_rolling(elg, g_code, 5, 3, "mean")

    # --- 基本面/规模 ---
    print("[6/7] Fundamental & size ...")
    df["circ_mv_log"] = np.log(df["circ_mv"].clip(lower=1))
    # PE/PB 滞后30交易日 (财报公告日错位修正)
    df["pe_ttm"] = df.groupby("ts_code", sort=False)["pe_ttm"].shift(30)
    df["pb"] = df.groupby("ts_code", sort=False)["pb"].shift(30)
    df["pe_ttm_rank"] = df.groupby("trade_date")["pe_ttm"].rank(pct=True) - 0.5
    df["pb_rank"] = df.groupby("trade_date")["pb"].rank(pct=True) - 0.5

    # --- 技术 ---
    print("[7/7] Technical ...")
    gain = df["ret_1"].clip(lower=0)
    loss = (-df["ret_1"]).clip(lower=0)
    avg_gain = _g_rolling(gain, g_code, 14, 7, "mean")
    avg_loss = _g_rolling(loss, g_code, 14, 7, "mean")
    df["rsi_14"] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-6))

    ma20 = _g_rolling(df["close"], g_code, 20, 10, "mean")
    df["bias_20"] = (df["close"] - ma20) / (ma20 + 1e-6)

    df["vwap_dev"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-6)

    vol_mean = _g_rolling(df["vol"], g_code, 60, 30, "mean")
    vol_std = _g_rolling(df["vol"], g_code, 60, 30, "std")
    df["vol_zscore"] = (df["vol"] - vol_mean) / (vol_std + 1e-6)

    print(f"\n[Stats] (time={time.time()-t0:.1f}s)")
    for c in FACTOR_COLS:
        na = df[c].isna().mean() * 100
        q01, q50, q99 = df[c].quantile([0.01, 0.50, 0.99])
        print(f"  {c:18s}  na={na:5.1f}%  q01={q01:10.4f}  q50={q50:10.4f}  q99={q99:10.4f}")

    # 保存
    keep = (
        ["trade_date", "ts_code", "industry", "circ_mv", "is_st", "list_days",
         "close", "pre_close", "vwap", "open", "vol", "amount"]
        + FACTOR_COLS
        + ["hs300_weight", "hs300_dweight", "cyb_weight"]
    )
    keep = list(dict.fromkeys(keep))  # dedupe (circ_mv_log already in FACTOR_COLS)

    # 转 float32
    for col in [c for c in FACTOR_COLS if df[c].dtype == "float64"]:
        df[col] = df[col].astype("float32")

    out = CACHE / "factors_raw.parquet"
    df[keep].to_parquet(out, index=False, compression="snappy")
    print(f"\nOK saved: {out} ({out.stat().st_size/1024/1024:.1f} MB)  total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    build()
