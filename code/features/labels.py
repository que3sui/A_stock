"""
计算未来5日累计log收益, 横截面rank做label
设计:
  - 在 t 时刻预测从 t+1 到 t+5 的 5 日累计对数收益
  - 用 pct_chg 累计 (已经除权调整)
  - rank 到 [-0.5, 0.5] 区间, 横截面分布均匀
Output: cache/labels.parquet (trade_date, ts_code, fwd_5d_log_ret, label)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"


def build():
    print("Loading panel ...")
    df = pd.read_parquet(
        CACHE / "panel.parquet",
        columns=["ts_code", "trade_date", "pct_chg"],
    )
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"  shape: {df.shape}")

    print("Computing 5-day forward log return ...")
    df["log_ret"] = np.log1p(df["pct_chg"] / 100)
    g = df.groupby("ts_code", sort=False)["log_ret"]

    # 从 t+1 到 t+5 的 5 个 log_ret 之和
    fwd = None
    for k in range(1, 6):
        s = g.shift(-k)
        fwd = s if fwd is None else fwd + s
    df["fwd_5d_log_ret"] = fwd

    print("Computing cross-section rank label ...")
    df["label"] = df.groupby("trade_date")["fwd_5d_log_ret"].rank(pct=True) - 0.5

    # 统计
    print("\n[Label stats]")
    na = df["label"].isna().mean() * 100
    q01, q50, q99 = df["label"].quantile([0.01, 0.50, 0.99])
    print(f"  label          na={na:5.1f}%  q01={q01:.4f}  q50={q50:.4f}  q99={q99:.4f}")
    na = df["fwd_5d_log_ret"].isna().mean() * 100
    q01, q50, q99 = df["fwd_5d_log_ret"].quantile([0.01, 0.50, 0.99])
    print(f"  fwd_5d_log_ret na={na:5.1f}%  q01={q01:.4f}  q50={q50:.4f}  q99={q99:.4f}")

    # 转 float32
    df["fwd_5d_log_ret"] = df["fwd_5d_log_ret"].astype("float32")
    df["label"] = df["label"].astype("float32")

    out = CACHE / "labels.parquet"
    df[["trade_date", "ts_code", "fwd_5d_log_ret", "label"]].to_parquet(out, index=False)
    print(f"\nOK saved: {out} ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    build()
