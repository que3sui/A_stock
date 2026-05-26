"""
整合 daily/metric/moneyflow/stock_st/basic 为长面板。
Output: cache/panel.parquet (索引 trade_date+ts_code, 约200万行)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)


def load_concat(folder, drop_cols=None, desc=None):
    files = sorted((ROOT / folder).glob("*.csv"))
    dfs = []
    for f in tqdm(files, desc=desc or folder, unit="f"):
        df = pd.read_csv(f)
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build():
    print("[1/5] Loading daily ...")
    daily = load_concat("daily")
    print(f"  rows={len(daily):,}, stocks={daily['ts_code'].nunique()}")

    print("[2/5] Loading metric (drop dup close) ...")
    metric = load_concat("metric", drop_cols=["close"])
    print(f"  rows={len(metric):,}")

    print("[3/5] Loading moneyflow ...")
    mf = load_concat("moneyflow")
    print(f"  rows={len(mf):,}")

    print("[4/5] Loading stock_st & basic ...")
    st_files = sorted((ROOT / "stock_st").glob("*.csv"))
    st = pd.concat(
        [pd.read_csv(f) for f in tqdm(st_files, desc="stock_st", unit="f")],
        ignore_index=True,
    )
    st = st[["ts_code", "trade_date"]].drop_duplicates()
    st["is_st"] = True
    basic = pd.read_csv(ROOT / "basic.csv")
    print(f"  st_rows={len(st):,}, basic_stocks={len(basic):,}")

    print("[5/5] Merging ...")
    df = daily.merge(metric, on=["ts_code", "trade_date"], how="left")
    df = df.merge(mf, on=["ts_code", "trade_date"], how="left")
    df = df.merge(st, on=["ts_code", "trade_date"], how="left")
    df["is_st"] = df["is_st"].fillna(False).astype(bool)
    df = df.merge(
        basic[["ts_code", "industry", "list_date", "market"]],
        on="ts_code", how="left",
    )

    # 上市天数
    df["list_days"] = (
        pd.to_datetime(df["trade_date"], format="%Y%m%d")
        - pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    ).dt.days

    df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # 压缩 float64 -> float32 (节省一半内存与磁盘)
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    out = CACHE / "panel.parquet"
    df.to_parquet(out, index=False, compression="snappy")

    print(f"\n{'='*60}")
    print(f"OK saved: {out}")
    print(f"  shape   : {df.shape}")
    print(f"  stocks  : {df['ts_code'].nunique()}")
    print(f"  dates   : {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"  size    : {out.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"    - {c:25s} {str(df[c].dtype):12s}  na={df[c].isna().mean()*100:.1f}%")


if __name__ == "__main__":
    build()
