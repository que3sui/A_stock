"""
北向资金 + 融资融券因子 (AKShare)

北向资金: 日频沪深股通净流入 (市场级, 加入 market_features)
融资融券: 日频两融余额变化 (暂未实现, 需要大量API调用)

Usage:
  from code.features.northbound import compute_northbound_features
  df = compute_northbound_features()  # 返回 [trade_date, northbound_net]
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"


def compute_northbound_features(force_refresh=False):
    """
    获取北向资金日频净流入数据。

    Returns:
        DataFrame [trade_date, northbound_net_5]  — 5日均净流入 (float32)
        若 AKShare 不可用则返回空 DataFrame
    """
    cache_path = CACHE / "northbound.parquet"
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        print(f"  northbound loaded from cache: {len(df)} days")
        return df

    try:
        import akshare as ak
    except ImportError:
        print("  WARN: akshare not installed, skip northbound")
        return pd.DataFrame()

    try:
        # 北向资金日频数据
        raw = ak.stock_hsgt_hist_em(symbol="北向资金")
        raw = raw.rename(columns={"日期": "trade_date", "当日成交净买额": "net_flow"})
        raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.strftime("%Y%m%d").astype(int)
        raw = raw.sort_values("trade_date").reset_index(drop=True)

        # 5日滚动均线
        raw["northbound_net_5"] = raw["net_flow"].rolling(5, min_periods=3).mean()
        raw["northbound_net_5"] = raw["northbound_net_5"].astype("float32")

        out = raw[["trade_date", "northbound_net_5"]].dropna()
        out.to_parquet(cache_path, index=False)
        print(f"  northbound saved: {len(out)} days, "
              f"range [{out.northbound_net_5.min():.0f}, {out.northbound_net_5.max():.0f}] (亿)")
        return out

    except Exception as e:
        print(f"  WARN: northbound fetch failed: {e}")
        return pd.DataFrame()


def compute_margin_features():
    """
    融资融券余额变化因子 (个股级, 暂未实现)
    需要调用 stock_margin_detail_sse/szse 逐股拉取, 量太大。
    """
    return pd.DataFrame()


if __name__ == "__main__":
    df = compute_northbound_features(force_refresh=True)
    if not df.empty:
        print(df.tail(10))
