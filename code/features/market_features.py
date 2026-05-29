"""
提取市场状态特征 (MASTER 用于 market-guided gating)
3 个指数: 上证(000001.SH) + 沪深300(000300.SH) + 创业板(399006.SZ)
每个指数 4 个特征 = 12 维:
  - 当日收益 (pct_chg)
  - 5日累计收益
  - 20日波动率
  - 成交量 Z-score (60日)
Output: cache/market_features.parquet (trade_date, 12 features)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"

INDICES = {
    "sse": "000001.SH",
    "hs300": "000300.SH",
    "cyb": "399006.SZ",
}


def compute_index_features(file_name, prefix):
    df = pd.read_csv(ROOT / "market" / f"{file_name}.csv")
    df = df.sort_values("trade_date").reset_index(drop=True)

    df[f"{prefix}_ret_1"] = df["pct_chg"] / 100
    df[f"{prefix}_ret_5"] = df["close"].pct_change(5)
    df[f"{prefix}_vol_20"] = df[f"{prefix}_ret_1"].rolling(20, min_periods=10).std()
    vol_mean = df["vol"].rolling(60, min_periods=20).mean()
    vol_std = df["vol"].rolling(60, min_periods=20).std()
    df[f"{prefix}_vol_zscore"] = (df["vol"] - vol_mean) / (vol_std + 1e-6)

    return df[["trade_date",
               f"{prefix}_ret_1", f"{prefix}_ret_5",
               f"{prefix}_vol_20", f"{prefix}_vol_zscore"]]


def build():
    print("Loading & computing index features ...")
    feats = []
    for prefix, code in INDICES.items():
        f = compute_index_features(code, prefix)
        feats.append(f)
        print(f"  {prefix} ({code}): {len(f)} rows")

    df = feats[0]
    for f in feats[1:]:
        df = df.merge(f, on="trade_date", how="outer")

    df = df.sort_values("trade_date").reset_index(drop=True)

    # 市场级新闻特征暂时禁用 (news_count 未带来显著提升)
    # from code.features.news_sentiment import compute_market_news_features

    # 时间维度 Z-score: 必须用 train 段 (<=2022) 的 mean/std, 避免未来函数泄露
    feat_cols = [c for c in df.columns if c != "trade_date"]
    TRAIN_MAX = 20221231
    train_mask = df["trade_date"] <= TRAIN_MAX
    train_stats = {
        c: (df.loc[train_mask, c].mean(), df.loc[train_mask, c].std()) for c in feat_cols
    }
    print(f"\n[Normalization] using train segment mean/std (rows<={TRAIN_MAX}: "
          f"{train_mask.sum():,} / {len(df):,})")
    for c in feat_cols:
        mu, sd = train_stats[c]
        df[c] = (df[c] - mu) / (sd + 1e-6)
        df[c] = df[c].astype("float32").clip(-5, 5)

    # 统计
    print(f"\n[Market features]")
    print(f"  rows: {len(df):,}")
    print(f"  cols: {len(feat_cols)}")
    print(f"  date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    for c in feat_cols:
        na = df[c].isna().mean() * 100
        q01, q50, q99 = df[c].quantile([0.01, 0.50, 0.99])
        print(f"    {c:25s}  na={na:5.1f}%  q01={q01:7.3f}  q50={q50:7.3f}  q99={q99:7.3f}")

    out = CACHE / "market_features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nOK saved: {out} ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build()
