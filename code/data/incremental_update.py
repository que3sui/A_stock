"""
增量数据更新: 新 CSV 放入目录后, 只处理新增日期, 避免全量重建。

Usage:
  python -m code.data.incremental_update                 # 自动检测新日期, 全流程更新
  python -m code.data.incremental_update --dry-run       # 只显示待处理日期, 不做任何写入
  python -m code.data.incremental_update --panel-only    # 只更新 panel, 不重算特征
  python -m code.data.incremental_update --skip-panel    # panel 已更新, 只重算特征/标签
  python -m code.data.incremental_update --from-date 20260520  # 指定起始日期

流程:
  1. 扫描 daily/ 中新增的 CSV → 增量追加 panel.parquet
  2. 增量计算新日期的因子 → 追加 factors_raw.parquet
  3. 修复尾部标签 (旧日期中因缺 forward 数据而 NaN 的 + 新日期) → 覆盖 labels.parquet
  4. 若跨月则更新 universe → 追加 universe.parquet
  5. 中性化新日期 → 追加 features.parquet
  6. 增量计算市场特征 → 追加 market_features.parquet
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

FACTOR_COLS = [
    "mom_5", "mom_20", "mom_60", "mom_120",
    "rev_1", "rev_5",
    "vol_20", "vol_60",
    "turnover_20", "amihud_20",
    "mf_net_5", "mf_lg_strength", "mf_elg_strength",
    "pe_ttm_rank", "pb_rank", "circ_mv_log",
    "rsi_14", "bias_20", "vwap_dev", "vol_zscore",
]
WEIGHT_COLS = ["hs300_weight", "hs300_dweight", "cyb_weight"]

MARKET_STATS_FILE = CACHE / "market_feature_stats.json"
MAX_ROLLING = 150  # 超过 max(mom_120, vol_60) 的 buffer
TRAIN_MAX = 20221231


# ============================================================
#   Utils
# ============================================================

def _g_rolling(series, group, window, min_periods, op):
    if op == "mean":
        return series.groupby(group, sort=False).transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean())
    if op == "sum":
        return series.groupby(group, sort=False).transform(
            lambda x: x.rolling(window, min_periods=min_periods).sum())
    if op == "std":
        return series.groupby(group, sort=False).transform(
            lambda x: x.rolling(window, min_periods=min_periods).std())
    raise ValueError(op)


def scan_new_dates(panel_path, daily_dir):
    """对比 daily/ 中的 CSV 与 panel 已有日期, 返回新增日期列表"""
    existing = set()
    if panel_path.exists():
        panel = pd.read_parquet(panel_path, columns=["trade_date"])
        existing = set(int(d) for d in panel["trade_date"].unique())

    csv_dates = set()
    for f in sorted(Path(daily_dir).glob("*.csv")):
        try:
            csv_dates.add(int(f.stem))
        except ValueError:
            pass

    new = sorted(csv_dates - existing)
    return new


# ============================================================
#   Step 1: Panel 增量
# ============================================================

def update_panel(new_dates, data_src):
    """读取新日期的 daily/metric/moneyflow CSV, merge 后追加到 panel.parquet"""
    panel_path = CACHE / "panel.parquet"
    basic = pd.read_csv(data_src / "basic.csv")
    st_files = {int(f.stem): f for f in (data_src / "stock_st").glob("*.csv")}

    rows = []
    for date in tqdm(new_dates, desc="panel"):
        date_str = str(date)

        # daily
        daily_f = data_src / "daily" / f"{date_str}.csv"
        if not daily_f.exists():
            print(f"  WARN: {daily_f} missing, skip")
            continue
        day = pd.read_csv(daily_f)

        # metric (optional)
        metric_f = data_src / "metric" / f"{date_str}.csv"
        if metric_f.exists():
            m = pd.read_csv(metric_f).drop(columns=["close"], errors="ignore")
            day = day.merge(m, on=["ts_code", "trade_date"], how="left")

        # moneyflow (optional)
        mf_f = data_src / "moneyflow" / f"{date_str}.csv"
        if mf_f.exists():
            mf = pd.read_csv(mf_f)
            day = day.merge(mf, on=["ts_code", "trade_date"], how="left")

        # ST
        if date in st_files:
            st = pd.read_csv(st_files[date])
            st = st[["ts_code", "trade_date"]].drop_duplicates()
            st["is_st"] = True
            day = day.merge(st, on=["ts_code", "trade_date"], how="left")
        else:
            day["is_st"] = False
        day["is_st"] = day["is_st"].fillna(False).astype(bool)

        # basic (industry / list_date / market)
        day = day.merge(
            basic[["ts_code", "industry", "list_date", "market"]],
            on="ts_code", how="left")

        # 上市天数
        day["list_days"] = (
            pd.to_datetime(day["trade_date"], format="%Y%m%d")
            - pd.to_datetime(day["list_date"], format="%Y%m%d", errors="coerce")
        ).dt.days

        rows.append(day)

    if not rows:
        print("  No new daily data to add.")
        return

    new_df = pd.concat(rows, ignore_index=True)
    new_df = new_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # 压缩 float64 → float32
    for col in new_df.select_dtypes(include=["float64"]).columns:
        new_df[col] = new_df[col].astype("float32")

    if panel_path.exists():
        existing = pd.read_parquet(panel_path)
        # 确保列对齐
        for col in existing.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        new_df = new_df[existing.columns.tolist()]
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        merged = merged.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    else:
        merged = new_df

    merged.to_parquet(panel_path, index=False, compression="snappy")
    print(f"  Panel updated: +{len(new_df)} rows, total={len(merged):,}  "
          f"dates={merged['trade_date'].min()}~{merged['trade_date'].max()}")


# ============================================================
#   Step 2: Factors 增量
# ============================================================

def update_factors(new_dates):
    """
    对新增日期计算因子。
    策略: 读取 panel 尾部 (last MAX_ROLLING days, 全股票), 计算全部因子后
          只保留新日期行, 追加到 factors_raw.parquet。
          必须加载全股票因为 pe_ttm_rank / pb_rank 是横截面 rank。
    """
    factors_path = CACHE / "factors_raw.parquet"
    panel = pd.read_parquet(CACHE / "panel.parquet")

    print(f"  Panel: {len(panel):,} rows, dates={panel['trade_date'].min()}~{panel['trade_date'].max()}")

    # 取尾部所有股票 (横截面因子需要全市场)
    all_dates = sorted(panel["trade_date"].unique())
    cutoff_idx = max(0, len(all_dates) - MAX_ROLLING - len(new_dates))
    cutoff_date = all_dates[cutoff_idx]
    print(f"  Loading panel tail from {cutoff_date} (~{len(all_dates) - cutoff_idx} days) ...")

    tail = panel[panel["trade_date"] >= cutoff_date]
    tail = tail.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"  Tail rows: {len(tail):,}")

    # 计算因子 (与 features/factors.py 相同逻辑)
    df = tail.copy()
    g_code = df["ts_code"]

    df["ret_1"] = df.groupby("ts_code", sort=False)["close"].pct_change(1)

    # 动量
    for n in [5, 20, 60, 120]:
        df[f"mom_{n}"] = df.groupby("ts_code", sort=False)["close"].pct_change(n)

    # 反转
    df["rev_1"] = -df["ret_1"]
    df["rev_5"] = -df["mom_5"]

    # 波动率
    df["vol_20"] = _g_rolling(df["ret_1"], g_code, 20, 10, "std")
    df["vol_60"] = _g_rolling(df["ret_1"], g_code, 60, 30, "std")

    # 流动性
    df["turnover_20"] = _g_rolling(df["turnover_rate"], g_code, 20, 10, "mean")
    log_amihud = np.log(df["ret_1"].abs() + 1e-6) - np.log(df["amount"] + 1e-6)
    df["amihud_20"] = _g_rolling(log_amihud, g_code, 20, 10, "mean")

    # 资金流
    df["mf_net_5"] = _g_rolling(df["net_mf_amount"], g_code, 5, 3, "sum")
    lg = (df["buy_lg_amount"] - df["sell_lg_amount"]) / (
        df["buy_lg_amount"] + df["sell_lg_amount"] + 1e-6)
    elg = (df["buy_elg_amount"] - df["sell_elg_amount"]) / (
        df["buy_elg_amount"] + df["sell_elg_amount"] + 1e-6)
    df["mf_lg_strength"] = _g_rolling(lg, g_code, 5, 3, "mean")
    df["mf_elg_strength"] = _g_rolling(elg, g_code, 5, 3, "mean")

    # 基本面 / 规模
    df["circ_mv_log"] = np.log(df["circ_mv"].clip(lower=1))
    df["pe_ttm_rank"] = df.groupby("trade_date")["pe_ttm"].rank(pct=True) - 0.5
    df["pb_rank"] = df.groupby("trade_date")["pb"].rank(pct=True) - 0.5

    # 技术指标
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

    # 只保留新日期
    new_rows = df[df["trade_date"].isin(new_dates)].copy()

    keep = (["trade_date", "ts_code", "industry", "circ_mv", "is_st", "list_days",
             "close", "pre_close", "vwap", "open", "vol", "amount"] + FACTOR_COLS)
    keep = list(dict.fromkeys(keep))
    new_rows = new_rows[keep]

    for col in [c for c in FACTOR_COLS if c in new_rows.columns and new_rows[c].dtype == "float64"]:
        new_rows[col] = new_rows[col].astype("float32")

    if factors_path.exists():
        existing = pd.read_parquet(factors_path)
        merged = pd.concat([existing, new_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        merged = merged.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    else:
        merged = new_rows

    merged.to_parquet(factors_path, index=False, compression="snappy")
    print(f"  Factors updated: +{len(new_rows)} rows, total={len(merged):,}  "
          f"dates covered={len(new_rows['trade_date'].unique())}")


# ============================================================
#   Step 3: Labels 修复 + 追加
# ============================================================

def update_labels(new_dates):
    """
    标签需要 forward 5 日数据。新增日期会让一些之前无法计算 label 的旧日期变得可算。
    策略: 取 panel 最后 ~15 个交易日全量重算 labels, 覆盖写入 labels.parquet
          (labels 计算很快, ~30s, 全量重算比增量更安全)
    """
    labels_path = CACHE / "labels.parquet"
    panel = pd.read_parquet(CACHE / "panel.parquet", columns=["ts_code", "trade_date", "pct_chg"])
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    all_dates = sorted(panel["trade_date"].unique())
    if len(all_dates) > 15:
        cutoff = all_dates[-15]
        panel = panel[panel["trade_date"] >= cutoff]

    print(f"  Recomputing labels for last {panel['trade_date'].nunique()} dates ...")

    panel["log_ret"] = np.log1p(panel["pct_chg"] / 100)
    g = panel.groupby("ts_code", sort=False)["log_ret"]
    fwd = None
    for k in range(1, 6):
        s = g.shift(-k)
        fwd = s if fwd is None else fwd + s
    panel["fwd_5d_log_ret"] = fwd
    panel["label"] = panel.groupby("trade_date")["fwd_5d_log_ret"].rank(pct=True) - 0.5

    panel["fwd_5d_log_ret"] = panel["fwd_5d_log_ret"].astype("float32")
    panel["label"] = panel["label"].astype("float32")
    new_labels = panel[["trade_date", "ts_code", "fwd_5d_log_ret", "label"]]

    if labels_path.exists():
        existing = pd.read_parquet(labels_path)
        # 去掉尾部会被覆盖的日期, 拼接新 labels
        keep_mask = ~existing["trade_date"].isin(new_labels["trade_date"].unique())
        existing = existing[keep_mask]
        merged = pd.concat([existing, new_labels], ignore_index=True)
        merged = merged.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    else:
        merged = new_labels

    merged.to_parquet(labels_path, index=False)
    na_pct = merged["label"].isna().mean() * 100
    print(f"  Labels updated: tail rows={len(new_labels):,}, total={len(merged):,}, "
          f"NaN={na_pct:.1f}%  (last 5 dates naturally NaN)")


# ============================================================
#   Step 4: Universe (跨月时更新)
# ============================================================

def update_universe(new_dates):
    """检查新日期是否跨月, 若是则追加新月份的 universe"""
    universe_path = CACHE / "universe.parquet"

    new_dates_arr = np.array(new_dates)
    new_yyyymm = np.unique(new_dates_arr // 100)

    if universe_path.exists():
        existing = pd.read_parquet(universe_path)
        existing_months = set(int(m) for m in existing["yyyymm"].unique())
    else:
        existing = pd.DataFrame()
        existing_months = set()

    missing_months = sorted(set(int(m) for m in new_yyyymm) - existing_months)
    if not missing_months:
        print(f"  Universe up to date (no new months in {sorted(new_yyyymm)})")
        return

    print(f"  New months for universe: {missing_months}")
    panel = pd.read_parquet(
        CACHE / "panel.parquet",
        columns=["trade_date", "ts_code", "is_st", "list_days", "circ_mv", "vol"])

    rows = []
    for yyyymm in tqdm(missing_months, desc="universe"):
        # 该月首个交易日
        month_mask = panel["trade_date"] // 100 == yyyymm
        month_dates = sorted(panel.loc[month_mask, "trade_date"].unique())
        if not month_dates:
            print(f"  WARN: no trading days in {yyyymm}, skip")
            continue
        first_date = month_dates[0]
        snap = panel[panel["trade_date"] == first_date]
        mask = (
            (~snap["is_st"])
            & (snap["list_days"].fillna(-1) >= 180)
            & (snap["circ_mv"].fillna(0) > 0)
            & (snap["vol"].fillna(0) > 0)
        )
        cand = snap[mask].nlargest(800, "circ_mv")
        rows.append(pd.DataFrame({
            "yyyymm": yyyymm,
            "trade_date": first_date,
            "ts_code": cand["ts_code"].values,
            "circ_mv": cand["circ_mv"].values,
        }))

    if rows:
        new_uni = pd.concat(rows, ignore_index=True)
        merged = pd.concat([existing, new_uni], ignore_index=True)
        merged.to_parquet(universe_path, index=False)
        print(f"  Universe updated: +{len(new_uni)} rows, total={len(merged):,}")


# ============================================================
#   Step 5: Neutralize 增量
# ============================================================

def mad_clip(arr, k=5):
    med = np.nanmedian(arr, axis=0)
    mad_val = np.nanmedian(np.abs(arr - med), axis=0) + 1e-12
    return np.clip(arr, med - k * mad_val, med + k * mad_val)


def _neutralize_one_day(day_df, factor_cols):
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
        col_means = np.nanmean(Y_use, axis=0)
        Y_fill = np.where(np.isnan(Y_use), col_means, Y_use)
        beta = np.linalg.lstsq(X_use, Y_fill, rcond=None)[0]
        Y[valid_base] = Y_use - X_use @ beta

    mean = np.nanmean(Y, axis=0)
    std = np.nanstd(Y, axis=0) + 1e-6
    Y_z = (Y - mean) / std
    return np.clip(Y_z, -5, 5).astype(np.float32)


def update_features(new_dates):
    """对新增日期做中性化, 追加到 features.parquet"""
    features_path = CACHE / "features.parquet"
    factors = pd.read_parquet(CACHE / "factors_raw.parquet")
    universe = pd.read_parquet(CACHE / "universe.parquet")

    # Filter to new dates
    new_factors = factors[factors["trade_date"].isin(new_dates)].copy()

    # Universe filter (按月份)
    new_factors["yyyymm"] = new_factors["trade_date"] // 100
    uni_set = universe[["yyyymm", "ts_code"]].drop_duplicates()
    uni_set["in_uni"] = True
    new_factors = new_factors.merge(uni_set, on=["yyyymm", "ts_code"], how="left")
    new_factors = new_factors[new_factors["in_uni"].fillna(False)].copy()
    new_factors = new_factors.drop(columns=["yyyymm", "in_uni"]).reset_index(drop=True)

    if len(new_factors) == 0:
        print("  No rows pass universe filter for new dates")
        return

    dates = sorted(new_factors["trade_date"].unique())
    date_idx = new_factors.groupby("trade_date").indices
    Y_out = np.full((len(new_factors), len(FACTOR_COLS)), np.nan, dtype=np.float32)

    for date in tqdm(dates, desc="neutralize"):
        idx = date_idx[date]
        if len(idx) < 50:
            continue
        Y_out[idx] = _neutralize_one_day(new_factors.iloc[idx], FACTOR_COLS)

    for i, c in enumerate(FACTOR_COLS):
        new_factors[c] = Y_out[:, i]

    keep = (["trade_date", "ts_code", "industry", "circ_mv", "close", "open", "vwap"]
            + FACTOR_COLS)
    extra_cols = [c for c in WEIGHT_COLS if c in new_factors.columns]
    keep = keep + extra_cols
    new_features = new_factors[keep]

    if features_path.exists():
        existing = pd.read_parquet(features_path)
        merged = pd.concat([existing, new_features], ignore_index=True)
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
        merged = merged.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    else:
        merged = new_features

    merged.to_parquet(features_path, index=False, compression="snappy")
    print(f"  Features updated: +{len(new_features)} rows, total={len(merged):,}  "
          f"dates={len(dates)}")


# ============================================================
#   Step 6: Market Features 增量
# ============================================================

INDICES = {
    "sse": "000001.SH",
    "hs300": "000300.SH",
    "cyb": "399006.SZ",
}


def _compute_idx_features(df, prefix):
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


def update_market_features(new_dates, data_src):
    """
    增量计算市场特征。
    第一次运行时保存 train 段标准化参数到 MARKET_STATS_FILE,
    后续增量更新时复用相同的 mean/std。
    """
    market_path = CACHE / "market_features.parquet"

    # 读每个指数 CSV, 取尾部 (最近 MAX_ROLLING 天 + 新日期)
    feats = []
    for prefix, code in INDICES.items():
        df = pd.read_csv(data_src / "market" / f"{code}.csv")
        df = _compute_idx_features(df, prefix)
        feats.append(df)

    full = feats[0]
    for f in feats[1:]:
        full = full.merge(f, on="trade_date", how="outer")
    full = full.sort_values("trade_date").reset_index(drop=True)

    # 市场级新闻特征: news_count only
    from code.features.news_sentiment import compute_market_news_features
    news_feats = compute_market_news_features(data_src)
    if not news_feats.empty:
        full = full.merge(news_feats[["trade_date", "news_count"]], on="trade_date", how="left")
        full["news_count"] = full["news_count"].fillna(0.0)

    feat_cols = [c for c in full.columns if c != "trade_date"]
    news_col = "news_count"

    # 加载或保存标准化参数
    if MARKET_STATS_FILE.exists():
        with open(MARKET_STATS_FILE) as f:
            train_stats = json.load(f)
        if news_col not in train_stats and news_col in full.columns:
            train_mask = full["trade_date"] <= TRAIN_MAX
            train_stats[news_col] = [float(full.loc[train_mask, news_col].mean()),
                                     float(full.loc[train_mask, news_col].std())]
            with open(MARKET_STATS_FILE, "w") as f:
                json.dump(train_stats, f, indent=2)
        print(f"  Loaded normalization stats from {MARKET_STATS_FILE}")
    else:
        train_mask = full["trade_date"] <= TRAIN_MAX
        train_stats = {
            c: [float(full.loc[train_mask, c].mean()),
                float(full.loc[train_mask, c].std())]
            for c in feat_cols
        }
        with open(MARKET_STATS_FILE, "w") as f:
            json.dump(train_stats, f, indent=2)
        print(f"  Saved normalization stats to {MARKET_STATS_FILE} "
              f"(train rows={train_mask.sum():,})")

    # 标准化
    for c in feat_cols:
        mu, sd = train_stats[c]
        full[c] = (full[c] - mu) / (sd + 1e-6)
        full[c] = full[c].astype("float32").clip(-5, 5)

    # 覆盖写入 (market features 行数少, 全量覆盖最安全)
    full.to_parquet(market_path, index=False)
    new_in = full["trade_date"].isin(new_dates).sum()
    print(f"  Market features saved: {len(full)} rows, {new_in} new dates covered")


# ============================================================
#   Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Incremental data update")
    parser.add_argument("--dry-run", action="store_true", help="只显示待处理日期")
    parser.add_argument("--panel-only", action="store_true", help="只更新 panel")
    parser.add_argument("--skip-panel", action="store_true", help="跳过 panel, 只更新特征")
    parser.add_argument("--from-date", type=int, default=0, help="指定起始日期 YYYYMMDD")
    parser.add_argument("--data-source", type=str, default="",
                        help="原始 CSV 数据源目录 (默认: 本地项目目录)。"
                             "例: --data-source E:/科大云盘/A股数据")
    args = parser.parse_args()

    data_src = Path(args.data_source) if args.data_source else ROOT
    if args.data_source:
        if not data_src.exists():
            print(f"ERROR: data source not found: {data_src}")
            return
        print(f"Data source: {data_src}")
    else:
        print(f"Data source: {data_src} (local project)")

    t0 = time.time()

    # 检测新日期 (对比 panel 已有日期 vs 数据源的 daily/)
    panel_path = CACHE / "panel.parquet"
    new_dates = scan_new_dates(panel_path, data_src / "daily")

    if args.from_date:
        new_dates = [d for d in new_dates if d >= args.from_date]

    if not new_dates:
        print("No new dates found. Everything is up to date.")
        return

    print(f"New dates to process: {len(new_dates)}")
    print(f"  Range: {new_dates[0]} ~ {new_dates[-1]}")
    if len(new_dates) <= 20:
        print(f"  Dates: {new_dates}")

    if args.dry_run:
        print("\n[Dry run] No changes made.")
        return

    print("\n" + "=" * 60)

    # Step 1: Panel
    if not args.skip_panel:
        print("[1/6] Updating panel ...")
        update_panel(new_dates, data_src)
    else:
        print("[1/6] Panel: SKIPPED")

    if args.panel_only:
        print(f"\nOK (panel-only) total={time.time() - t0:.1f}s")
        return

    # Step 2: Factors
    print("\n[2/6] Updating factors ...")
    update_factors(new_dates)

    # Step 3: Labels
    print("\n[3/6] Updating labels ...")
    update_labels(new_dates)

    # Step 4: Universe (跨月检查)
    print("\n[4/6] Updating universe ...")
    update_universe(new_dates)

    # Step 5: Neutralize
    print("\n[5/6] Updating features (neutralize) ...")
    update_features(new_dates)

    # Step 6: Market features
    print("\n[6/6] Updating market features ...")
    update_market_features(new_dates, data_src)

    print("\n" + "=" * 60)
    print(f"OK all done. total={time.time() - t0:.1f}s")
    print(f"  New dates processed: {len(new_dates)}")
    print(f"  If model retraining is needed: python -m code.models.master")


if __name__ == "__main__":
    main()
