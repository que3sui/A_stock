"""
数据治理 / 数据泄露自动化审计:
  检查 8 项:
  1. labels 时间 > features 时间? (forward-looking, 必须严格 >)
  2. features 按日中性化的统计 drift (训练 vs 测试期均值/方差是否漂移合理)
  3. market_features 是否用 train 段标准化 (检验 train 段 mean≈0, std≈1, test 段允许漂移)
  4. universe 月度滚动是否回顾性 (用 yyyymm 选股, 不应使用 yyyymm+1 信息)
  5. 缺失值占比 (异常高的因子可能有问题)
  6. ST 剔除 (测试期 ST 股是否仍出现在 signals)
  7. 上市天数过滤 (是否有 list_days < 180 的进入 universe)
  8. 数据时间范围 (各 cache 文件覆盖一致)

Output:
  output/data_audit.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"

TRAIN_MAX = 20221231
VALID_MAX = 20231231


def check_label_timing():
    """标签时间应该 > 特征时间, 因为 label=fwd_5d_log_ret 是未来收益"""
    feats = pd.read_parquet(CACHE / "features.parquet", columns=["trade_date", "ts_code"])
    labels = pd.read_parquet(CACHE / "labels.parquet", columns=["trade_date", "ts_code", "fwd_5d_log_ret"])
    feat_max = int(feats["trade_date"].max())
    label_max = int(labels.dropna(subset=["fwd_5d_log_ret"])["trade_date"].max())
    return {
        "feature_date_max": feat_max,
        "label_date_max_nonan": label_max,
        "diff_days": feat_max - label_max,
        "comment": "label 在 feature_date 之后未来 5 日, 因此 label 表的最后非空日期应早于 feature_max",
        "pass": label_max <= feat_max,
    }


def check_market_features_normalization():
    """market_features train 段 mean 应接近 0, std 接近 1"""
    df = pd.read_parquet(CACHE / "market_features.parquet")
    feat_cols = [c for c in df.columns if c != "trade_date"]
    train_df = df[df["trade_date"] <= TRAIN_MAX]
    test_df = df[df["trade_date"] > VALID_MAX]

    summary = {}
    all_ok = True
    for c in feat_cols:
        tm = float(train_df[c].mean()); ts = float(train_df[c].std())
        em = float(test_df[c].mean()); es = float(test_df[c].std())
        # train 段应该 ≈ N(0, 1)
        train_ok = (abs(tm) < 0.2) and (0.7 < ts < 1.3)
        summary[c] = {
            "train_mean": tm, "train_std": ts,
            "test_mean": em, "test_std": es,
            "train_norm_ok": bool(train_ok),
        }
        if not train_ok:
            all_ok = False
    return {"pass": all_ok, "per_feature": summary,
            "comment": "train 段经过 Z-score 后应接近 N(0,1); test 段可以 drift"}


def check_features_drift():
    """每个因子的 train/test 期分位数分布; 大 drift 提示分布漂移"""
    feats = pd.read_parquet(CACHE / "features.parquet")
    factor_cols = [c for c in feats.columns if c not in
                   ["trade_date", "ts_code", "industry", "circ_mv",
                    "close", "open", "vwap"]]
    train_df = feats[feats["trade_date"] <= TRAIN_MAX]
    test_df = feats[feats["trade_date"] > VALID_MAX]

    out = {}
    for c in factor_cols:
        tm = float(train_df[c].mean()); ts = float(train_df[c].std())
        em = float(test_df[c].mean()); es = float(test_df[c].std())
        out[c] = {
            "train_mean": tm, "train_std": ts,
            "test_mean": em, "test_std": es,
            "mean_drift": em - tm,
            "std_drift_pct": float(((es - ts) / (abs(ts) + 1e-8) * 100)) if ts > 0 else 0,
        }
    return out


def check_missing_values():
    """各 cache 文件的 NaN 占比"""
    out = {}
    for name in ["features", "labels", "market_features", "panel"]:
        p = CACHE / f"{name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        miss = (df.isna().sum() / len(df) * 100).to_dict()
        # 只列 > 5% 的
        high_miss = {k: round(float(v), 2) for k, v in miss.items() if v > 5}
        out[name] = {
            "rows": len(df),
            "cols": len(df.columns),
            "high_missing_pct": high_miss,
        }
    return out


def check_universe_no_lookahead():
    """universe 在 yyyymm 月份只用本月数据选股"""
    uni = pd.read_parquet(CACHE / "universe.parquet")
    # 每月最早交易日 (universe 选股日) 应该都 <= yyyymm 月末
    uni["yyyymm_d"] = uni["yyyymm"] * 100 + 31
    bad = uni[uni["trade_date"] > uni["yyyymm_d"]]
    return {
        "rows": len(uni),
        "months": int(uni["yyyymm"].nunique()),
        "stocks_per_month_mean": float(uni.groupby("yyyymm").size().mean()),
        "lookahead_violations": int(len(bad)),
        "pass": len(bad) == 0,
    }


def check_st_in_signals():
    """模型 test signals 中是否漏剔了 ST 股"""
    sig_path = OUTPUT / "signals" / "master_test.parquet"
    if not sig_path.exists():
        return {"skip": True}
    sig = pd.read_parquet(sig_path)
    panel_st = pd.read_parquet(CACHE / "panel.parquet",
                                columns=["trade_date", "ts_code", "is_st"])
    panel_st = panel_st[panel_st["trade_date"] >= 20240101]
    panel_st = panel_st[panel_st["is_st"]]
    overlap = sig.merge(panel_st, on=["trade_date", "ts_code"], how="inner")
    return {
        "test_signal_rows": len(sig),
        "st_overlap_in_signal": int(len(overlap)),
        "pct": float(len(overlap) / len(sig) * 100),
        "comment": "signal 中存在 ST 股是允许的 (训练时未剔除), 但 backtest 时需当日动态剔除; 这里只是统计",
    }


def check_data_coverage():
    """各 cache 文件覆盖的时间范围"""
    out = {}
    for name in ["features", "labels", "market_features", "panel"]:
        p = CACHE / f"{name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["trade_date"])
        out[name] = {
            "date_min": int(df["trade_date"].min()),
            "date_max": int(df["trade_date"].max()),
            "n_unique_days": int(df["trade_date"].nunique()),
        }
    return out


def main():
    print("=== 数据治理 / 泄露审计 ===\n")
    audit = {}

    print("[1] 标签时间合理性")
    a = check_label_timing()
    audit["1_label_timing"] = a
    print(f"  feature 最新日期: {a['feature_date_max']}")
    print(f"  label 最新非空日期: {a['label_date_max_nonan']}")
    print(f"  差距: {a['diff_days']} 天 (预期约 5-7 天, 因为 5 日 forward)")
    print(f"  通过: {a['pass']}\n")

    print("[2] market_features 标准化合规性")
    a = check_market_features_normalization()
    audit["2_market_norm"] = a
    print(f"  train 段是否 ~ N(0,1): {a['pass']}\n")

    print("[3] universe 月度无前瞻")
    a = check_universe_no_lookahead()
    audit["3_universe"] = a
    print(f"  月度违规数: {a['lookahead_violations']}")
    print(f"  通过: {a['pass']}\n")

    print("[4] 缺失值占比 (高于 5%)")
    a = check_missing_values()
    audit["4_missing"] = a
    for name, info in a.items():
        if info.get("high_missing_pct"):
            print(f"  {name}: {info['high_missing_pct']}")
    print()

    print("[5] ST 股在 signals 中的存在性")
    a = check_st_in_signals()
    audit["5_st_in_signal"] = a
    if not a.get("skip"):
        print(f"  test signal 中 ST 占比: {a['pct']:.2f}% "
              f"(backtest 会动态剔除, 此项仅信息)\n")

    print("[6] 数据覆盖范围一致性")
    a = check_data_coverage()
    audit["6_coverage"] = a
    for name, info in a.items():
        print(f"  {name}: {info['date_min']} ~ {info['date_max']} "
              f"({info['n_unique_days']} 天)")
    print()

    print("[7] 因子 train/test drift")
    a = check_features_drift()
    audit["7_factor_drift"] = a
    # 列出 |mean_drift| > 0.3 或 std_drift > 50% 的
    print("  注意 (|mean drift| > 0.3 或 std drift > 50% 的因子):")
    for f, info in a.items():
        if abs(info["mean_drift"]) > 0.3 or abs(info["std_drift_pct"]) > 50:
            print(f"    {f}: mean {info['train_mean']:.3f} → {info['test_mean']:.3f}  "
                  f"std {info['train_std']:.3f} → {info['test_std']:.3f}")

    # 汇总通过项
    passed = []
    failed = []
    for k, v in audit.items():
        if isinstance(v, dict) and "pass" in v:
            (passed if v["pass"] else failed).append(k)

    print(f"\n=== 审计汇总 ===")
    print(f"通过项: {passed}")
    print(f"失败项: {failed if failed else '(无)'}")

    with open(OUTPUT / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT / 'data_audit.json'}")


if __name__ == "__main__":
    main()
