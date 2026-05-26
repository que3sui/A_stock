"""
LightGBM baseline:
  训练 2016-2022 / 验证 2023 / 测试 2024-2025
  目标: 预测 5日横截面 rank label
  评估: 日度横截面 Pearson IC, Spearman RankIC, ICIR
Output:
  output/checkpoints/lgbm.pkl
  output/signals/lgbm_test.parquet  (trade_date, ts_code, score, label)
  output/lgbm_metrics.json
"""
import json
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
(OUTPUT / "checkpoints").mkdir(parents=True, exist_ok=True)
(OUTPUT / "signals").mkdir(parents=True, exist_ok=True)

FACTOR_COLS = [
    "mom_5", "mom_20", "mom_60", "mom_120",
    "rev_1", "rev_5",
    "vol_20", "vol_60",
    "turnover_20", "amihud_20",
    "mf_net_5", "mf_lg_strength", "mf_elg_strength",
    "pe_ttm_rank", "pb_rank", "circ_mv_log",
    "rsi_14", "bias_20", "vwap_dev", "vol_zscore",
]


def compute_ic(df, score_col="score", label_col="label", min_n=30):
    """日度横截面 IC: Pearson + Spearman(rankIC)"""
    rows = []
    for date, day in df.groupby("trade_date"):
        d = day.dropna(subset=[score_col, label_col])
        if len(d) < min_n or d[score_col].std() == 0 or d[label_col].std() == 0:
            continue
        rows.append({
            "trade_date": date,
            "n": len(d),
            "ic": d[score_col].corr(d[label_col]),
            "rank_ic": d[score_col].rank().corr(d[label_col].rank()),
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("Loading features & labels ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    df = feats.merge(labels[["trade_date", "ts_code", "label"]],
                     on=["trade_date", "ts_code"], how="inner")
    print(f"  merged: {df.shape} ({time.time()-t0:.1f}s)")

    train = df[df["trade_date"] < 20230101].dropna(subset=["label"])
    valid = df[(df["trade_date"] >= 20230101) & (df["trade_date"] < 20240101)].dropna(subset=["label"])
    test = df[df["trade_date"] >= 20240101].dropna(subset=["label"])
    print(f"  train={len(train):,}  valid={len(valid):,}  test={len(test):,}")

    X_train = train[FACTOR_COLS].fillna(0).values.astype(np.float32)
    y_train = train["label"].values.astype(np.float32)
    X_valid = valid[FACTOR_COLS].fillna(0).values.astype(np.float32)
    y_valid = valid["label"].values.astype(np.float32)
    X_test = test[FACTOR_COLS].fillna(0).values.astype(np.float32)

    print("\nTraining LightGBM ...")
    model = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=63,
        learning_rate=0.03,
        n_estimators=2000,
        feature_fraction=0.85,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    print("\nPredicting on test set ...")
    test = test.copy()
    test["score"] = model.predict(X_test)

    # IC 评估
    print("\nComputing IC on test set ...")
    ic = compute_ic(test)
    metrics = {
        "test_rows": int(len(test)),
        "test_days": int(ic.shape[0]),
        "ic_mean": float(ic["ic"].mean()),
        "ic_std": float(ic["ic"].std()),
        "icir": float(ic["ic"].mean() / ic["ic"].std()),
        "rank_ic_mean": float(ic["rank_ic"].mean()),
        "rank_ic_std": float(ic["rank_ic"].std()),
        "rank_icir": float(ic["rank_ic"].mean() / ic["rank_ic"].std()),
        "best_iter": int(model.best_iteration_),
    }
    print("\n=== LightGBM Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:18s} {v:.4f}" if isinstance(v, float) else f"  {k:18s} {v}")

    # 注意 IC 年化 IR
    metrics["icir_annual"] = float(metrics["icir"] * np.sqrt(252))
    metrics["rank_icir_annual"] = float(metrics["rank_icir"] * np.sqrt(252))
    print(f"  icir_annual        {metrics['icir_annual']:.4f}")
    print(f"  rank_icir_annual   {metrics['rank_icir_annual']:.4f}")

    # 特征重要性
    importance = pd.DataFrame({
        "feature": FACTOR_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importance:")
    print(importance.to_string(index=False))

    # 保存
    with open(OUTPUT / "checkpoints" / "lgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    test[["trade_date", "ts_code", "score", "label"]].to_parquet(
        OUTPUT / "signals" / "lgbm_test.parquet", index=False)
    with open(OUTPUT / "lgbm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    importance.to_csv(OUTPUT / "lgbm_feature_importance.csv", index=False)

    print(f"\nOK (total {time.time()-t0:.1f}s). Saved:")
    print(f"  {OUTPUT / 'checkpoints' / 'lgbm.pkl'}")
    print(f"  {OUTPUT / 'signals' / 'lgbm_test.parquet'}")
    print(f"  {OUTPUT / 'lgbm_metrics.json'}")


if __name__ == "__main__":
    main()
