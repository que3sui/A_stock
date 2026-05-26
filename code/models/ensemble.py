"""
三模型 Ensemble: rank-normalize 后加权
  - 读 master / lgbm / gru 三个 *_test.parquet
  - 对每日 score 做 percentile rank
  - 加权: 0.5 * master + 0.3 * lgbm + 0.2 * gru (默认)
  - 输出 output/signals/ensemble_test.parquet

Usage:
  python -m code.models.ensemble
  python -m code.models.ensemble --w-master 0.45 --w-lgbm 0.35 --w-gru 0.2
  python -m code.models.ensemble --include-v2  # 当 master_v2 训练好后包含进来

后续运行回测:
  python -m code.backtest.engine --model ensemble --n 10 --k 2
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"


def load_signal(name):
    path = OUTPUT / "signals" / f"{name}_test.parquet"
    if not path.exists():
        print(f"WARN: {path} not found, skip")
        return None
    df = pd.read_parquet(path)
    print(f"  loaded {name}: {len(df):,} rows, "
          f"{df['trade_date'].nunique()} days, "
          f"{df['ts_code'].nunique()} stocks")
    return df


def compute_daily_ic(df):
    """逐日 IC / RankIC, 返回 (ic, rank_ic, rank_icir)"""
    ics, rank_ics = [], []
    for _, day in df.groupby("trade_date"):
        if len(day) < 30 or day["score"].std() == 0:
            continue
        ics.append(day["score"].corr(day["label"]))
        rank_ics.append(day["score"].rank().corr(day["label"].rank()))
    ic = float(np.mean(ics))
    rank_ic = float(np.mean(rank_ics))
    rank_ic_std = float(np.std(rank_ics))
    return ic, rank_ic, rank_ic_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-master", type=float, default=0.5)
    parser.add_argument("--w-lgbm", type=float, default=0.3)
    parser.add_argument("--w-gru", type=float, default=0.2)
    parser.add_argument("--include-v2", action="store_true",
                        help="若 master_v2 已训练, 用 v2 代替 v1")
    args = parser.parse_args()

    print("Loading signals ...")
    master_name = "master_v2" if args.include_v2 and (OUTPUT / "signals" / "master_v2_test.parquet").exists() else "master"
    print(f"  using master variant: {master_name}")
    sigs = {
        master_name: load_signal(master_name),
        "lgbm": load_signal("lgbm"),
        "gru": load_signal("gru"),
    }
    weight_map = {master_name: args.w_master, "lgbm": args.w_lgbm, "gru": args.w_gru}

    # 删除缺失模型
    sigs = {k: v for k, v in sigs.items() if v is not None}
    if not sigs:
        print("ERROR: no signals available")
        return
    print(f"  active models: {list(sigs.keys())}")

    # 标准化权重 (剔除缺失模型后 renormalize)
    active_weights = {k: weight_map[k] for k in sigs}
    total_w = sum(active_weights.values())
    active_weights = {k: v / total_w for k, v in active_weights.items()}
    print(f"  weights (normalized): {active_weights}")

    # 每个模型: 逐日 rank_pct
    rank_dfs = {}
    for name, df in sigs.items():
        df = df.copy()
        # 横截面 rank percentile
        df["score_rank"] = df.groupby("trade_date")["score"].rank(pct=True)
        rank_dfs[name] = df[["trade_date", "ts_code", "score_rank", "label"]].rename(
            columns={"score_rank": f"rank_{name}"}
        )

    # Outer join on (trade_date, ts_code) 然后填均值
    merged = rank_dfs[list(sigs.keys())[0]][["trade_date", "ts_code", "label"]].copy()
    for name in sigs:
        merged = merged.merge(
            rank_dfs[name][["trade_date", "ts_code", f"rank_{name}"]],
            on=["trade_date", "ts_code"], how="outer",
        )

    # 同一 (date, code) 在所有模型间应该一致, 但 outer 后 label 可能缺失, 用 max 还原
    if merged["label"].isna().any():
        # 修复 label NaN: 用任一 sig 的 label
        for name in sigs:
            ref = sigs[name][["trade_date", "ts_code", "label"]].rename(
                columns={"label": "_label"}
            )
            merged = merged.merge(ref, on=["trade_date", "ts_code"], how="left")
            merged["label"] = merged["label"].fillna(merged["_label"])
            merged = merged.drop(columns=["_label"])

    rank_cols = [f"rank_{n}" for n in sigs]
    # 缺失模型用均值填 (rank 都在 [0,1])
    merged[rank_cols] = merged[rank_cols].apply(lambda r: r.fillna(r.mean()), axis=1)

    # 加权求和
    merged["score"] = sum(active_weights[n] * merged[f"rank_{n}"] for n in sigs)
    merged = merged.dropna(subset=["score", "label"]).reset_index(drop=True)

    print(f"\nEnsemble: {len(merged):,} rows, {merged['trade_date'].nunique()} days")

    # 评估
    print("\n=== Per-model IC ===")
    for name, df in sigs.items():
        ic, rank_ic, rank_ic_std = compute_daily_ic(df)
        print(f"  {name:10s}  ic={ic:.4f}  rank_ic={rank_ic:.4f}  "
              f"icir_annual={rank_ic/(rank_ic_std+1e-8)*np.sqrt(252):.2f}")

    ic, rank_ic, rank_ic_std = compute_daily_ic(merged)
    print(f"\n=== Ensemble Test Metrics ===")
    print(f"  test_ic_mean             {ic:.4f}")
    print(f"  test_rank_ic_mean        {rank_ic:.4f}")
    print(f"  test_rank_ic_std         {rank_ic_std:.4f}")
    print(f"  test_rank_icir           {rank_ic/(rank_ic_std+1e-8):.4f}")
    print(f"  test_rank_icir_annual    {rank_ic/(rank_ic_std+1e-8)*np.sqrt(252):.4f}")

    metrics = {
        "models": list(sigs.keys()),
        "weights": active_weights,
        "test_days": int(merged["trade_date"].nunique()),
        "test_rows": int(len(merged)),
        "test_ic_mean": ic,
        "test_rank_ic_mean": rank_ic,
        "test_rank_ic_std": rank_ic_std,
        "test_rank_icir": float(rank_ic / (rank_ic_std + 1e-8)),
        "test_rank_icir_annual": float(rank_ic / (rank_ic_std + 1e-8) * np.sqrt(252)),
    }

    # 保存
    out_sig = OUTPUT / "signals" / "ensemble_test.parquet"
    merged[["trade_date", "ts_code", "score", "label"]].to_parquet(out_sig, index=False)
    print(f"\nSaved: {out_sig}")

    with open(OUTPUT / "ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
