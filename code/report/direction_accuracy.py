"""
方向胜率分析 (作业 5.1 可选评估指标):
  - 信号方向准确率 = sign(score - median(score)) == sign(label) 的比例
  - 多头胜率 = top-K 持仓中实际正收益 (label > 0) 的比例
  - 空头胜率 = bottom-K 中实际负收益的比例
Output:
  output/direction_accuracy.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"


def direction_accuracy(df_sig, top_k_ratio=0.1):
    """
    df_sig: trade_date, ts_code, score, label
    返回:
      sign_acc: sign(score-median) == sign(label) 的比例
      long_winrate: top-K 中 label > 0 的比例
      short_winrate: bottom-K 中 label < 0 的比例
      long_minus_short: 多空 spread (top-K label均值 - bottom-K label均值)
    """
    sign_accs, long_wrs, short_wrs, lms_list = [], [], [], []
    for d, day in df_sig.groupby("trade_date"):
        if len(day) < 30:
            continue
        med = day["score"].median()
        sign_pred = np.sign(day["score"].values - med)
        sign_true = np.sign(day["label"].values)
        valid = (sign_true != 0)  # 排除 label=0 的中位数样本
        if valid.sum() == 0:
            continue
        sign_acc = (sign_pred[valid] == sign_true[valid]).mean()

        k = max(int(len(day) * top_k_ratio), 5)
        top = day.nlargest(k, "score")
        bot = day.nsmallest(k, "score")
        long_wr = (top["label"] > 0).mean()
        short_wr = (bot["label"] < 0).mean()
        lms = top["label"].mean() - bot["label"].mean()

        sign_accs.append(sign_acc)
        long_wrs.append(long_wr)
        short_wrs.append(short_wr)
        lms_list.append(lms)

    return {
        "n_days": len(sign_accs),
        "sign_accuracy_mean": float(np.mean(sign_accs)),
        "sign_accuracy_std": float(np.std(sign_accs)),
        "long_winrate_mean": float(np.mean(long_wrs)),
        "short_winrate_mean": float(np.mean(short_wrs)),
        "long_minus_short": float(np.mean(lms_list)),
        "long_minus_short_std": float(np.std(lms_list)),
    }


def main():
    print("Computing direction accuracy for each model ...")
    results = {}
    for name in ["mlp", "lgbm", "gru", "master", "master_v2", "master_v3", "ensemble"]:
        path = OUTPUT / "signals" / f"{name}_test.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "label" not in df.columns:
            continue
        m = direction_accuracy(df)
        results[name] = m
        print(f"\n  {name}:")
        for k, v in m.items():
            print(f"    {k:25s} {v:.4f}" if isinstance(v, float) else f"    {k:25s} {v}")

    with open(OUTPUT / "direction_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT / 'direction_accuracy.json'}")


if __name__ == "__main__":
    main()
