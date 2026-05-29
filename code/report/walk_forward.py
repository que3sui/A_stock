"""
Walk-forward 滚动窗口验证: 每年重训 MASTER v1, 验证逐年稳定性。

年份切分: 2017-2018训/2019验/2020测, 2018-2019训/2020验/2021测, ...
每窗口用 2年训练 + 1年验证 + 1年测试。

Output: output/walk_forward.json + output/reports/figs/walk_forward.png
"""
import json, time, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
FIGS = OUTPUT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

from code.models.master import (
    MASTER, DailyPanelDataset, collate_single,
    combined_loss, evaluate, prepare,
    FACTOR_COLS, T, N_FEAT, N_WEIGHT,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def train_and_eval(train_end, valid_end, test_end, X, y, trade_dates, ts_codes,
                    market_X, market_date_idx, X_w, device, F_market, F_weight):
    """训练一个窗口并评估测试集"""
    torch.manual_seed(42); np.random.seed(42)

    train_mask = trade_dates <= train_end
    # 需要至少 T 天历史
    valid_mask = (trade_dates > train_end) & (trade_dates <= valid_end)
    test_mask = (trade_dates > valid_end) & (trade_dates <= test_end)

    if valid_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    full_ds = DailyPanelDataset(X, y, trade_dates, ts_codes, market_X, market_date_idx, T, X_w=X_w)
    train_dates = sorted(set(trade_dates[train_mask]) & set(full_ds.dates))
    valid_dates = sorted(set(trade_dates[valid_mask]) & set(full_ds.dates))
    test_dates = sorted(set(trade_dates[test_mask]) & set(full_ds.dates))

    if len(train_dates) < 100 or len(test_dates) < 30:
        return None

    def make_sub(dates):
        sub = object.__new__(DailyPanelDataset)
        sub.X, sub.X_w, sub.y = X, X_w, y
        sub.trade_dates, sub.market_X = trade_dates, market_X
        sub.market_date_idx, sub.T = market_date_idx, T
        sub.dates, sub.date_to_endpoints = dates, full_ds.date_to_endpoints
        return sub

    train_loader = DataLoader(make_sub(train_dates), batch_size=1, shuffle=True, collate_fn=collate_single)
    valid_loader = DataLoader(make_sub(valid_dates), batch_size=1, shuffle=False, collate_fn=collate_single)
    test_loader = DataLoader(make_sub(test_dates), batch_size=1, shuffle=False, collate_fn=collate_single)

    model = MASTER(F_stock=N_FEAT, F_market=F_market, H=64, T=T, nhead=4, dropout=0.2,
                   n_intra_layers=2, n_inter_layers=1, F_weight=F_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    best_val_ric = -1.0; patience = 0
    for epoch in range(20):
        model.train()
        for X_d, m_d, y_d, _, _, X_w_d in train_loader:
            X_d, m_d, y_d = X_d.to(device), m_d.to(device), y_d.to(device)
            X_w_d = X_w_d.to(device) if X_w_d is not None and X_w_d.numel() > 0 else None
            loss = combined_loss(model(X_d, m_d, X_w_d), y_d, alpha=0.6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        val_ic, val_ric, _, _ = evaluate(model, valid_loader, device, None)
        if val_ric > best_val_ric:
            best_val_ric = val_ric; patience = 0
        else:
            patience += 1
            if patience >= 4: break

    test_ic, test_ric, test_ric_std, _ = evaluate(model, test_loader, device, None)
    return {
        "train_end": int(train_end), "test_end": int(test_end),
        "train_days": len(train_dates), "test_days": len(test_dates),
        "test_ic": float(test_ic), "test_rank_ic": float(test_ric),
        "test_rank_ic_std": float(test_ric_std),
        "test_rank_icir": float(test_ric / (test_ric_std + 1e-8)),
    }


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")
    X, y, trade_dates, ts_codes, code_uniq, market_X, market_date_idx, _, X_w = prepare(feats, labels, market)
    F_market = market_X.shape[1]
    F_weight = N_WEIGHT if X_w is not None else 0
    print(f"  F_market={F_market}, F_weight={F_weight}")

    # 滚动窗口: train_end 从 20181231 到 20231231, 每年步进
    windows = []
    for train_end in range(20181231, 20241231, 10000):
        valid_end = train_end + 10000  # 往后一年
        test_end = valid_end + 10000
        if test_end > 20260528:
            test_end = 20260528

        result = train_and_eval(train_end, valid_end, test_end, X, y, trade_dates, ts_codes,
                                market_X, market_date_idx, X_w, device, F_market, F_weight)
        if result:
            windows.append(result)
            print(f"  {result['train_end']} → train={result['train_days']}d test={result['test_days']}d "
                  f"RIC={result['test_rank_ic']:.4f} ICIR={result['test_rank_icir']:.2f}")

    # 全期对比
    print("\nFull-period baseline (from master_metrics.json):")
    with open(OUTPUT / "master_metrics.json") as f:
        full = json.load(f)
        print(f"  RIC={full['test_rank_ic_mean']:.4f} ICIR={full['test_rank_icir_annual']:.2f}")

    # 保存
    with open(OUTPUT / "walk_forward.json", "w") as f:
        json.dump({"windows": windows, "full_period": full}, f, indent=2)

    # 画图
    if len(windows) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        years = [str(w["test_end"] // 10000) for w in windows]
        rics = [w["test_rank_ic"] for w in windows]
        icirs = [w["test_rank_icir"] * np.sqrt(252) for w in windows]

        axes[0].bar(years, rics, color="steelblue")
        axes[0].axhline(y=full["test_rank_ic_mean"], color="red", linestyle="--", label="全期基准")
        axes[0].set_title("逐年 RankIC")
        axes[0].legend()

        axes[1].bar(years, icirs, color="darkorange")
        axes[1].axhline(y=full["test_rank_icir_annual"], color="red", linestyle="--", label="全期基准")
        axes[1].set_title("逐年 ICIR (年化)")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(FIGS / "walk_forward.png", dpi=120)
        print(f"\nSaved: {FIGS / 'walk_forward.png'}")

    print(f"OK total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
