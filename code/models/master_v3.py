"""
MASTER v3: Multi-seed averaging
策略: 用 v1 完全相同的架构和超参, 训练 2 个额外 seed (1337, 2024),
      预测时取 3 个模型 (含原 seed=42) score 的 rank-percentile 平均.

为什么这样做:
  - v2 (加深+加宽+长T) 反而过拟合 → v1 容量是最优的
  - Multi-seed 是 ML 经典稳健 trick: 不增容量, 只减方差
  - 预期收益: IC 提升 0.3-0.8% (典型量级)

Output:
  output/checkpoints/master_seed{1337,2024}.pt
  output/signals/master_v3_test.parquet  (3 seeds 平均)
  output/master_v3_metrics.json
"""
import json
import time
import math
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# 复用 v1 的实现
from code.models.master import (
    MASTER, DailyPanelDataset, collate_single,
    combined_loss, evaluate, prepare,
    FACTOR_COLS, T, N_FEAT, N_WEIGHT, TRAIN_MAX, VALID_MAX,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
(OUTPUT / "checkpoints").mkdir(parents=True, exist_ok=True)
(OUTPUT / "signals").mkdir(parents=True, exist_ok=True)


def train_one_seed(seed, X, y, trade_dates, ts_codes, market_X, market_date_idx,
                   train_dates, valid_dates, test_dates, df_full, device):
    """用给定 seed 训练 v1 架构, 返回 test_df (含 score)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

    def make_sub(dates):
        sub = object.__new__(DailyPanelDataset)
        sub.X = X
        sub.X_w = X_w
        sub.y = y
        sub.trade_dates = trade_dates
        sub.market_X = market_X
        sub.market_date_idx = market_date_idx
        sub.T = T
        sub.dates = dates
        sub.date_to_endpoints = full_endpoints
        return sub

    train_ds = make_sub(train_dates)
    valid_ds = make_sub(valid_dates)
    test_ds = make_sub(test_dates)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              collate_fn=collate_single)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=0,
                              collate_fn=collate_single)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=collate_single)

    F_market = market_X.shape[1]
    F_weight = N_WEIGHT if X_w is not None else 0
    model = MASTER(F_stock=N_FEAT, F_market=F_market, H=64, T=T,
                   nhead=4, dropout=0.2, n_intra_layers=2, n_inter_layers=1,
                   F_weight=F_weight).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    ckpt = OUTPUT / "checkpoints" / f"master_seed{seed}.pt"
    best_val_rank_ic = -1.0
    patience = 0
    patience_max = 6
    epoch_best = 0
    max_epochs = 20

    for epoch in range(max_epochs):
        model.train()
        losses = []
        for X_d, m_d, y_d, _, _, X_w_d in train_loader:
            X_d, m_d, y_d = X_d.to(device), m_d.to(device), y_d.to(device)
            X_w_d = X_w_d.to(device) if X_w_d.numel() > 0 else None
            pred = model(X_d, m_d, X_w_d)
            loss = combined_loss(pred, y_d, alpha=0.6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_ic, val_rank_ic, _, _ = evaluate(model, valid_loader, device, None)
        print(f"  Epoch {epoch+1:2d}: train_loss={np.mean(losses):.4f}  "
              f"val_ic={val_ic:.4f}  val_rank_ic={val_rank_ic:.4f}")
        if val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = val_rank_ic
            epoch_best = epoch + 1
            patience = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= patience_max:
                print(f"  Early stop, best=epoch{epoch_best} val_rank_ic={best_val_rank_ic:.4f}")
                break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    test_ic, test_rank_ic, test_rank_ic_std, test_df = evaluate(model, test_loader, device, None)

    test_df["ts_code"] = df_full.loc[test_df["ep"].values, "ts_code"].values
    test_df = test_df.drop(columns=["ep"])
    test_df = test_df.rename(columns={"score": f"score_s{seed}"})
    print(f"  Seed {seed} test: ic={test_ic:.4f}  rank_ic={test_rank_ic:.4f}  "
          f"best_epoch={epoch_best}")
    return test_df, {"seed": seed, "best_epoch": epoch_best,
                     "best_val_rank_ic": float(best_val_rank_ic),
                     "test_ic": float(test_ic), "test_rank_ic": float(test_rank_ic)}


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    print("\nLoading features / labels / market ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")

    print("Preparing ...")
    X, y, trade_dates, ts_codes, code_uniq, market_X, market_date_idx, df_full, X_w = prepare(
        feats, labels, market
    )
    if X_w is not None:
        print(f"  weight channel: {X_w.shape[1]} cols")

    print("Building dataset ...")
    full_ds = DailyPanelDataset(X, y, trade_dates, ts_codes, market_X, market_date_idx, T, X_w=X_w)
    global full_endpoints
    full_endpoints = full_ds.date_to_endpoints

    train_dates = [d for d in full_ds.dates if d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]
    print(f"  train_days={len(train_dates)}  valid_days={len(valid_dates)}  test_days={len(test_dates)}")

    # 训练 2 个新 seed (原 seed=42 已有 master.pt)
    seeds = [1337, 2024]
    test_dfs = []
    seed_metrics = []
    for s in seeds:
        td, m = train_one_seed(s, X, y, trade_dates, ts_codes, market_X, market_date_idx,
                                train_dates, valid_dates, test_dates, df_full, device)
        test_dfs.append(td)
        seed_metrics.append(m)

    # 加载 v1 (seed=42) 的 test signals
    print("\nLoading seed=42 (v1) signals ...")
    s42 = pd.read_parquet(OUTPUT / "signals" / "master_test.parquet")
    s42 = s42[["trade_date", "ts_code", "score", "label"]].rename(columns={"score": "score_s42"})
    test_dfs.append(s42)

    # Merge 三个 seed 的 score
    print("\nMerging multi-seed predictions ...")
    merged = test_dfs[0][["trade_date", "ts_code", "label"]].copy()
    if "label" not in merged.columns:
        # label 可能在其他 df 里
        for td in test_dfs:
            if "label" in td.columns:
                merged = merged.merge(td[["trade_date", "ts_code", "label"]],
                                       on=["trade_date", "ts_code"], how="outer")
                break

    for td in test_dfs:
        score_col = [c for c in td.columns if c.startswith("score_s")][0]
        cols = ["trade_date", "ts_code", score_col]
        merged = merged.merge(td[cols], on=["trade_date", "ts_code"], how="inner")

    score_cols = [c for c in merged.columns if c.startswith("score_s")]
    print(f"  score columns: {score_cols}")
    print(f"  rows: {len(merged):,}")

    # 每日 rank-percentile 平均
    for c in score_cols:
        merged[f"{c}_rank"] = merged.groupby("trade_date")[c].rank(pct=True)
    rank_cols = [f"{c}_rank" for c in score_cols]
    merged["score"] = merged[rank_cols].mean(axis=1)

    # 计算指标
    ics, rank_ics = [], []
    for _, day in merged.groupby("trade_date"):
        if len(day) < 30 or day["score"].std() == 0:
            continue
        ics.append(day["score"].corr(day["label"]))
        rank_ics.append(day["score"].rank().corr(day["label"].rank()))
    ic = float(np.mean(ics))
    rank_ic = float(np.mean(rank_ics))
    rank_ic_std = float(np.std(rank_ics))

    metrics = {
        "n_seeds": len(score_cols),
        "seeds": seeds + [42],
        "seed_metrics": seed_metrics,
        "test_days": int(merged["trade_date"].nunique()),
        "test_rows": int(len(merged)),
        "test_ic_mean": ic,
        "test_rank_ic_mean": rank_ic,
        "test_rank_ic_std": rank_ic_std,
        "test_rank_icir": float(rank_ic / (rank_ic_std + 1e-8)),
        "test_rank_icir_annual": float(rank_ic / (rank_ic_std + 1e-8) * np.sqrt(252)),
    }

    print("\n=== MASTER v3 (multi-seed) Test Metrics ===")
    print(f"  ic={ic:.4f}  rank_ic={rank_ic:.4f}  "
          f"icir_annual={metrics['test_rank_icir_annual']:.2f}")

    merged[["trade_date", "ts_code", "score", "label"]].to_parquet(
        OUTPUT / "signals" / "master_v3_test.parquet", index=False
    )
    with open(OUTPUT / "master_v3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOK total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
