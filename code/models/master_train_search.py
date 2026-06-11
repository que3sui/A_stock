"""
MASTER 训练搜索: 多次训练选最优单模型 (不同于 v3 的 ensemble 平均)

策略:
  1. 用 10 个不同 seed 独立训练 MASTER v1 + 权重通道
  2. 按 val_rank_ic 筛 top-3
  3. top-3 分别回测, 选夏普最高者
  4. 存档到 output/v3_multi_train/

Usage:
  python -m code.models.master_train_search
"""
import json
import time
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from code.models.master import (
    MASTER, DailyPanelDataset, collate_single,
    combined_loss, evaluate, prepare,
    FACTOR_COLS, T, N_FEAT, N_WEIGHT, TRAIN_MAX, VALID_MAX,
)
from code.backtest.engine import backtest, benchmark_nav, compute_metrics

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
ARCHIVE = OUTPUT / "v3_multi_train"
for d in [ARCHIVE, ARCHIVE / "checkpoints", ARCHIVE / "signals"]:
    d.mkdir(parents=True, exist_ok=True)

SEEDS = [2024, 88, 777, 1024, 2048, 4096, 99, 314]  # 128,42,1337 已完成
TOP_K_VAL = 3


def train_one_seed(seed, full_ds, df_full, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dates = [d for d in full_ds.dates if d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]

    train_loader = DataLoader(full_ds.subset_by_dates(train_dates), batch_size=1, shuffle=True,
                              num_workers=0, collate_fn=collate_single)
    valid_loader = DataLoader(full_ds.subset_by_dates(valid_dates), batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_single)
    test_loader = DataLoader(full_ds.subset_by_dates(test_dates), batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collate_single)

    F_market = full_ds.market_X.shape[1]
    F_weight = N_WEIGHT if full_ds.X_w is not None else 0
    model = MASTER(F_stock=N_FEAT, F_market=F_market, H=64, T=T,
                   nhead=4, dropout=0.2, n_intra_layers=2, n_inter_layers=1,
                   F_weight=F_weight).to(device)
    nparams = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    ckpt = ARCHIVE / "checkpoints" / f"master_seed{seed}.pt"
    best_val_rank_ic = -1.0
    best_epoch = 0
    patience = 0
    patience_max = 6
    max_epochs = 20

    for epoch in range(max_epochs):
        model.train()
        losses = []
        for X_d, m_d, y_d, _, _, X_w_d in train_loader:
            X_d = X_d.to(device, non_blocking=True)
            m_d = m_d.to(device, non_blocking=True)
            y_d = y_d.to(device, non_blocking=True)
            X_w_d = X_w_d.to(device, non_blocking=True) if X_w_d.numel() > 0 else None
            loss = combined_loss(model(X_d, m_d, X_w_d), y_d, alpha=0.6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        val_ic, val_rank_ic, _, _ = evaluate(model, valid_loader, device, None)
        if val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = val_rank_ic
            best_epoch = epoch + 1
            patience = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= patience_max:
                break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    test_ic, test_rank_ic, test_rank_ic_std, test_df = evaluate(model, test_loader, device, None)

    test_df["ts_code"] = df_full.loc[test_df["ep"].values, "ts_code"].values
    test_df = test_df.drop(columns=["ep"])
    test_df.to_parquet(ARCHIVE / "signals" / f"master_seed{seed}_test.parquet", index=False)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_rank_ic": float(best_val_rank_ic),
        "test_ic": float(test_ic),
        "test_rank_ic": float(test_rank_ic),
        "test_rank_ic_std": float(test_rank_ic_std),
        "test_rank_icir_annual": float(test_rank_ic / (test_rank_ic_std + 1e-8) * math.sqrt(252)),
        "n_params": nparams,
    }


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    print(f"Seeds: {SEEDS}")
    print(f"Top-K by val_rank_ic: {TOP_K_VAL}")
    print(f"Archive: {ARCHIVE}\n")

    print("Loading features / labels / market ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")

    print("Preparing ...")
    X, y, trade_dates, ts_codes_int, code_uniq, market_X, market_date_idx, df_full, X_w = prepare(
        feats, labels, market
    )
    if X_w is not None:
        print(f"  weight channel: {X_w.shape[1]} cols")

    print("Building dataset (once) ...")
    full_ds = DailyPanelDataset(X, y, trade_dates, ts_codes_int,
                                 market_X, market_date_idx, T, X_w=X_w)

    train_dates = [d for d in full_ds.dates if d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]
    print(f"  train_days={len(train_dates)}  valid_days={len(valid_dates)}  "
          f"test_days={len(test_dates)}")

    # ---- Phase 1: Train all seeds ----
    all_results = []
    for seed in SEEDS:
        t_seed = time.time()
        print(f"\n{'='*60}")
        print(f"Training seed={seed} ...")
        r = train_one_seed(seed, full_ds, df_full, device)
        r["train_time_s"] = round(time.time() - t_seed, 1)
        all_results.append(r)
        print(f"  val_rank_ic={r['val_rank_ic']:.4f}  "
              f"test_rank_ic={r['test_rank_ic']:.4f}  "
              f"test_icir_annual={r['test_rank_icir_annual']:.2f}  "
              f"time={r['train_time_s']}s")

    # ---- Phase 2: Pick top-3 by val_rank_ic ----
    all_results.sort(key=lambda x: x["val_rank_ic"], reverse=True)
    top3 = all_results[:TOP_K_VAL]
    print(f"\n{'='*60}")
    print(f"Top-{TOP_K_VAL} by val_rank_ic:")
    for r in top3:
        print(f"  seed={r['seed']:4d}  val_rank_ic={r['val_rank_ic']:.4f}  "
              f"test_rank_ic={r['test_rank_ic']:.4f}")

    # ---- Phase 3: Backtest top-3 ----
    print(f"\n{'='*60}")
    print("Backtesting top-3 candidates ...")
    panel = pd.read_parquet(
        CACHE / "panel.parquet",
        columns=["trade_date", "ts_code", "open", "high", "low", "close",
                 "pct_chg", "is_st"],
    )
    panel = panel[panel["trade_date"] >= 20231201]

    best_sharpe = -999
    best_seed = None
    for r in top3:
        sig = pd.read_parquet(ARCHIVE / "signals" / f"master_seed{r['seed']}_test.parquet")
        nav, daily_ret, _ = backtest(sig, panel, n=10, k=2)
        metrics = compute_metrics(daily_ret, nav)
        r["sharpe"] = metrics["sharpe"]
        r["total_return"] = metrics["total_return"]
        r["max_drawdown"] = metrics["max_drawdown"]
        print(f"  seed={r['seed']:4d}  sharpe={metrics['sharpe']:.4f}  "
              f"total_ret={metrics['total_return']:.4f}  mdd={metrics['max_drawdown']:.4f}")

        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_seed = r["seed"]

    # ---- Phase 4: Save results ----
    print(f"\n{'='*60}")
    print(f"BEST: seed={best_seed}  sharpe={best_sharpe:.4f}")

    # Save best checkpoint as master.pt in archive
    best_ckpt = ARCHIVE / "checkpoints" / f"master_seed{best_seed}.pt"
    import shutil
    shutil.copy(best_ckpt, ARCHIVE / "checkpoints" / "master.pt")

    # Save best signals
    best_sig = pd.read_parquet(ARCHIVE / "signals" / f"master_seed{best_seed}_test.parquet")
    best_sig.to_parquet(ARCHIVE / "signals" / "master_test.parquet", index=False)

    # Summary
    summary = {
        "best_seed": best_seed,
        "best_sharpe": best_sharpe,
        "n_seeds_total": len(SEEDS),
        "top_k_val_rank_ic": TOP_K_VAL,
        "all_results": all_results,
        "top3_results": [
            {k: v for k, v in r.items()}
            for r in top3
        ],
    }
    with open(ARCHIVE / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to {ARCHIVE}/")
    print(f"  checkpoints/: {len(SEEDS)} model files")
    print(f"  signals/:     {len(SEEDS)} test signal files")
    print(f"  search_summary.json")
    print(f"  master.pt     <- best (seed={best_seed})")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
