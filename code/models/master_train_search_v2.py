"""
MASTER 训练搜索 v2: 全量回测 + 更多种子 + 超参微调

改进:
  1. 训练 N 个种子, 全部回测 (不依赖 val_rank_ic 筛选)
  2. 对 top-3 (按夏普) 做超参微调
  3. 最终输出最优模型

Usage:
  python -m code.models.master_train_search_v2
"""
import json
import time
import math
import copy
import shutil
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
from code.backtest.engine import backtest, compute_metrics

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
ARCHIVE = OUTPUT / "v3_multi_train"
for d in [ARCHIVE, ARCHIVE / "checkpoints", ARCHIVE / "signals"]:
    d.mkdir(parents=True, exist_ok=True)

# Phase 1: 20 seeds (已有的 10 个 + 新增 10 个)
SEEDS_ALL = [42, 1337, 2024, 88, 777, 1024, 2048, 4096, 99, 314,
             512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

# Phase 2: 超参微调 (对 top-3 种子做)
HP_GRID = [
    {"lr": 5e-4, "dropout": 0.20, "alpha": 0.60, "H": 64},   # baseline
    {"lr": 3e-4, "dropout": 0.20, "alpha": 0.60, "H": 64},   # lower lr
    {"lr": 7e-4, "dropout": 0.20, "alpha": 0.60, "H": 64},   # higher lr
    {"lr": 5e-4, "dropout": 0.15, "alpha": 0.60, "H": 64},   # less dropout
    {"lr": 5e-4, "dropout": 0.20, "alpha": 0.50, "H": 64},   # more margin
    {"lr": 5e-4, "dropout": 0.20, "alpha": 0.70, "H": 64},   # more IC
    {"lr": 5e-4, "dropout": 0.20, "alpha": 0.60, "H": 80},   # wider
]


def train_one(seed, X, X_w, y, trade_dates, ts_codes, market_X, market_date_idx,
              full_ds, df_full, device, hp_overrides=None):
    """训练一个模型, 返回 (result_dict, test_df)"""
    hp = {"lr": 5e-4, "dropout": 0.2, "alpha": 0.6, "H": 64, "wd": 1e-3}
    if hp_overrides:
        hp.update(hp_overrides)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dates = [d for d in full_ds.dates if d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]

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
        sub.date_to_endpoints = full_ds.date_to_endpoints
        return sub

    train_loader = DataLoader(make_sub(train_dates), batch_size=1, shuffle=True,
                              num_workers=0, collate_fn=collate_single)
    valid_loader = DataLoader(make_sub(valid_dates), batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_single)
    test_loader = DataLoader(make_sub(test_dates), batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collate_single)

    F_market = market_X.shape[1]
    F_weight = N_WEIGHT if X_w is not None else 0
    model = MASTER(F_stock=N_FEAT, F_market=F_market, H=hp["H"], T=T,
                   nhead=4, dropout=hp["dropout"], n_intra_layers=2, n_inter_layers=1,
                   F_weight=F_weight).to(device)
    nparams = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    hp_tag = f"seed{seed}"
    if hp_overrides:
        hp_tag += "_" + "_".join(f"{k}{v}" for k, v in sorted(hp_overrides.items())
                                  if k not in ("wd",))
    ckpt = ARCHIVE / "checkpoints" / f"master_{hp_tag}.pt"

    best_val_rank_ic = -1.0
    best_epoch = 0
    patience = 0
    patience_max = 6

    for epoch in range(20):
        model.train()
        losses = []
        for X_d, m_d, y_d, _, _, X_w_d in train_loader:
            X_d = X_d.to(device, non_blocking=True)
            m_d = m_d.to(device, non_blocking=True)
            y_d = y_d.to(device, non_blocking=True)
            X_w_d = X_w_d.to(device, non_blocking=True) if X_w_d.numel() > 0 else None
            loss = combined_loss(model(X_d, m_d, X_w_d), y_d, alpha=hp["alpha"])
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
    test_df.to_parquet(ARCHIVE / "signals" / f"master_{hp_tag}_test.parquet", index=False)

    return {
        "tag": hp_tag,
        "seed": seed,
        "hp": hp,
        "best_epoch": best_epoch,
        "val_rank_ic": float(best_val_rank_ic),
        "test_ic": float(test_ic),
        "test_rank_ic": float(test_rank_ic),
        "test_rank_ic_std": float(test_rank_ic_std),
        "test_rank_icir_annual": float(test_rank_ic / (test_rank_ic_std + 1e-8) * math.sqrt(252)),
        "n_params": nparams,
    }, test_df


def backtest_one(sig, panel):
    nav, daily_ret, _ = backtest(sig, panel, n=10, k=2)
    m = compute_metrics(daily_ret, nav)
    return {"sharpe": m["sharpe"], "total_return": m["total_return"],
            "max_drawdown": m["max_drawdown"], "annual_return": m["annual_return"]}


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
    print(f"  train={len(train_dates)}  valid={len(valid_dates)}  test={len(test_dates)}")

    # Load panel once for all backtests
    panel = pd.read_parquet(
        CACHE / "panel.parquet",
        columns=["trade_date", "ts_code", "open", "high", "low", "close",
                 "pct_chg", "is_st"],
    )
    panel = panel[panel["trade_date"] >= 20231201]

    # ================================================================
    # Phase 1: Train all seeds (skip already-trained ones)
    # ================================================================
    existing_tags = {f.stem.replace("master_", "").replace("_test", "")
                     for f in (ARCHIVE / "signals").glob("master_*_test.parquet")}
    # Only match pure seed tags (no hp overrides)
    existing_seeds = set()
    for tag in existing_tags:
        try:
            existing_seeds.add(int(tag.replace("seed", "").split("_")[0]
                                   .replace("seed", "")))
        except ValueError:
            pass

    new_seeds = [s for s in SEEDS_ALL if s not in existing_seeds]
    print(f"\n{'='*60}")
    print(f"Phase 1: Training {len(new_seeds)} new seeds (skip {len(existing_seeds)} existing)")
    print(f"New seeds: {new_seeds}")

    all_results = []
    for seed in new_seeds:
        t_seed = time.time()
        print(f"\n  Training seed={seed} ...")
        r, _ = train_one(seed, X, X_w, y, trade_dates, ts_codes_int,
                         market_X, market_date_idx, full_ds, df_full, device)
        r["train_time_s"] = round(time.time() - t_seed, 1)
        all_results.append(r)
        print(f"    val_rank_ic={r['val_rank_ic']:.4f}  "
              f"test_rank_ic={r['test_rank_ic']:.4f}  "
              f"test_icir={r['test_rank_icir_annual']:.2f}  "
              f"time={r['train_time_s']}s")

    # Load existing results from checkpoints
    for seed in existing_seeds:
        sig_path = ARCHIVE / "signals" / f"master_seed{seed}_test.parquet"
        if not sig_path.exists():
            continue
        sig = pd.read_parquet(sig_path)
        # Extract basic metrics
        ics, rank_ics = [], []
        for _, day in sig.groupby("trade_date"):
            if len(day) >= 30 and day["score"].std() > 0:
                ics.append(day["score"].corr(day["label"]))
                rank_ics.append(day["score"].rank().corr(day["label"].rank()))
        r = {
            "tag": f"seed{seed}",
            "seed": seed,
            "hp": {"lr": 5e-4, "dropout": 0.2, "alpha": 0.6, "H": 64, "wd": 1e-3},
            "test_ic": float(np.mean(ics)) if ics else 0,
            "test_rank_ic": float(np.mean(rank_ics)) if rank_ics else 0,
            "test_rank_ic_std": float(np.std(rank_ics)) if rank_ics else 0,
        }
        r["test_rank_icir_annual"] = float(r["test_rank_ic"] / (r["test_rank_ic_std"] + 1e-8) * math.sqrt(252))
        all_results.append(r)

    # ================================================================
    # Phase 2: Backtest ALL seeds
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Backtesting ALL {len(all_results)} models ...")

    for r in all_results:
        sig_path = ARCHIVE / "signals" / f"master_{r['tag']}_test.parquet"
        if not sig_path.exists():
            print(f"  {r['tag']}: signals missing, skip")
            continue
        sig = pd.read_parquet(sig_path)
        bt = backtest_one(sig, panel)
        r.update(bt)
        print(f"  {r['tag']:30s}  sharpe={bt['sharpe']:.4f}  "
              f"total_ret={bt['total_return']:.4f}  mdd={bt['max_drawdown']:.4f}")

    # Sort by Sharpe
    all_results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

    print(f"\n{'='*60}")
    print("Full ranking (by Sharpe):")
    for i, r in enumerate(all_results):
        s = r.get("sharpe", 0)
        ret = r.get("total_return", 0)
        mdd = r.get("max_drawdown", 0)
        print(f"  {i+1:2d}. {r['tag']:30s}  sharpe={s:.4f}  ret={ret:.4f}  mdd={mdd:.4f}")

    # ================================================================
    # Phase 3: Hyperparameter finetuning on top-3
    # ================================================================
    top3 = all_results[:3]
    print(f"\n{'='*60}")
    print(f"Phase 3: Hyperparameter finetuning on top-3 seeds")
    print(f"Top-3: {[r['seed'] for r in top3]}")

    hp_results = []
    for rank_i, base_r in enumerate(top3):
        seed = base_r["seed"]
        print(f"\n  --- Seed {seed} (rank {rank_i+1}, sharpe={base_r.get('sharpe', 0):.4f}) ---")
        for hp_i, hp_override in enumerate(HP_GRID):
            # Skip baseline (already trained)
            if hp_override == {"lr": 5e-4, "dropout": 0.20, "alpha": 0.60, "H": 64}:
                continue
            t_hp = time.time()
            short = f"s{seed}_lr{hp_override['lr']:.0e}_d{hp_override['dropout']}_a{hp_override['alpha']}_H{hp_override['H']}"
            short = short.replace("e-0", "e-").replace(".0", "")
            print(f"    hp: {short} ...")
            r, test_df = train_one(seed, X, X_w, y, trade_dates, ts_codes_int,
                                   market_X, market_date_idx, full_ds, df_full, device,
                                   hp_overrides=hp_override)
            bt = backtest_one(test_df, panel)
            r.update(bt)
            r["train_time_s"] = round(time.time() - t_hp, 1)
            hp_results.append(r)
            print(f"      sharpe={bt['sharpe']:.4f}  ret={bt['total_return']:.4f}  "
                  f"val_rank_ic={r['val_rank_ic']:.4f}  time={r['train_time_s']}s")

    # Merge all results
    all_with_hp = all_results + hp_results
    all_with_hp.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

    # ================================================================
    # Phase 4: Save final best
    # ================================================================
    best = all_with_hp[0]
    print(f"\n{'='*60}")
    print(f"FINAL BEST: {best['tag']}")
    print(f"  sharpe={best['sharpe']:.4f}  total_ret={best['total_return']:.4f}  "
          f"mdd={best['max_drawdown']:.4f}")
    print(f"  val_rank_ic={best.get('val_rank_ic', '?'):.4f}" if isinstance(best.get('val_rank_ic'), float) else f"  val_rank_ic={best.get('val_rank_ic', '?')}")
    if "hp" in best:
        print(f"  hp={best['hp']}")

    # Copy best to archive root
    best_ckpt = ARCHIVE / "checkpoints" / f"master_{best['tag']}.pt"
    best_sig = ARCHIVE / "signals" / f"master_{best['tag']}_test.parquet"
    if best_ckpt.exists():
        shutil.copy(best_ckpt, ARCHIVE / "checkpoints" / "master.pt")
    if best_sig.exists():
        shutil.copy(best_sig, ARCHIVE / "signals" / "master_test.parquet")

    # Full summary
    summary = {
        "best": best,
        "total_models": len(all_with_hp),
        "full_ranking": all_with_hp,
    }
    with open(ARCHIVE / "search_summary_v2.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved to {ARCHIVE}/")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
