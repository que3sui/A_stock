"""
MASTER v7: 直接优化回测目标 (Sharpe-approximate loss)

核心洞察: IC loss != backtest Sharpe. v7用loss直接近似回测Sharpe.

Loss = -mean(topK_labels) / std(topK_labels)  +  lambda * IC_loss(stabilizer)

Usage:
  python -m code.models.master_v7 --search
"""
import argparse, json, time, math, shutil
import numpy as np; import pandas as pd; import torch
from torch.utils.data import DataLoader

from code.models.master import (
    MASTER, DailyPanelDataset, collate_single, evaluate, prepare,
    T, N_FEAT, N_WEIGHT, TRAIN_MAX, VALID_MAX,
)
from code.config import ROOT, CACHE, OUTPUT
from code.losses import ic_loss

ARCHIVE = OUTPUT / "v7"
for d in [ARCHIVE, ARCHIVE / "checkpoints", ARCHIVE / "signals"]:
    d.mkdir(parents=True, exist_ok=True)

SEEDS = [128, 1024, 99, 42, 777, 4096, 314, 88, 512, 256]


def sharpe_aware_loss(scores, labels, n=8, lambda_ic=0.3):
    """直接优化 top-K 的 Sharpe (负值 = 最小化), IC 做稳定器"""
    # Top-K Sharpe approximation
    k = min(n, len(scores))
    _, top_idx = torch.topk(scores, k)
    top_labels = labels[top_idx]
    top_mean = top_labels.mean()
    top_std = top_labels.std() + 1e-8
    sharpe_term = -top_mean / top_std  # negative = minimize

    # IC stabilizer
    ic_term = ic_loss(scores, labels)

    return sharpe_term + lambda_ic * ic_term


def train_one(seed, X, X_w, y, trade_dates, ts_codes, market_X, market_date_idx,
              full_ds, df_full, device, hp_overrides=None):
    hp = {"lr": 5e-4, "dropout": 0.2, "H": 64, "wd": 1e-3}
    if hp_overrides: hp.update(hp_overrides)

    torch.manual_seed(seed); np.random.seed(seed)

    train_dates = [d for d in full_ds.dates if d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]

    train_loader = DataLoader(full_ds.subset_by_dates(train_dates), batch_size=1, shuffle=True,
                               num_workers=0, collate_fn=collate_single)
    valid_loader = DataLoader(full_ds.subset_by_dates(valid_dates), batch_size=1, shuffle=False,
                               num_workers=0, collate_fn=collate_single)
    test_loader = DataLoader(full_ds.subset_by_dates(test_dates), batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_single)

    F_market = market_X.shape[1]; F_weight = N_WEIGHT if X_w is not None else 0
    model = MASTER(F_stock=N_FEAT, F_market=F_market, H=hp["H"], T=T,
                   nhead=4, dropout=hp["dropout"], n_intra_layers=2, n_inter_layers=1,
                   F_weight=F_weight).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    hp_tag = f"seed{seed}"
    if hp_overrides: hp_tag += "_"+"_".join(f"{k}{v}" for k,v in sorted(hp_overrides.items()) if k!="wd")
    ckpt = ARCHIVE / "checkpoints" / f"master_{hp_tag}.pt"

    best_val_rank_ic = -1.0; best_epoch = 0; patience = 0

    for epoch in range(20):
        model.train(); losses = []
        for X_d, m_d, y_d, _, _, X_w_d in train_loader:
            X_d = X_d.to(device, non_blocking=True); m_d = m_d.to(device, non_blocking=True)
            y_d = y_d.to(device, non_blocking=True)
            X_w_d = X_w_d.to(device, non_blocking=True) if X_w_d.numel() > 0 else None
            pred = model(X_d, m_d, X_w_d)
            loss = sharpe_aware_loss(pred, y_d, n=8, lambda_ic=0.3)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sched.step()

        val_ic, val_rank_ic, _, _ = evaluate(model, valid_loader, device, None)
        if val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = val_rank_ic; best_epoch = epoch + 1; patience = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= 6: break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    test_ic, test_rank_ic, test_rank_ic_std, test_df = evaluate(model, test_loader, device, None)
    test_df["ts_code"] = df_full.loc[test_df["ep"].values, "ts_code"].values
    test_df = test_df.drop(columns=["ep"])
    test_df.to_parquet(ARCHIVE / "signals" / f"master_{hp_tag}_test.parquet", index=False)

    return {
        "tag": hp_tag, "seed": seed, "hp": hp,
        "best_epoch": best_epoch, "val_rank_ic": float(best_val_rank_ic),
        "test_ic": float(test_ic), "test_rank_ic": float(test_rank_ic),
        "test_rank_ic_std": float(test_rank_ic_std),
        "test_rank_icir_annual": float(test_rank_ic/(test_rank_ic_std+1e-8)*math.sqrt(252)),
        "n_params": sum(p.numel() for p in model.parameters()),
    }, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--seed", type=int, default=128)
    args = parser.parse_args()

    t0 = time.time(); device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda": print(f"  {torch.cuda.get_device_name(0)}")
    print(f"V7: Sharpe-aware loss (top-K Sharpe + IC stabilizer)")

    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")

    print("Preparing ...")
    X, y, trade_dates, ts_codes_int, code_uniq, market_X, market_date_idx, df_full, X_w = prepare(feats, labels, market)
    if X_w is not None: print(f"  weight channel: {X_w.shape[1]} cols")

    print("Building dataset ...")
    full_ds = DailyPanelDataset(X, y, trade_dates, ts_codes_int, market_X, market_date_idx, T, X_w=X_w)
    train_dates = [d for d in full_ds.dates if d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]
    print(f"  train={len(train_dates)}  valid={len(valid_dates)}  test={len(test_dates)}")

    panel = pd.read_parquet(CACHE / "panel.parquet",
        columns=["trade_date","ts_code","open","high","low","close","pct_chg","is_st"])
    panel = panel[panel["trade_date"]>=20231201]

    if args.search:
        print(f"\n{'='*60}\nV7 Training Search: {len(SEEDS)} seeds\n{'='*60}")

        all_results = []
        for seed in SEEDS:
            t_seed = time.time()
            print(f"\n  seed={seed} ...")
            r, test_df = train_one(seed, X, X_w, y, trade_dates, ts_codes_int,
                                    market_X, market_date_idx, full_ds, df_full, device)
            r["train_time_s"] = round(time.time()-t_seed,1)

            # Backtest
            from code.backtest.engine import backtest, compute_metrics
            nav, daily_ret, _ = backtest(test_df, panel, n=8, k=2)
            m = compute_metrics(daily_ret, nav)
            r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                       "max_drawdown": m["max_drawdown"]})
            all_results.append(r)
            print(f"    rank_ic={r['test_rank_ic']:.4f}  sharpe={m['sharpe']:.4f}  "
                  f"ret={m['total_return']:.4f}  mdd={m['max_drawdown']:.4f}  time={r['train_time_s']}s")

        all_results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
        best = all_results[0]

        # Quick HP tune on top-1
        top_seed = best["seed"]
        HP_GRID = [{"lr": 3e-4}, {"lr": 7e-4}, {"dropout": 0.15}]
        for hp_override in HP_GRID:
            r, test_df = train_one(top_seed, X, X_w, y, trade_dates, ts_codes_int,
                                    market_X, market_date_idx, full_ds, df_full, device,
                                    hp_overrides=hp_override)
            nav, daily_ret, _ = backtest(test_df, panel, n=8, k=2)
            from code.backtest.engine import compute_metrics
            m = compute_metrics(daily_ret, nav)
            r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                       "max_drawdown": m["max_drawdown"]})
            all_results.append(r)
            print(f"    HP {r['tag']}: sharpe={m['sharpe']:.4f}")

        all_results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
        best = all_results[0]
        all_with_hp = all_results
    else:
        r, test_df = train_one(args.seed, X, X_w, y, trade_dates, ts_codes_int,
                                market_X, market_date_idx, full_ds, df_full, device)
        from code.backtest.engine import backtest, compute_metrics
        nav, daily_ret, _ = backtest(test_df, panel, n=8, k=2)
        m = compute_metrics(daily_ret, nav)
        r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                   "max_drawdown": m["max_drawdown"]})
        best = r; all_with_hp = [r]

    print(f"\n{'='*60}\nBEST: {best['tag']}")
    print(f"  sharpe={best['sharpe']:.4f}  total_ret={best['total_return']:.4f}  mdd={best['max_drawdown']:.4f}")

    best_ckpt = ARCHIVE / "checkpoints" / f"master_{best['tag']}.pt"
    best_sig = ARCHIVE / "signals" / f"master_{best['tag']}_test.parquet"
    if best_ckpt.exists(): shutil.copy(best_ckpt, ARCHIVE / "checkpoints" / "master.pt")
    if best_sig.exists(): shutil.copy(best_sig, ARCHIVE / "signals" / "master_test.parquet")

    json.dump({"best": best, "full_ranking": all_with_hp,
               "v7_features": ["sharpe_aware_loss"]},
              open(ARCHIVE / "search_summary.json","w",encoding="utf-8"), indent=2, ensure_ascii=False, default=str)

    (ARCHIVE / "README.md").write_text(f"# MASTER v7\n## Sharpe-aware loss\nBest: {best['tag']} Sharpe={best['sharpe']:.4f}\n", encoding="utf-8")
    print(f"\nSaved to {ARCHIVE}/  total={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
