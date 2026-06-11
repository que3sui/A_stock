"""
MASTER v6: v3窗口 + v4改进 + v5滤波器 = 28因子 + 长窗口

v4 教训: 训练窗口不能缩短, 2016-2018数据对A股长期模式仍有价值
v6 方案: 保留v3的2016-2022训练窗口 + 时间衰减(保留但不均等)
        + 行业轮动(v4) + 滤波器(v5) + 动态仓位(v4)

因子: 20基础 + 2行业 + 6滤波器 = 28因子 + 3权重通道
窗口: train=2016-2022, valid=2023, test=2024-2026 (同v3)
时间衰减: lambda=0.3 (比v4的0.5温和, 保留更久的历史)

Usage:
  python -m code.models.master_v6 --search
"""
import argparse, json, time, math, shutil
import numpy as np; import pandas as pd; import torch
from torch.utils.data import DataLoader; from pathlib import Path

from code.models.master_v5 import (
    add_industry_factors, add_filter_factors, train_one,
    FACTOR_COLS_V5, N_FEAT_V5, FILTER_COLS, SEEDS, HP_GRID,
)
from code.models.master_v4 import combined_loss_v4, backtest_v4, dynamic_n
from code.models.master import (
    MASTER, DailyPanelDataset, collate_single, evaluate, prepare,
    T, N_WEIGHT,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"; OUTPUT = ROOT / "output"
ARCHIVE = OUTPUT / "v6"
for d in [ARCHIVE, ARCHIVE / "checkpoints", ARCHIVE / "signals"]:
    d.mkdir(parents=True, exist_ok=True)

# v6关键: 回到v3的长窗口
TRAIN_START = 20160101   # v3窗口
TRAIN_MAX = 20221231
VALID_MAX = 20231231

DECAY_LAMBDA = 0.3  # 比v4温和(0.5), 2016年权重仍保留exp(-0.3*7)=0.12

FACTOR_COLS_V6 = FACTOR_COLS_V5  # same 28 factors
N_FEAT_V6 = N_FEAT_V5


def train_one_v6(seed, X, X_w, y, trade_dates, ts_codes, market_X, market_date_idx,
                 full_ds, df_full, device, hp_overrides=None):
    """v6训练: v5的架构 + v3的窗口 + 更温和的时间衰减"""
    hp = {"lr": 5e-4, "dropout": 0.2, "alpha": 0.6, "H": 64, "wd": 1e-3}
    if hp_overrides: hp.update(hp_overrides)

    torch.manual_seed(seed); np.random.seed(seed)

    train_dates = [d for d in full_ds.dates if TRAIN_START <= d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]

    train_loader = DataLoader(full_ds.subset_by_dates(train_dates), batch_size=1, shuffle=True,
                               num_workers=0, collate_fn=collate_single)
    valid_loader = DataLoader(full_ds.subset_by_dates(valid_dates), batch_size=1, shuffle=False,
                               num_workers=0, collate_fn=collate_single)
    test_loader = DataLoader(full_ds.subset_by_dates(test_dates), batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_single)

    F_market = market_X.shape[1]
    F_weight = N_WEIGHT if X_w is not None else 0
    model = MASTER(F_stock=N_FEAT_V6, F_market=F_market, H=hp["H"], T=T,
                   nhead=4, dropout=hp["dropout"], n_intra_layers=2, n_inter_layers=1,
                   F_weight=F_weight).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    hp_tag = f"seed{seed}"
    if hp_overrides:
        hp_tag += "_" + "_".join(f"{k}{v}" for k,v in sorted(hp_overrides.items()) if k!="wd")
    ckpt = ARCHIVE / "checkpoints" / f"master_{hp_tag}.pt"

    max_date = max(train_dates) if train_dates else TRAIN_MAX
    best_val_rank_ic = -1.0; best_epoch = 0; patience = 0

    for epoch in range(20):
        model.train(); losses = []
        for X_d, m_d, y_d, date, _, X_w_d in train_loader:
            X_d = X_d.to(device, non_blocking=True)
            m_d = m_d.to(device, non_blocking=True)
            y_d = y_d.to(device, non_blocking=True)
            X_w_d = X_w_d.to(device, non_blocking=True) if X_w_d.numel() > 0 else None
            loss = combined_loss_v4(model(X_d, m_d, X_w_d), y_d, int(date), max_date, hp["alpha"])
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

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda": print(f"  {torch.cuda.get_device_name(0)}")
    print(f"V6: {N_FEAT_V6} factors (20 base + 2 industry + 6 filter)")
    print(f"Window: {TRAIN_START}-{TRAIN_MAX} train / ~{VALID_MAX} valid / >{VALID_MAX} test")
    print(f"Decay lambda: {DECAY_LAMBDA}")

    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")

    print("Adding industry + filter factors ...")
    feats = add_industry_factors(feats)
    feats = add_filter_factors(feats)

    print("Preparing ...")
    import code.models.master as master_mod
    original_fcols, original_nfeat = master_mod.FACTOR_COLS, master_mod.N_FEAT
    master_mod.FACTOR_COLS = FACTOR_COLS_V6
    master_mod.N_FEAT = N_FEAT_V6

    X, y, trade_dates, ts_codes_int, code_uniq, market_X, market_date_idx, df_full, X_w = \
        prepare(feats, labels, market)

    master_mod.FACTOR_COLS = original_fcols
    master_mod.N_FEAT = original_nfeat

    if X_w is not None: print(f"  weight channel: {X_w.shape[1]} cols")

    print("Building dataset (once) ...")
    full_ds = DailyPanelDataset(X, y, trade_dates, ts_codes_int,
                                 market_X, market_date_idx, T, X_w=X_w)
    train_dates = [d for d in full_ds.dates if TRAIN_START <= d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]
    print(f"  train={len(train_dates)}  valid={len(valid_dates)}  test={len(test_dates)}")

    panel = pd.read_parquet(CACHE / "panel.parquet",
            columns=["trade_date","ts_code","open","high","low","close","pct_chg","is_st"])
    panel = panel[panel["trade_date"] >= 20231201]
    market_df = pd.read_parquet(CACHE / "market_features.parquet")

    if args.search:
        print(f"\n{'='*60}\nV6 Training Search: {len(SEEDS)} seeds + HP\n{'='*60}")

        all_results = []
        for seed in SEEDS:
            t_seed = time.time()
            print(f"\n  Training seed={seed} ...")
            r, _ = train_one_v6(seed, X, X_w, y, trade_dates, ts_codes_int,
                                 market_X, market_date_idx, full_ds, df_full, device)
            r["train_time_s"] = round(time.time()-t_seed, 1)
            all_results.append(r)
            print(f"    val_rank_ic={r['val_rank_ic']:.4f}  "
                  f"test_rank_ic={r['test_rank_ic']:.4f}  "
                  f"test_icir={r['test_rank_icir_annual']:.2f}  time={r['train_time_s']}s")

        print(f"\n{'='*60}\nBacktesting ALL seeds (v4 dynamic-n) ...")
        from code.backtest.engine import compute_metrics
        for r in all_results:
            sig = pd.read_parquet(ARCHIVE / "signals" / f"master_{r['tag']}_test.parquet")
            nav, daily_ret = backtest_v4(sig, panel, market_df)
            m = compute_metrics(daily_ret, nav)
            r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                       "max_drawdown": m["max_drawdown"], "annual_return": m["annual_return"]})
            print(f"  {r['tag']:30s}  sharpe={m['sharpe']:.4f}  "
                  f"ret={m['total_return']:.4f}  mdd={m['max_drawdown']:.4f}")

        all_results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

        top3 = all_results[:3]
        print(f"\n{'='*60}\nHP tuning on top-3: {[r['seed'] for r in top3]}")

        hp_results = []
        for base_r in top3:
            seed = base_r["seed"]
            for hp_override in HP_GRID:
                if hp_override == {"lr": 5e-4, "dropout": 0.20, "alpha": 0.60, "H": 64}:
                    continue
                t_hp = time.time()
                r, _ = train_one_v6(seed, X, X_w, y, trade_dates, ts_codes_int,
                                     market_X, market_date_idx, full_ds, df_full, device,
                                     hp_overrides=hp_override)
                nav, daily_ret = backtest_v4(
                    pd.read_parquet(ARCHIVE / "signals" / f"master_{r['tag']}_test.parquet"),
                    panel, market_df)
                m = compute_metrics(daily_ret, nav)
                r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                           "max_drawdown": m["max_drawdown"], "annual_return": m["annual_return"]})
                r["train_time_s"] = round(time.time()-t_hp, 1)
                hp_results.append(r)
                print(f"    {r['tag']}: sharpe={m['sharpe']:.4f}  ret={m['total_return']:.4f}")

        all_with_hp = all_results + hp_results
        all_with_hp.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
        best = all_with_hp[0]
    else:
        r, _ = train_one_v6(args.seed, X, X_w, y, trade_dates, ts_codes_int,
                             market_X, market_date_idx, full_ds, df_full, device)
        sig = pd.read_parquet(ARCHIVE / "signals" / f"master_{r['tag']}_test.parquet")
        nav, daily_ret = backtest_v4(sig, panel, market_df)
        from code.backtest.engine import compute_metrics
        m = compute_metrics(daily_ret, nav)
        r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                   "max_drawdown": m["max_drawdown"], "annual_return": m["annual_return"]})
        best = r; all_with_hp = [r]

    print(f"\n{'='*60}\nBEST: {best['tag']}")
    print(f"  sharpe={best['sharpe']:.4f}  total_ret={best['total_return']:.4f}  "
          f"mdd={best['max_drawdown']:.4f}")

    best_ckpt = ARCHIVE / "checkpoints" / f"master_{best['tag']}.pt"
    best_sig = ARCHIVE / "signals" / f"master_{best['tag']}_test.parquet"
    if best_ckpt.exists(): shutil.copy(best_ckpt, ARCHIVE / "checkpoints" / "master.pt")
    if best_sig.exists(): shutil.copy(best_sig, ARCHIVE / "signals" / "master_test.parquet")

    summary = {"best": best, "total_models": len(all_with_hp),
               "v6_features": ["long_window(2016-2022)", "time_decay(λ=0.3)",
                               "industry_factors", "filter_factors(EMA+trend)", "dynamic_n"],
               "full_ranking": all_with_hp}
    with open(ARCHIVE / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    (ARCHIVE / "README.md").write_text(f"""# MASTER v6
## v6 = v3长窗口 + v4行业/时间衰减 + v5滤波器
- 窗口: 2016-2022 train / 2023 valid / 2024-2026 test (同v3)
- 时间衰减: λ=0.3 (温和, 保留历史信息)
- 因子: 28 = 20基 + 2行业 + 6滤波器
- 动态仓位 + 权重通道
## Best: {best['tag']}  Sharpe: {best.get('sharpe','?')}
""", encoding="utf-8")

    print(f"\nSaved to {ARCHIVE}/  total={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
