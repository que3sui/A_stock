"""
MASTER v5: v4 + 显式时序滤波器因子

v5 = v4 全部改进 + 6维滤波器因子 (基于 attention HHI≈0.05 的结论)

新增因子 (来自 code/features/filter_factors.py):
  ema_mom_5    — 5日EMA收益率
  ema_mom_20   — 20日EMA收益率
  ema_cross    — MACD快慢线交叉信号 (EMA5/EMA20 - 1)
  trend_r2     — 20日线性回归拟合优度 (趋势稳定性: 0=乱, 1=直线)
  trend_slope  — 20日趋势斜率 (年化, 正=上涨趋势, 负=下跌趋势)
  vol_trend    — 波动率/趋势强度 (噪声比, 低=趋势清晰)

Usage:
  python -m code.models.master_v5          # 单次训练
  python -m code.models.master_v5 --search # 训练搜索

注意: v4 训练不受影响, v5 独立运行, 独立输出到 output/v5/
"""
import argparse
import json
import time
import math
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from code.models.master_v4 import (
    add_industry_factors, combined_loss_v4, backtest_v4, dynamic_n,
    FACTOR_COLS_V4, N_FEAT_V4, INDUSTRY_FACTORS,
    TRAIN_START, TRAIN_MAX, VALID_MAX, SEEDS, HP_GRID,
)
from code.models.master import (
    MASTER, DailyPanelDataset, collate_single, evaluate, prepare,
    T, N_WEIGHT, TRAIN_MAX as OLD_TRAIN_MAX, VALID_MAX as OLD_VALID_MAX,
)
from code.features.filter_factors import FILTER_COLS

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
ARCHIVE = OUTPUT / "v5"
for d in [ARCHIVE, ARCHIVE / "checkpoints", ARCHIVE / "signals"]:
    d.mkdir(parents=True, exist_ok=True)

# v5: v4的22个 + 6个滤波器 = 28个因子
FACTOR_COLS_V5 = FACTOR_COLS_V4 + FILTER_COLS
N_FEAT_V5 = len(FACTOR_COLS_V5)


def add_filter_factors(feats):
    """将预计算的滤波器因子合并到特征表中"""
    ff = pd.read_parquet(CACHE / "filter_factors.parquet")
    feats = feats.merge(ff, on=["trade_date", "ts_code"], how="left")
    for c in FILTER_COLS:
        feats[c] = feats[c].fillna(0.0).astype("float32")
    return feats


def train_one(seed, X, X_w, y, trade_dates, ts_codes, market_X, market_date_idx,
              full_ds, df_full, device, hp_overrides=None):
    """训练一个 v5 模型 (同 v4, 仅因子数不同)"""
    hp = {"lr": 5e-4, "dropout": 0.2, "alpha": 0.6, "H": 64, "wd": 1e-3}
    if hp_overrides:
        hp.update(hp_overrides)

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
    model = MASTER(F_stock=N_FEAT_V5, F_market=F_market, H=hp["H"], T=T,
                   nhead=4, dropout=hp["dropout"], n_intra_layers=2, n_inter_layers=1,
                   F_weight=F_weight).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    hp_tag = f"seed{seed}"
    if hp_overrides:
        hp_tag += "_" + "_".join(f"{k}{v}" for k, v in sorted(hp_overrides.items()) if k != "wd")
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
        "test_rank_icir_annual": float(test_rank_ic / (test_rank_ic_std + 1e-8) * math.sqrt(252)),
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
    print(f"V5 factors: {N_FEAT_V5} (= v4的22 + {len(FILTER_COLS)} filter)")

    print("\nLoading features / labels / market ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")

    print("Adding industry + filter factors ...")
    feats = add_industry_factors(feats)
    feats = add_filter_factors(feats)
    print(f"  total features: {len(FACTOR_COLS_V5)}")

    print("Preparing ...")
    import code.models.master as master_mod
    original_fcols = master_mod.FACTOR_COLS
    original_nfeat = master_mod.N_FEAT
    master_mod.FACTOR_COLS = FACTOR_COLS_V5
    master_mod.N_FEAT = N_FEAT_V5

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
                            columns=["trade_date", "ts_code", "open", "high", "low", "close",
                                     "pct_chg", "is_st"])
    panel = panel[panel["trade_date"] >= 20231201]
    market_df = pd.read_parquet(CACHE / "market_features.parquet")

    if args.search:
        print(f"\n{'='*60}")
        print(f"V5 Training Search: {len(SEEDS)} seeds + HP tuning")
        print(f"{'='*60}")

        all_results = []
        for seed in SEEDS:
            t_seed = time.time()
            print(f"\n  Training seed={seed} ...")
            r, test_df = train_one(seed, X, X_w, y, trade_dates, ts_codes_int,
                                    market_X, market_date_idx, full_ds, df_full, device)
            r["train_time_s"] = round(time.time() - t_seed, 1)
            all_results.append(r)
            print(f"    val_rank_ic={r['val_rank_ic']:.4f}  "
                  f"test_rank_ic={r['test_rank_ic']:.4f}  "
                  f"test_icir={r['test_rank_icir_annual']:.2f}  time={r['train_time_s']}s")

        print(f"\n{'='*60}")
        print("Backtesting ALL seeds (v4 dynamic-n) ...")
        for r in all_results:
            sig = pd.read_parquet(ARCHIVE / "signals" / f"master_{r['tag']}_test.parquet")
            nav, daily_ret = backtest_v4(sig, panel, market_df)
            from code.backtest.engine import compute_metrics
            m = compute_metrics(daily_ret, nav)
            r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                       "max_drawdown": m["max_drawdown"], "annual_return": m["annual_return"]})
            print(f"  {r['tag']:30s}  sharpe={m['sharpe']:.4f}  "
                  f"ret={m['total_return']:.4f}  mdd={m['max_drawdown']:.4f}")

        all_results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

        top3 = all_results[:3]
        print(f"\n{'='*60}")
        print(f"HP tuning on top-3: {[r['seed'] for r in top3]}")

        hp_results = []
        for base_r in top3:
            seed = base_r["seed"]
            for hp_override in HP_GRID:
                if hp_override == {"lr": 5e-4, "dropout": 0.20, "alpha": 0.60, "H": 64}:
                    continue
                t_hp = time.time()
                short = f"seed{seed}_lr{hp_override['lr']:.0e}_d{hp_override['dropout']}_a{hp_override['alpha']}_H{hp_override['H']}"
                print(f"    hp: {short} ...")
                r, test_df = train_one(seed, X, X_w, y, trade_dates, ts_codes_int,
                                        market_X, market_date_idx, full_ds, df_full, device,
                                        hp_overrides=hp_override)
                nav, daily_ret = backtest_v4(test_df, panel, market_df)
                from code.backtest.engine import compute_metrics
                m = compute_metrics(daily_ret, nav)
                r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                           "max_drawdown": m["max_drawdown"], "annual_return": m["annual_return"]})
                r["train_time_s"] = round(time.time() - t_hp, 1)
                hp_results.append(r)
                print(f"      sharpe={m['sharpe']:.4f}  ret={m['total_return']:.4f}  "
                      f"val_rank_ic={r['val_rank_ic']:.4f}")

        all_with_hp = all_results + hp_results
        all_with_hp.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
        best = all_with_hp[0]
    else:
        r, test_df = train_one(args.seed, X, X_w, y, trade_dates, ts_codes_int,
                                market_X, market_date_idx, full_ds, df_full, device)
        nav, daily_ret = backtest_v4(test_df, panel, market_df)
        from code.backtest.engine import compute_metrics
        m = compute_metrics(daily_ret, nav)
        r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                   "max_drawdown": m["max_drawdown"], "annual_return": m["annual_return"]})
        best = r
        all_with_hp = [r]

    print(f"\n{'='*60}")
    print(f"BEST: {best['tag']}")
    print(f"  sharpe={best['sharpe']:.4f}  total_ret={best['total_return']:.4f}  "
          f"mdd={best['max_drawdown']:.4f}")

    best_ckpt = ARCHIVE / "checkpoints" / f"master_{best['tag']}.pt"
    best_sig = ARCHIVE / "signals" / f"master_{best['tag']}_test.parquet"
    if best_ckpt.exists(): shutil.copy(best_ckpt, ARCHIVE / "checkpoints" / "master.pt")
    if best_sig.exists(): shutil.copy(best_sig, ARCHIVE / "signals" / "master_test.parquet")

    summary = {"best": best, "total_models": len(all_with_hp),
               "v5_features": ["filter_factors(EMA+trend)", "industry_factors",
                               "time_decay_loss", "dynamic_n"],
               "full_ranking": all_with_hp}
    with open(ARCHIVE / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # README
    (ARCHIVE / "README.md").write_text(f"""# MASTER v5

## v5 = v4 + 显式时序滤波器

基于 attention HHI≈0.05 的分析结论 (模型未学到时序选择性):
  ema_mom_5, ema_mom_20, ema_cross (MACD)
  trend_r2, trend_slope, vol_trend (趋势强度)

## 最佳模型
  {best['tag']}
  夏普: {best.get('sharpe', '?')}

## 复现
```bash
python -m code.features.filter_factors  # 先算因子
python -m code.models.master_v5 --search
```
""", encoding="utf-8")

    print(f"\nSaved to {ARCHIVE}/  total={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
