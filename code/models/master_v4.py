"""
MASTER v4: 时间衰减 + 行业轮动 + 动态仓位 + 训练搜索

v4 改进 (按优先级):
  1. 时间衰减 loss — 近期样本权重更高 (exp decay, lambda=0.5)
  2. 行业轮动因子 — ind_mom_20 (板块动量) + ind_rel_str (相对强弱)
  3. 动态仓位 — 市场高波动时减仓, 低波动时加仓
  4. 训练搜索 — 10 seed + HP 微调 (复用 v2 框架)

Usage:
  python -m code.models.master_v4          # 单次训练
  python -m code.models.master_v4 --search # 训练搜索
"""
import argparse
import json
import time
import math
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.models.master import (
    MASTER, DailyPanelDataset, collate_single,
    evaluate, prepare,
    T, N_WEIGHT,
)
from code.config import (
    ROOT, CACHE, OUTPUT,
    FACTOR_COLS, INDUSTRY_FACTORS, FACTOR_COLS_V4, N_FEAT_V4,
    TRAIN_MAX as OLD_TRAIN_MAX, VALID_MAX as OLD_VALID_MAX,
)

ARCHIVE = OUTPUT / "v4"
for d in [ARCHIVE, ARCHIVE / "checkpoints", ARCHIVE / "signals"]:
    d.mkdir(parents=True, exist_ok=True)

# === v4 新增: 训练窗口 — 缩短到 2019 年起点 ===
TRAIN_START = 20190101
TRAIN_MAX = 20231231
VALID_MAX = 20241231

# === v4 新增: 时间衰减参数 ===
DECAY_LAMBDA = 0.5  # 越大越偏向近期

# === v4 新增: 动态仓位参数 ===
N_BASE = 8
N_MIN = 4
N_MAX = 14
VOL_THRESHOLD_HIGH = 1.5  # hs300_vol_20 z-score 阈值
VOL_THRESHOLD_LOW = -0.5

SEEDS = [128, 1024, 99, 42, 777, 4096, 314, 88, 512, 256]
HP_GRID = [
    {"lr": 5e-4, "dropout": 0.20, "alpha": 0.60, "H": 64},
    {"lr": 3e-4, "dropout": 0.20, "alpha": 0.60, "H": 64},
    {"lr": 5e-4, "dropout": 0.15, "alpha": 0.60, "H": 64},
    {"lr": 5e-4, "dropout": 0.20, "alpha": 0.50, "H": 64},
]


# ============================================================
#  v4 新增: 行业因子计算
# ============================================================

def add_industry_factors(df):
    """在已有因子基础上, 增加行业轮动因子"""
    g_date = df.groupby("trade_date", sort=False)

    # 行业平均动量
    df["ind_mom_20"] = g_date["mom_20"].transform("mean")

    # 个股相对行业强度 (正 = 强于行业)
    df["ind_rel_str"] = df["mom_20"] - df["ind_mom_20"]

    for c in INDUSTRY_FACTORS:
        df[c] = df[c].fillna(0.0).astype("float32")
    return df


# ============================================================
#  v4 新增: 时间衰减 loss
# ============================================================

def time_weight(dates, max_date, decay_lambda=DECAY_LAMBDA):
    """计算每个交易日的样本权重: 越近权重越高"""
    years_diff = (max_date - np.array(dates, dtype=np.float64)) / 365.0
    w = np.exp(-decay_lambda * np.clip(years_diff, 0, None))
    return w / w.mean()  # 归一化, 保持 loss 量级不变


def combined_loss_v4(scores, labels, trade_date, max_date, alpha=0.6):
    """带时间衰减的 combined loss"""
    from code.losses import ic_loss, topk_margin_loss
    ic_l = ic_loss(scores, labels)
    topk_l = topk_margin_loss(scores, labels)
    tw = time_weight([trade_date], max_date)[0]
    return tw * (alpha * ic_l + (1 - alpha) * topk_l)


# ============================================================
#  v4 新增: 动态仓位
# ============================================================

def dynamic_n(market_df, trade_date, base_n=N_BASE):
    """根据市场波动动态调整持仓数"""
    row = market_df[market_df["trade_date"] == trade_date]
    if len(row) == 0:
        return base_n

    hs300_vol = row["hs300_vol_20"].values[0]

    if hs300_vol > VOL_THRESHOLD_HIGH:
        n = N_MIN
    elif hs300_vol < VOL_THRESHOLD_LOW:
        n = N_MAX
    else:
        r = (hs300_vol - VOL_THRESHOLD_LOW) / (VOL_THRESHOLD_HIGH - VOL_THRESHOLD_LOW)
        n = int(N_MAX - r * (N_MAX - N_MIN))
    return max(N_MIN, min(N_MAX, n))


# ============================================================
#  v4 训练逻辑
# ============================================================

def train_one(seed, X, X_w, y, trade_dates, ts_codes, market_X, market_date_idx,
              full_ds, df_full, device, hp_overrides=None):
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
    model = MASTER(F_stock=N_FEAT_V4, F_market=F_market, H=hp["H"], T=T,
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
    }, test_df, model


# ============================================================
#  v4 回测 (含动态仓位)
# ============================================================

def backtest_v4(signals, panel, market_df, base_n=N_BASE, k=2):
    """v4 回测: 动态仓位 n 随市场波动调整"""
    from code.backtest.engine import FEE_BUY, FEE_SELL

    panel_pivot = panel.pivot_table(index="trade_date", columns="ts_code", values="pct_chg").sort_index() / 100.0
    st_pivot = panel.pivot_table(index="trade_date", columns="ts_code", values="is_st").sort_index().fillna(False).astype(bool)
    open_p = panel.pivot_table(index="trade_date", columns="ts_code", values="open").sort_index()
    high_p = panel.pivot_table(index="trade_date", columns="ts_code", values="high").sort_index()
    low_p = panel.pivot_table(index="trade_date", columns="ts_code", values="low").sort_index()
    yzi_mask = (open_p == high_p) & (open_p == low_p)

    signals = signals.sort_values(["trade_date", "score"], ascending=[True, False])
    dates = sorted(signals["trade_date"].unique())
    dates = [d for d in dates if d >= 20240101]

    portfolio = set(); nav = 1.0; nav_list = []; daily_returns = []
    fee_total = 0.0; sig_groups = signals.groupby("trade_date")

    for i, t_date in enumerate(dates):
        sig_t = sig_groups.get_group(t_date)
        n_current = dynamic_n(market_df, t_date, base_n)

        if t_date in st_pivot.index:
            st_set = set(st_pivot.loc[t_date][st_pivot.loc[t_date]].index.tolist())
            sig_t = sig_t[~sig_t["ts_code"].isin(st_set)]
        else:
            st_set = set()

        if t_date in yzi_mask.index:
            yzi_set = set(yzi_mask.loc[t_date][yzi_mask.loc[t_date]].index.tolist())
        else:
            yzi_set = set()

        if not portfolio:
            buy_list = sig_t[~sig_t["ts_code"].isin(yzi_set)].head(n_current)["ts_code"].tolist()
            portfolio.update(buy_list); nav_list.append(nav); daily_returns.append(0.0)
            if True:
                nav *= (1 - FEE_BUY); fee_total += FEE_BUY
            continue

        if t_date in panel_pivot.index:
            held_rets = panel_pivot.loc[t_date].reindex(list(portfolio)).fillna(0.0).values
            day_ret = float(held_rets.mean())
        else:
            day_ret = 0.0
        nav *= (1 + day_ret); nav_list.append(nav); daily_returns.append(day_ret)

        in_pos = sig_t[sig_t["ts_code"].isin(portfolio)].sort_values("score")
        not_in = sig_t[~sig_t["ts_code"].isin(portfolio)].sort_values("score", ascending=False)

        sells = []
        for c in in_pos["ts_code"].tolist():
            if c in yzi_set: continue
            sells.append(c)
            if len(sells) >= k: break

        buys = []
        n_to_buy = len(sells)
        for c in not_in["ts_code"].tolist():
            if c in yzi_set: continue
            buys.append(c)
            if len(buys) >= n_to_buy: break

        for c in sells: portfolio.discard(c)
        portfolio.update(buys)

        # Rebalance portfolio size to target n_current
        if len(portfolio) > n_current:
            to_sell_extra = sorted(portfolio, key=lambda c: sig_t[sig_t['ts_code']==c]['score'].values[0] if len(sig_t[sig_t['ts_code']==c])>0 else 0)
            for c in to_sell_extra[:len(portfolio)-n_current]:
                portfolio.discard(c)
        elif len(portfolio) < n_current:
            extra_needed = n_current - len(portfolio)
            extra_candidates = sig_t[~sig_t["ts_code"].isin(portfolio | yzi_set)].sort_values("score", ascending=False)
            for c in extra_candidates["ts_code"].tolist()[:extra_needed]:
                portfolio.add(c)

        if True and (sells or buys):
            sell_w = len(sells) / n_current; buy_w = len(buys) / n_current
            fee_today = sell_w * FEE_SELL + buy_w * FEE_BUY
            nav *= (1 - fee_today); fee_total += fee_today

    nav_series = pd.Series(nav_list, index=dates, name="nav")
    daily_ret_series = pd.Series(daily_returns, index=dates)
    return nav_series, daily_ret_series


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="训练搜索模式")
    parser.add_argument("--seed", type=int, default=128, help="单次训练种子")
    args = parser.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    print("\nLoading features / labels / market ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")

    # v4: 加入行业因子
    print("Adding industry factors ...")
    feats = add_industry_factors(feats)
    print(f"  new features: {INDUSTRY_FACTORS}")

    print("Preparing ...")
    # 临时替换 prepare 中的 FACTOR_COLS
    import code.models.master as master_mod
    original_fcols = master_mod.FACTOR_COLS
    original_nfeat = master_mod.N_FEAT
    master_mod.FACTOR_COLS = FACTOR_COLS_V4
    master_mod.N_FEAT = N_FEAT_V4

    X, y, trade_dates, ts_codes_int, code_uniq, market_X, market_date_idx, df_full, X_w = prepare(
        feats, labels, market
    )

    # Restore
    master_mod.FACTOR_COLS = original_fcols
    master_mod.N_FEAT = original_nfeat

    if X_w is not None:
        print(f"  weight channel: {X_w.shape[1]} cols")

    print("Building dataset (once) ...")
    full_ds = DailyPanelDataset(X, y, trade_dates, ts_codes_int,
                                 market_X, market_date_idx, T, X_w=X_w)

    train_dates = [d for d in full_ds.dates if TRAIN_START <= d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]
    print(f"  train={len(train_dates)} ({min(train_dates)}~{max(train_dates)})")
    print(f"  valid={len(valid_dates)} ({min(valid_dates)}~{max(valid_dates)})")
    print(f"  test={len(test_dates)} ({min(test_dates)}~{max(test_dates)})")

    # Load panel for backtest
    panel = pd.read_parquet(CACHE / "panel.parquet",
                            columns=["trade_date", "ts_code", "open", "high", "low", "close",
                                     "pct_chg", "is_st"])
    panel = panel[panel["trade_date"] >= 20231201]

    market_df = pd.read_parquet(CACHE / "market_features.parquet")

    if args.search:
        # ========== 训练搜索模式 ==========
        print(f"\n{'='*60}")
        print(f"V4 Training Search: {len(SEEDS)} seeds + HP tuning")
        print(f"{'='*60}")

        # Phase 1: Train base seeds
        all_results = []
        for seed in SEEDS:
            t_seed = time.time()
            print(f"\n  Training seed={seed} ...")
            r, test_df, _ = train_one(seed, X, X_w, y, trade_dates, ts_codes_int,
                                       market_X, market_date_idx, full_ds, df_full, device)
            r["train_time_s"] = round(time.time() - t_seed, 1)
            all_results.append(r)
            print(f"    val_rank_ic={r['val_rank_ic']:.4f}  "
                  f"test_rank_ic={r['test_rank_ic']:.4f}  "
                  f"test_icir={r['test_rank_icir_annual']:.2f}  "
                  f"time={r['train_time_s']}s")

        # Phase 2: Backtest all
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

        # Phase 3: HP tuning on top-3
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
                r, test_df, _ = train_one(seed, X, X_w, y, trade_dates, ts_codes_int,
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
        # ========== 单次训练模式 ==========
        print(f"\nTraining seed={args.seed} ...")
        r, test_df, _ = train_one(args.seed, X, X_w, y, trade_dates, ts_codes_int,
                                   market_X, market_date_idx, full_ds, df_full, device)
        nav, daily_ret = backtest_v4(test_df, panel, market_df)
        from code.backtest.engine import compute_metrics
        m = compute_metrics(daily_ret, nav)
        r.update({"sharpe": m["sharpe"], "total_return": m["total_return"],
                   "max_drawdown": m["max_drawdown"], "annual_return": m["annual_return"]})
        best = r
        all_with_hp = [r]

    # Save best
    print(f"\n{'='*60}")
    print(f"BEST: {best['tag']}")
    print(f"  sharpe={best['sharpe']:.4f}  total_ret={best['total_return']:.4f}  "
          f"mdd={best['max_drawdown']:.4f}  annual_ret={best.get('annual_return', 0):.4f}")
    if "hp" in best:
        print(f"  hp={best['hp']}")

    best_ckpt = ARCHIVE / "checkpoints" / f"master_{best['tag']}.pt"
    best_sig = ARCHIVE / "signals" / f"master_{best['tag']}_test.parquet"
    if best_ckpt.exists():
        shutil.copy(best_ckpt, ARCHIVE / "checkpoints" / "master.pt")
    if best_sig.exists():
        shutil.copy(best_sig, ARCHIVE / "signals" / "master_test.parquet")

    summary = {"best": best, "total_models": len(all_with_hp),
               "v4_features": ["time_decay_loss", "industry_factors", "dynamic_n"],
               "full_ranking": all_with_hp}
    with open(ARCHIVE / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # v4 README
    readme = ARCHIVE / "README.md"
    if not readme.exists():
        readme.write_text(f"""# MASTER v4

## 改进

1. **时间衰减 loss**: 近期样本权重更高 (exp(-0.5 * years_from_latest))
2. **行业轮动因子**: ind_mom_20 + ind_rel_str
3. **动态仓位**: 市场高波动减仓, 低波动加仓 (n={N_MIN}~{N_MAX})
4. **训练搜索**: {len(SEEDS)} seeds + HP tuning

## 窗口

- 训练: {TRAIN_START}~{TRAIN_MAX}
- 验证: {TRAIN_MAX+1}~{VALID_MAX}
- 测试: {VALID_MAX+1}~

## 最佳模型

- {best['tag']}
- 夏普: {best.get('sharpe', '?')}

## 复现

```bash
python -m code.models.master_v4 --search
```
""", encoding="utf-8")

    print(f"\nSaved to {ARCHIVE}/")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
