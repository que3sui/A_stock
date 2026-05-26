"""
MLP Baseline (作业明确建议的简单深度学习对比模型)
  - 3层 MLP, 输入 (T, F)=20×20=400 拉平
  - 教授讲义"方案A: 最简基线"实现
Splits: train 2016-2022 / valid 2023 / test 2024-2025
Output:
  output/checkpoints/mlp.pt
  output/signals/mlp_test.parquet
  output/mlp_metrics.json
  output/mlp_loss.json  (训练 loss history)
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
(OUTPUT / "checkpoints").mkdir(parents=True, exist_ok=True)
(OUTPUT / "signals").mkdir(parents=True, exist_ok=True)

FACTOR_COLS = [
    "mom_5", "mom_20", "mom_60", "mom_120",
    "rev_1", "rev_5",
    "vol_20", "vol_60",
    "turnover_20", "amihud_20",
    "mf_net_5", "mf_lg_strength", "mf_elg_strength",
    "pe_ttm_rank", "pb_rank", "circ_mv_log",
    "rsi_14", "bias_20", "vwap_dev", "vol_zscore",
]
T = 20
N_FEAT = len(FACTOR_COLS)
TRAIN_MAX = 20221231
VALID_MAX = 20231231


class PanelDataset(Dataset):
    def __init__(self, X, y, trade_dates, ts_codes, endpoints, T):
        self.X = X
        self.y = y
        self.trade_dates = trade_dates
        self.ts_codes = ts_codes
        self.endpoints = endpoints
        self.T = T

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, i):
        end = self.endpoints[i]
        X_win = self.X[end - self.T + 1: end + 1]
        return (
            torch.from_numpy(X_win.reshape(-1)),  # flatten (T*F,)
            torch.tensor(self.y[end], dtype=torch.float32),
            torch.tensor(self.trade_dates[end], dtype=torch.int64),
            torch.tensor(self.ts_codes[end], dtype=torch.int64),
        )


def prepare(features_df, labels_df):
    df = features_df.merge(
        labels_df[["trade_date", "ts_code", "label"]],
        on=["trade_date", "ts_code"], how="inner",
    )
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    X = df[FACTOR_COLS].fillna(0).values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    trade_dates = df["trade_date"].values.astype(np.int64)
    code_uniq, code_int = np.unique(df["ts_code"].values, return_inverse=True)
    ts_codes = code_int.astype(np.int64)

    code_groups = df.groupby("ts_code", sort=False).indices
    endpoints = []
    for code, rows in tqdm(code_groups.items(), desc="endpoints", unit="stk"):
        rows = np.sort(rows)
        if len(rows) >= T:
            endpoints.append(rows[T - 1:])
    endpoints = np.concatenate(endpoints).astype(np.int64)
    print(f"  total endpoints: {len(endpoints):,}")

    return X, y, trade_dates, ts_codes, endpoints, code_uniq


def split_endpoints(trade_dates, endpoints):
    end_dates = trade_dates[endpoints]
    train_ep = endpoints[end_dates <= TRAIN_MAX]
    valid_ep = endpoints[(end_dates > TRAIN_MAX) & (end_dates <= VALID_MAX)]
    test_ep = endpoints[end_dates > VALID_MAX]
    return train_ep, valid_ep, test_ep


class MLP(nn.Module):
    """讲义方案A: T*F 拉平 → 3 层 MLP"""
    def __init__(self, in_dim=T * N_FEAT, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def ic_loss(pred, target):
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    num = (pred_c * target_c).sum()
    den = torch.sqrt((pred_c ** 2).sum() * (target_c ** 2).sum() + 1e-12)
    return -num / den


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets, dates, codes = [], [], [], []
    for X, y, date, code in loader:
        X = X.to(device, non_blocking=True)
        p = model(X).cpu().numpy()
        preds.append(p); targets.append(y.numpy())
        dates.append(date.numpy()); codes.append(code.numpy())
    df = pd.DataFrame({
        "score": np.concatenate(preds),
        "label": np.concatenate(targets),
        "trade_date": np.concatenate(dates),
        "code_int": np.concatenate(codes),
    })
    ics, rank_ics = [], []
    for _, day in df.groupby("trade_date"):
        if len(day) < 30 or day["score"].std() == 0:
            continue
        ics.append(day["score"].corr(day["label"]))
        rank_ics.append(day["score"].rank().corr(day["label"].rank()))
    return float(np.mean(ics)), float(np.mean(rank_ics)), float(np.std(rank_ics)), df


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(42); np.random.seed(42)

    print("Loading features & labels ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")

    X, y, trade_dates, ts_codes, endpoints, code_uniq = prepare(feats, labels)
    train_ep, valid_ep, test_ep = split_endpoints(trade_dates, endpoints)
    print(f"  train={len(train_ep):,}  valid={len(valid_ep):,}  test={len(test_ep):,}")

    train_ds = PanelDataset(X, y, trade_dates, ts_codes, train_ep, T)
    valid_ds = PanelDataset(X, y, trade_dates, ts_codes, valid_ep, T)
    test_ds = PanelDataset(X, y, trade_dates, ts_codes, test_ep, T)

    BATCH = 4096
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = MLP().to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: {nparams:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    ckpt = OUTPUT / "checkpoints" / "mlp.pt"
    best_val_rank_ic = -1.0
    patience = 0
    epoch_best = 0
    max_epochs = 30
    patience_max = 6

    loss_history = {"epoch": [], "train_loss": [], "val_ic": [], "val_rank_ic": [], "lr": []}

    for epoch in range(max_epochs):
        model.train()
        ep_losses = []
        for X_b, y_b, _, _ in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            pred = model(X_b)
            loss = ic_loss(pred, y_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        sched.step()

        val_ic, val_rank_ic, _, _ = evaluate(model, valid_loader, device)
        train_loss = float(np.mean(ep_losses))
        lr_cur = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.4f}  "
              f"val_ic={val_ic:.4f}  val_rank_ic={val_rank_ic:.4f}  lr={lr_cur:.5f}")
        loss_history["epoch"].append(epoch + 1)
        loss_history["train_loss"].append(train_loss)
        loss_history["val_ic"].append(val_ic)
        loss_history["val_rank_ic"].append(val_rank_ic)
        loss_history["lr"].append(lr_cur)

        if val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = val_rank_ic
            epoch_best = epoch + 1
            patience = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= patience_max:
                print(f"Early stop at epoch {epoch+1}, best=epoch{epoch_best}")
                break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    test_ic, test_rank_ic, test_rank_ic_std, test_df = evaluate(model, test_loader, device)
    test_df["ts_code"] = code_uniq[test_df["code_int"].values]
    test_df = test_df.drop(columns=["code_int"])

    metrics = {
        "test_days": int(test_df["trade_date"].nunique()),
        "test_rows": int(len(test_df)),
        "best_val_rank_ic": float(best_val_rank_ic),
        "best_epoch": int(epoch_best),
        "test_ic_mean": float(test_ic),
        "test_rank_ic_mean": float(test_rank_ic),
        "test_rank_ic_std": float(test_rank_ic_std),
        "test_rank_icir": float(test_rank_ic / (test_rank_ic_std + 1e-8)),
        "test_rank_icir_annual": float(test_rank_ic / (test_rank_ic_std + 1e-8) * np.sqrt(252)),
        "n_params": int(nparams),
    }
    print("\n=== MLP Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:25s} {v}" if isinstance(v, int) else f"  {k:25s} {v:.4f}")

    test_df.to_parquet(OUTPUT / "signals" / "mlp_test.parquet", index=False)
    with open(OUTPUT / "mlp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(OUTPUT / "mlp_loss.json", "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)
    print(f"\nOK total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
