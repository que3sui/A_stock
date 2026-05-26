"""
GRU + Attention 主模型
  输入: [B, T=20, F=20] (过去20日的20个因子)
  输出: 标量 (预测 5日横截面rank)
  损失: IC Loss (-Pearson correlation)
  训练: AdamW + Cosine + 早停 patience=5
Splits:
  train 2016-2022 / valid 2023 / test 2024-2025
Output:
  output/checkpoints/gru_att.pt
  output/signals/gru_test.parquet
  output/gru_metrics.json
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
    """通过 endpoint 索引切片 (T,F) 窗口"""
    def __init__(self, X, y, trade_dates, ts_codes, endpoints, T):
        self.X = X
        self.y = y
        self.trade_dates = trade_dates
        self.ts_codes = ts_codes  # numeric mapping (int)
        self.endpoints = endpoints
        self.T = T

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, i):
        end = self.endpoints[i]
        X_win = self.X[end - self.T + 1: end + 1]  # (T, F)
        return (
            torch.from_numpy(X_win),
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
    # ts_code 编码为整数 (用于回测时还原)
    code_uniq, code_int = np.unique(df["ts_code"].values, return_inverse=True)
    ts_codes = code_int.astype(np.int64)

    # 构造 endpoints: 每只股票第 T-1 行后的 row index
    print("Building endpoints ...")
    code_groups = df.groupby("ts_code", sort=False).indices  # dict: code -> array of row idx
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


class GRUAtt(nn.Module):
    def __init__(self, n_feat=20, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):                  # x: (B, T, F)
        h, _ = self.gru(x)                 # (B, T, H)
        w = torch.softmax(self.attn(h), 1) # (B, T, 1)
        ctx = (h * w).sum(1)               # (B, H)
        return self.head(ctx).squeeze(-1)  # (B,)


def ic_loss(pred, target):
    """Negative Pearson correlation"""
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
        preds.append(p)
        targets.append(y.numpy())
        dates.append(date.numpy())
        codes.append(code.numpy())
    df = pd.DataFrame({
        "score": np.concatenate(preds),
        "label": np.concatenate(targets),
        "trade_date": np.concatenate(dates),
        "code_int": np.concatenate(codes),
    })
    ics = []
    rank_ics = []
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
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    np.random.seed(42)

    print("\nLoading features & labels ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")

    print("Preparing ...")
    X, y, trade_dates, ts_codes, endpoints, code_uniq = prepare(feats, labels)

    train_ep, valid_ep, test_ep = split_endpoints(trade_dates, endpoints)
    print(f"  train={len(train_ep):,}  valid={len(valid_ep):,}  test={len(test_ep):,}")

    train_ds = PanelDataset(X, y, trade_dates, ts_codes, train_ep, T)
    valid_ds = PanelDataset(X, y, trade_dates, ts_codes, valid_ep, T)
    test_ds = PanelDataset(X, y, trade_dates, ts_codes, test_ep, T)

    BATCH = 4096
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                             num_workers=0, pin_memory=(device == "cuda"))

    model = GRUAtt(n_feat=N_FEAT, hidden=64, num_layers=2, dropout=0.4).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {nparams:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    ckpt = OUTPUT / "checkpoints" / "gru_att.pt"
    best_val_rank_ic = -1.0
    patience = 0
    epoch_best = 0
    max_epochs = 50
    patience_max = 8

    print("\nTraining ...")
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
        print(f"Epoch {epoch+1:2d}: train_loss={np.mean(ep_losses):.4f}  "
              f"val_ic={val_ic:.4f}  val_rank_ic={val_rank_ic:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.5f}")

        if val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = val_rank_ic
            epoch_best = epoch + 1
            patience = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= patience_max:
                print(f"Early stop at epoch {epoch+1}, best=epoch{epoch_best} val_rank_ic={best_val_rank_ic:.4f}")
                break

    # 加载最佳权重 → 测试集
    model.load_state_dict(torch.load(ckpt))
    test_ic, test_rank_ic, test_rank_ic_std, test_df = evaluate(model, test_loader, device)

    # 还原 ts_code
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
    print("\n=== GRU+Att Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:25s} {v}" if isinstance(v, int) else f"  {k:25s} {v:.4f}")

    test_df.to_parquet(OUTPUT / "signals" / "gru_test.parquet", index=False)
    with open(OUTPUT / "gru_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOK total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
