"""
MASTER v2: 优化版主模型
对 v1 的改动:
  - T 20 -> 30 (更长时序窗口, 容纳 120日动量充分历史)
  - H 64 -> 96 (适度加宽)
  - intra_layers 2->3, inter_layers 1->2 (加深横截面建模)
  - dropout 0.2 -> 0.25
  - lr 5e-4 -> 3e-4 + warmup 2 epochs (缓和早期梯度)
  - EMA decay 0.999 (减少 batch_size=1 训练方差)
  - max_epochs 20 -> 30, patience 6 -> 10 (给更深模型空间)
  - Loss: alpha 0.6 -> 0.5 + 增大 topk margin 0.1 -> 0.2

Splits: train 2016-2022 / valid 2023 / test 2024-2025
Output:
  output/checkpoints/master_v2.pt
  output/signals/master_v2_test.parquet
  output/master_v2_metrics.json
"""
import json
import time
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from code.config import (
    ROOT, CACHE, OUTPUT,
    FACTOR_COLS, N_FEAT, TRAIN_MAX, VALID_MAX,
)

T = 30  # v2: longer lookback window

(OUTPUT / "checkpoints").mkdir(parents=True, exist_ok=True)
(OUTPUT / "signals").mkdir(parents=True, exist_ok=True)


# ========== Dataset ==========

class DailyPanelDataset(Dataset):
    """每个样本是一天: (X[N,T,F], market[T,Fm], y[N], date)"""

    def __init__(self, X, y, trade_dates, ts_codes,
                 market_X, market_date_idx, T, min_stocks=30):
        self.X = X
        self.y = y
        self.trade_dates = trade_dates
        self.market_X = market_X
        self.market_date_idx = market_date_idx
        self.T = T

        print("  Building daily endpoints ...")
        df_tmp = pd.DataFrame({"code": ts_codes, "row": np.arange(len(ts_codes)),
                                "date": trade_dates})
        date_eps = {}
        for code, sub in tqdm(df_tmp.groupby("code", sort=False), desc="    stocks", unit="stk"):
            rows = sub["row"].values
            dates = sub["date"].values
            for k in range(T - 1, len(rows)):
                d = dates[k]
                date_eps.setdefault(d, []).append(rows[k])

        self.dates = sorted(
            d for d, eps in date_eps.items()
            if len(eps) >= min_stocks and d in market_date_idx
            and market_date_idx[d] >= T - 1  # market window 也要够
        )
        self.date_to_endpoints = {d: np.array(date_eps[d], dtype=np.int64) for d in self.dates}
        print(f"    total days: {len(self.dates)}, "
              f"avg stocks/day: {np.mean([len(v) for v in self.date_to_endpoints.values()]):.0f}")

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, i):
        date = self.dates[i]
        endpoints = self.date_to_endpoints[date]
        starts = endpoints - self.T + 1
        rng = np.arange(self.T)
        idx_2d = starts[:, None] + rng[None, :]
        X_d = self.X[idx_2d]
        y_d = self.y[endpoints]

        m_end = self.market_date_idx[date]
        market_window = self.market_X[m_end - self.T + 1 : m_end + 1]

        return (
            torch.from_numpy(X_d.copy()),
            torch.from_numpy(market_window.copy()),
            torch.from_numpy(y_d.copy()),
            int(date),
            endpoints,
        )

    def subset_by_dates(self, dates):
        missing = set(dates) - set(self.date_to_endpoints)
        if missing:
            raise ValueError(f"Dates not in dataset: {sorted(missing)[:5]}...")

        sub = object.__new__(type(self))
        sub.X = self.X
        sub.y = self.y
        sub.trade_dates = self.trade_dates
        sub.market_X = self.market_X
        sub.market_date_idx = self.market_date_idx
        sub.T = self.T
        sub.dates = list(dates)
        sub.date_to_endpoints = self.date_to_endpoints
        return sub


def collate_single(batch):
    return batch[0]


# ========== Model ==========

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class MASTERv2(nn.Module):
    def __init__(self, F_stock=20, F_market=12, H=96, T=30, nhead=4, dropout=0.25,
                 n_intra_layers=3, n_inter_layers=2):
        super().__init__()
        # Market guidance
        self.market_proj = nn.Linear(F_market, H)
        self.gate_proj = nn.Linear(H, F_stock)

        # Stock embedding
        self.stock_proj = nn.Linear(F_stock, H)
        self.pos_enc = PositionalEncoding(H, max_len=T + 8)

        # Intra-stock Transformer (temporal axis)
        intra_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=H * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.intra_tx = nn.TransformerEncoder(intra_layer, num_layers=n_intra_layers)

        # Temporal aggregation
        self.temp_query = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.temp_attn = nn.MultiheadAttention(H, nhead, dropout=dropout, batch_first=True)
        self.temp_ln = nn.LayerNorm(H)

        # Inter-stock Transformer (cross-sectional axis)
        inter_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=H * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.inter_tx = nn.TransformerEncoder(inter_layer, num_layers=n_inter_layers)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, market):
        N, T_in, _ = X.shape

        # Market guidance gate
        m_emb = self.market_proj(market)
        gate = torch.sigmoid(self.gate_proj(m_emb))
        X = X * gate.unsqueeze(0)

        # Stock projection + pos enc
        Z = self.stock_proj(X)
        Z = self.pos_enc(Z)
        Z = self.dropout(Z)

        # Intra-stock temporal Transformer
        Z = self.intra_tx(Z)

        # Temporal aggregation (attention pooling) + residual
        q = self.temp_query.expand(N, -1, -1)
        Z_t, _ = self.temp_attn(q, Z, Z)
        Z_t = self.temp_ln(Z_t.squeeze(1) + Z.mean(1))  # 加 mean pool 残差

        # Inter-stock cross-sectional Transformer
        Z_c = self.inter_tx(Z_t.unsqueeze(0)).squeeze(0)

        Z_final = Z_t + Z_c
        return self.head(Z_final).squeeze(-1)


# ========== EMA ==========

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        """返回一个加载了 EMA 权重的 model 副本"""
        ema_state = {}
        for k, v in model.state_dict().items():
            ema_state[k] = self.shadow[k] if k in self.shadow else v
        model.load_state_dict(ema_state)


# ========== Loss (v2 uses larger margin=0.2) ==========

from code.losses import ic_loss, combined_loss
from code.losses import topk_margin_loss as _topk_margin_loss

def topk_margin_loss(scores, labels, k_ratio=0.2, margin=0.2):
    return _topk_margin_loss(scores, labels, k_ratio=k_ratio, margin=margin)


from code.metrics import ic_summary

# ========== Evaluate ==========

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for X_d, m_d, y_d, date, endpoints in loader:
        X_d, m_d = X_d.to(device, non_blocking=True), m_d.to(device, non_blocking=True)
        scores = model(X_d, m_d).cpu().numpy()
        for s, y, end in zip(scores, y_d.numpy(), endpoints):
            rows.append({"trade_date": date, "ep": int(end), "score": float(s), "label": float(y)})
    df = pd.DataFrame(rows)
    ic, ric, ric_std = ic_summary(df)
    return ic, ric, ric_std, df


# ========== Main ==========

def prepare(features_df, labels_df, market_df):
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

    market_df = market_df.sort_values("trade_date").reset_index(drop=True)
    market_cols = [c for c in market_df.columns if c != "trade_date"]
    market_X = market_df[market_cols].fillna(0).values.astype(np.float32)
    market_date_idx = {int(d): i for i, d in enumerate(market_df["trade_date"].values)}

    return X, y, trade_dates, ts_codes, code_uniq, market_X, market_date_idx, df


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42); np.random.seed(42)

    print("\nLoading features / labels / market ...")
    feats = pd.read_parquet(CACHE / "features.parquet")
    labels = pd.read_parquet(CACHE / "labels.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")

    print("Preparing ...")
    X, y, trade_dates, ts_codes, code_uniq, market_X, market_date_idx, df_full = prepare(
        feats, labels, market
    )

    print(f"Building dataset (T={T}) ...")
    full_ds = DailyPanelDataset(X, y, trade_dates, ts_codes, market_X, market_date_idx, T)

    train_dates = [d for d in full_ds.dates if d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]
    print(f"  train_days={len(train_dates)}  valid_days={len(valid_dates)}  test_days={len(test_dates)}")

    train_ds = full_ds.subset_by_dates(train_dates)
    valid_ds = full_ds.subset_by_dates(valid_dates)
    test_ds = full_ds.subset_by_dates(test_dates)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              collate_fn=collate_single)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=0,
                              collate_fn=collate_single)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=collate_single)

    F_market = market_X.shape[1]
    model = MASTERv2(F_stock=N_FEAT, F_market=F_market, H=96, T=T,
                     nhead=4, dropout=0.25,
                     n_intra_layers=3, n_inter_layers=2).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {nparams:,} params")

    # v2: lr 降低 + warmup
    base_lr = 3e-4
    warmup_epochs = 2
    max_epochs = 30
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=2e-3)

    # Custom LR schedule: warmup -> cosine
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ema = EMA(model, decay=0.999)

    ckpt = OUTPUT / "checkpoints" / "master_v2.pt"
    best_val_rank_ic = -1.0
    patience = 0
    patience_max = 10
    epoch_best = 0

    print(f"\nTraining (max_epochs={max_epochs}, patience={patience_max}, "
          f"base_lr={base_lr}, warmup={warmup_epochs}) ...")
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for X_d, m_d, y_d, _, _ in train_loader:
            X_d = X_d.to(device, non_blocking=True)
            m_d = m_d.to(device, non_blocking=True)
            y_d = y_d.to(device, non_blocking=True)
            pred = model(X_d, m_d)
            loss = combined_loss(pred, y_d, alpha=0.5)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            losses.append(loss.item())
        sched.step()

        # 在 EMA 副本上评估
        eval_model = copy.deepcopy(model)
        ema.apply_to(eval_model)
        val_ic, val_rank_ic, _, _ = evaluate(eval_model, valid_loader, device)
        del eval_model

        print(f"Epoch {epoch+1:2d}: train_loss={np.mean(losses):.4f}  "
              f"val_ic={val_ic:.4f}  val_rank_ic={val_rank_ic:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.5f}")

        if val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = val_rank_ic
            epoch_best = epoch + 1
            patience = 0
            # 保存 EMA 权重
            ema_model = copy.deepcopy(model)
            ema.apply_to(ema_model)
            torch.save(ema_model.state_dict(), ckpt)
            del ema_model
        else:
            patience += 1
            if patience >= patience_max:
                print(f"Early stop at epoch {epoch+1}, best=epoch{epoch_best} "
                      f"val_rank_ic={best_val_rank_ic:.4f}")
                break

    # 用最佳 EMA 权重 → 测试
    model.load_state_dict(torch.load(ckpt))
    test_ic, test_rank_ic, test_rank_ic_std, test_df = evaluate(model, test_loader, device)

    test_df["ts_code"] = df_full.loc[test_df["ep"].values, "ts_code"].values
    test_df = test_df.drop(columns=["ep"])

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
        "config": {
            "T": T, "H": 96, "nhead": 4, "intra_layers": 3, "inter_layers": 2,
            "dropout": 0.25, "base_lr": base_lr, "weight_decay": 2e-3,
            "warmup": warmup_epochs, "ema_decay": 0.999,
            "loss": "0.5*IC + 0.5*TopK(margin=0.2)",
        },
    }
    print("\n=== MASTER v2 Test Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        elif isinstance(v, int):
            print(f"  {k:25s} {v}")
        else:
            print(f"  {k:25s} {v:.4f}")

    test_df.to_parquet(OUTPUT / "signals" / "master_v2_test.parquet", index=False)
    with open(OUTPUT / "master_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOK total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
