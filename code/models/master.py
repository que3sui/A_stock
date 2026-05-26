"""
MASTER: Market-guided Stock Transformer (AAAI 2024, 简化版)
  - Market-guided gating: 用市场状态信号调制原始因子
  - Intra-stock Transformer (时序 axis)
  - Inter-stock Transformer (横截面 axis)
  - Listwise loss: IC + Top-K margin

By-day batch: 每个 batch 是一天的所有股票, 模型直接对当日横截面做选股

Splits: train 2016-2022 / valid 2023 / test 2024-2025
Output:
  output/checkpoints/master.pt
  output/signals/master_test.parquet
  output/master_metrics.json
"""
import json
import time
import math
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

        # 构造每个日期的有效 endpoints
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
        )
        self.date_to_endpoints = {d: np.array(date_eps[d], dtype=np.int64) for d in self.dates}
        print(f"    total days: {len(self.dates)}, "
              f"avg stocks/day: {np.mean([len(v) for v in self.date_to_endpoints.values()]):.0f}")

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, i):
        date = self.dates[i]
        endpoints = self.date_to_endpoints[date]
        # X[end-T+1 : end+1] for each endpoint
        starts = endpoints - self.T + 1
        # 用 fancy indexing
        rng = np.arange(self.T)
        idx_2d = starts[:, None] + rng[None, :]  # [N, T]
        X_d = self.X[idx_2d]                     # [N, T, F]
        y_d = self.y[endpoints]                  # [N]

        # Market window
        m_end = self.market_date_idx[date]
        market_window = self.market_X[m_end - self.T + 1 : m_end + 1]  # [T, Fm]

        return (
            torch.from_numpy(X_d.copy()),
            torch.from_numpy(market_window.copy()),
            torch.from_numpy(y_d.copy()),
            int(date),
            endpoints,
        )


def collate_single(batch):
    """batch 大小固定为 1, 直接 unwrap"""
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

    def forward(self, x):  # x: [N, T, d]
        return x + self.pe[:x.size(1)].unsqueeze(0)


class MASTER(nn.Module):
    def __init__(self, F_stock=20, F_market=12, H=64, T=20, nhead=4, dropout=0.2,
                 n_intra_layers=2, n_inter_layers=1):
        super().__init__()
        # Market guidance
        self.market_proj = nn.Linear(F_market, H)
        self.gate_proj = nn.Linear(H, F_stock)

        # Stock embedding
        self.stock_proj = nn.Linear(F_stock, H)
        self.pos_enc = PositionalEncoding(H, max_len=T + 4)

        # Intra-stock Transformer (temporal axis)
        intra_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=H * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.intra_tx = nn.TransformerEncoder(intra_layer, num_layers=n_intra_layers)

        # Temporal aggregation: 学习的 query 做 attention pooling
        self.temp_query = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.temp_attn = nn.MultiheadAttention(H, nhead, dropout=dropout, batch_first=True)

        # Inter-stock Transformer (cross-sectional axis)
        inter_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=H * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.inter_tx = nn.TransformerEncoder(inter_layer, num_layers=n_inter_layers)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, market):
        """
        X: [N, T, F_stock]
        market: [T, F_market]
        Returns: scores [N]
        """
        N, T_in, _ = X.shape

        # Market guidance gate
        m_emb = self.market_proj(market)          # [T, H]
        gate = torch.sigmoid(self.gate_proj(m_emb))  # [T, F_stock]
        X = X * gate.unsqueeze(0)                 # [N, T, F_stock]

        # Stock projection + pos enc
        Z = self.stock_proj(X)                    # [N, T, H]
        Z = self.pos_enc(Z)
        Z = self.dropout(Z)

        # Intra-stock temporal Transformer
        Z = self.intra_tx(Z)                      # [N, T, H]

        # Temporal aggregation (attention pooling)
        q = self.temp_query.expand(N, -1, -1)     # [N, 1, H]
        Z_t, _ = self.temp_attn(q, Z, Z)          # [N, 1, H]
        Z_t = Z_t.squeeze(1)                      # [N, H]

        # Inter-stock cross-sectional Transformer
        Z_c = self.inter_tx(Z_t.unsqueeze(0)).squeeze(0)  # [N, H]

        # Residual + head
        Z_final = Z_t + Z_c
        return self.head(Z_final).squeeze(-1)     # [N]


# ========== Loss ==========

def ic_loss(scores, labels):
    s = scores - scores.mean()
    l = labels - labels.mean()
    num = (s * l).sum()
    den = torch.sqrt((s ** 2).sum() * (l ** 2).sum() + 1e-12)
    return -num / den


def topk_margin_loss(scores, labels, k_ratio=0.2):
    """Top-K margin: 让真实 top-K 的预测分 > 真实 bottom-K 的预测分"""
    n = scores.size(0)
    k = max(int(n * k_ratio), 5)
    _, top_idx = torch.topk(labels, k)
    _, bot_idx = torch.topk(-labels, k)
    s_top = scores[top_idx].unsqueeze(1)  # [k, 1]
    s_bot = scores[bot_idx].unsqueeze(0)  # [1, k]
    margin = torch.clamp(s_bot - s_top + 0.1, min=0)
    return margin.mean()


def combined_loss(scores, labels, alpha=0.5):
    return alpha * ic_loss(scores, labels) + (1 - alpha) * topk_margin_loss(scores, labels)


# ========== Evaluate ==========

@torch.no_grad()
def evaluate(model, loader, device, code_uniq):
    model.eval()
    rows = []
    for X_d, m_d, y_d, date, endpoints in loader:
        X_d, m_d = X_d.to(device, non_blocking=True), m_d.to(device, non_blocking=True)
        scores = model(X_d, m_d).cpu().numpy()
        for s, y, end in zip(scores, y_d.numpy(), endpoints):
            rows.append({"trade_date": date, "ep": int(end), "score": float(s), "label": float(y)})
    df = pd.DataFrame(rows)

    ics, rank_ics = [], []
    for _, day in df.groupby("trade_date"):
        if len(day) < 30 or day["score"].std() == 0:
            continue
        ics.append(day["score"].corr(day["label"]))
        rank_ics.append(day["score"].rank().corr(day["label"].rank()))
    return float(np.mean(ics)), float(np.mean(rank_ics)), float(np.std(rank_ics)), df


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

    # Market features
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

    # 构造一个汇总 Dataset, 然后按日期 split
    print("Building dataset ...")
    full_ds = DailyPanelDataset(X, y, trade_dates, ts_codes, market_X, market_date_idx, T)

    # 按日期 split
    train_dates = [d for d in full_ds.dates if d <= TRAIN_MAX]
    valid_dates = [d for d in full_ds.dates if TRAIN_MAX < d <= VALID_MAX]
    test_dates = [d for d in full_ds.dates if d > VALID_MAX]
    print(f"  train_days={len(train_dates)}  valid_days={len(valid_dates)}  test_days={len(test_dates)}")

    # Sub-dataset (用同一份数据, 不同 dates 列表)
    def make_sub(dates):
        sub = object.__new__(DailyPanelDataset)
        sub.X = full_ds.X
        sub.y = full_ds.y
        sub.trade_dates = full_ds.trade_dates
        sub.market_X = full_ds.market_X
        sub.market_date_idx = full_ds.market_date_idx
        sub.T = full_ds.T
        sub.dates = dates
        sub.date_to_endpoints = full_ds.date_to_endpoints
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
    model = MASTER(F_stock=N_FEAT, F_market=F_market, H=64, T=T,
                   nhead=4, dropout=0.2, n_intra_layers=2, n_inter_layers=1).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {nparams:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    ckpt = OUTPUT / "checkpoints" / "master.pt"
    best_val_rank_ic = -1.0
    patience = 0
    patience_max = 6
    epoch_best = 0
    max_epochs = 20

    print("\nTraining ...")
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for X_d, m_d, y_d, _, _ in train_loader:
            X_d = X_d.to(device, non_blocking=True)
            m_d = m_d.to(device, non_blocking=True)
            y_d = y_d.to(device, non_blocking=True)
            pred = model(X_d, m_d)
            loss = combined_loss(pred, y_d, alpha=0.6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        val_ic, val_rank_ic, _, _ = evaluate(model, valid_loader, device, code_uniq)
        print(f"Epoch {epoch+1:2d}: train_loss={np.mean(losses):.4f}  "
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

    # 加载最佳权重 → 测试
    model.load_state_dict(torch.load(ckpt))
    test_ic, test_rank_ic, test_rank_ic_std, test_df = evaluate(model, test_loader, device, code_uniq)

    # 还原 ts_code (从 endpoint 找到对应 ts_code)
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
    }
    print("\n=== MASTER Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:25s} {v}" if isinstance(v, int) else f"  {k:25s} {v:.4f}")

    test_df.to_parquet(OUTPUT / "signals" / "master_test.parquet", index=False)
    with open(OUTPUT / "master_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOK total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
