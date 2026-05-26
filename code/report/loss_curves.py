"""
解析训练 log, 提取 loss curves 并画图
  - master / master_v2 / master_v3 / mlp 从 log 抽 epoch/loss/val_ic
  - 输出 figs/loss_curves.png + figs/loss_curves_grid.png
"""
import re
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
FIGS = OUTPUT / "reports" / "figs"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# 标准 epoch 行: "Epoch  1: train_loss=-0.0039  val_ic=0.0500  val_rank_ic=0.0453  lr=0.00050"
RE_EPOCH = re.compile(
    r"Epoch\s+(\d+):\s+train_loss=([-\d.]+)\s+val_ic=([-\d.]+)\s+val_rank_ic=([-\d.]+)"
)


def parse_log(path):
    if not path.exists():
        return None
    epochs, train_losses, val_ics, val_rank_ics = [], [], [], []
    seed = None
    seed_blocks = []  # list of (seed, [(ep, loss, ic, rank_ic)])
    cur_block = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    for line in lines:
        m_seed = re.match(r"Seed\s+(\d+)", line.strip())
        if m_seed:
            if cur_block and seed is not None:
                seed_blocks.append((seed, cur_block))
            seed = int(m_seed.group(1))
            cur_block = []
            continue
        m = RE_EPOCH.search(line)
        if m:
            ep, tl, vic, vric = m.groups()
            row = (int(ep), float(tl), float(vic), float(vric))
            epochs.append(row[0]); train_losses.append(row[1])
            val_ics.append(row[2]); val_rank_ics.append(row[3])
            cur_block.append(row)
    if cur_block and seed is not None:
        seed_blocks.append((seed, cur_block))
    return {
        "epochs": epochs, "train_losses": train_losses,
        "val_ics": val_ics, "val_rank_ics": val_rank_ics,
        "seed_blocks": seed_blocks,
    }


def parse_mlp_json():
    p = OUTPUT / "mlp_loss.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "epochs": d["epoch"],
        "train_losses": d["train_loss"],
        "val_ics": d["val_ic"],
        "val_rank_ics": d["val_rank_ic"],
        "seed_blocks": [],
    }


def plot_loss_grid(curves_dict, save):
    """4 张子图: 一个模型一张, 左轴 train_loss, 右轴 val_rank_ic"""
    models = [(k, v) for k, v in curves_dict.items() if v is not None]
    n = len(models)
    if n == 0:
        return False
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), squeeze=False)
    for ax, (name, c) in zip(axes.flatten(), models):
        if not c["epochs"]:
            ax.set_visible(False); continue
        ax2 = ax.twinx()
        ax.plot(c["epochs"], c["train_losses"], "o-", color="#1f77b4",
                label="train_loss (IC loss)", lw=2)
        ax2.plot(c["epochs"], c["val_rank_ics"], "s--", color="#d62728",
                 label="val rank_ic", lw=1.8, alpha=0.85)
        # 标记 best epoch
        best_idx = max(range(len(c["val_rank_ics"])), key=lambda i: c["val_rank_ics"][i])
        ax2.scatter([c["epochs"][best_idx]], [c["val_rank_ics"][best_idx]],
                    s=140, marker="*", color="#fdc500", zorder=5, edgecolor="#a86d00",
                    label=f"best ep{c['epochs'][best_idx]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss", color="#1f77b4")
        ax2.set_ylabel("Val RankIC", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)
    # 隐藏多余的子图
    for i in range(n, rows * cols):
        axes.flatten()[i].set_visible(False)
    fig.suptitle("各模型训练 Loss & Validation RankIC 曲线", fontsize=14)
    fig.tight_layout()
    fig.savefig(save, dpi=120); plt.close(fig)
    return True


def plot_v3_seeds(v3_curve, save):
    """v3 multi-seed 三条 loss + val 曲线"""
    if v3_curve is None or not v3_curve.get("seed_blocks"):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, (seed, block) in enumerate(v3_curve["seed_blocks"]):
        eps = [r[0] for r in block]
        tl = [r[1] for r in block]
        vric = [r[3] for r in block]
        c = colors[i % len(colors)]
        axes[0].plot(eps, tl, "o-", color=c, label=f"seed={seed}", lw=2)
        axes[1].plot(eps, vric, "s-", color=c, label=f"seed={seed}", lw=2)
        best_i = max(range(len(vric)), key=lambda j: vric[j])
        axes[1].scatter([eps[best_i]], [vric[best_i]], marker="*", s=140,
                        color="#fdc500", zorder=5, edgecolor="#a86d00")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss (IC loss)")
    axes[0].set_title("MASTER v3: train loss per seed"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val RankIC")
    axes[1].set_title("MASTER v3: val RankIC per seed (★=best)"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120); plt.close(fig)
    return True


def main():
    print("Parsing logs ...")
    master = parse_log(OUTPUT / "master_retrain.log")
    master_v2 = parse_log(OUTPUT / "master_v2_train.log")
    master_v3 = parse_log(OUTPUT / "master_v3_retrain.log")
    mlp = parse_mlp_json()

    print(f"  master:    {len(master['epochs']) if master else 0} epochs")
    print(f"  master_v2: {len(master_v2['epochs']) if master_v2 else 0} epochs")
    print(f"  master_v3: {sum(len(b[1]) for b in master_v3['seed_blocks']) if master_v3 else 0} epochs (multi-seed)")
    print(f"  mlp:       {len(mlp['epochs']) if mlp else 0} epochs")

    curves = {
        "MLP (基线, T*F→MLP)": mlp,
        "MASTER v1 (主模型)": master,
        "MASTER v2 (加深+加宽)": master_v2,
    }

    plot_loss_grid(curves, FIGS / "loss_curves.png")
    print(f"  saved: {FIGS / 'loss_curves.png'}")

    plot_v3_seeds(master_v3, FIGS / "loss_curves_v3.png")
    print(f"  saved: {FIGS / 'loss_curves_v3.png'}")


if __name__ == "__main__":
    main()
