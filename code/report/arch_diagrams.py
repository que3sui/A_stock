"""
项目架构图与数据流程图 (报告专业度提升):
  - data_flow.png: 数据 → 特征 → 标签 → 模型 → 回测 → 信号 全流程
  - master_arch.png: MASTER 模型架构 (Market gate + Intra TX + Inter TX + Head)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "output" / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, color="#cce5ff", edge="#2266cc", fontsize=10):
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def arrow(ax, x1, y1, x2, y2, label="", color="#444", style="->", text_offset=(0, 0)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5))
    if label:
        ax.text((x1 + x2) / 2 + text_offset[0], (y1 + y2) / 2 + text_offset[1],
                label, ha="center", fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


def plot_data_flow():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15); ax.set_ylim(0, 10)
    ax.axis("off")

    # 第一层: 原始数据
    box(ax, 0.3, 8.3, 2, 1.2, "basic.csv\ntrade_cal.csv", "#fff3cd", "#d4a000")
    box(ax, 2.8, 8.3, 2, 1.2, "daily/\n(量价 2515天)", "#fff3cd", "#d4a000")
    box(ax, 5.3, 8.3, 2, 1.2, "metric/\n(基本面)", "#fff3cd", "#d4a000")
    box(ax, 7.8, 8.3, 2, 1.2, "moneyflow/\n(资金流)", "#fff3cd", "#d4a000")
    box(ax, 10.3, 8.3, 2, 1.2, "stock_st/\n(ST 名单)", "#fff3cd", "#d4a000")
    box(ax, 12.8, 8.3, 2, 1.2, "market/\n(三大指数)", "#fff3cd", "#d4a000")

    # 第二层: 整合
    box(ax, 1, 6.5, 9, 1.0, "build_panel.py: 整合 daily + metric + moneyflow + stock_st + basic → panel.parquet (1.07G, 1070万行)", "#cfe2ff", "#2266cc")
    box(ax, 11, 6.5, 3.5, 1.0, "universe.py: 月度中证800 (circ_mv top800)", "#cfe2ff", "#2266cc")

    # 第三层: 特征 / 标签
    box(ax, 0.3, 4.5, 3.2, 1.3,
        "factors.py\n20 维原始因子\n(动量4+反转2+波动2+\n流动性2+资金流3+\n基本面3+技术4)",
        "#d4edda", "#28a745", fontsize=9)
    box(ax, 4, 4.5, 3, 1.3,
        "neutralize.py\nMAD去极值 → 行业dummy+\nlog_mv OLS 残差 → Z-score\n(按日横截面)",
        "#d4edda", "#28a745", fontsize=9)
    box(ax, 7.5, 4.5, 3, 1.3,
        "labels.py\n5 日 forward log return\n→ 横截面 rank [-0.5, 0.5]",
        "#d4edda", "#28a745", fontsize=9)
    box(ax, 11, 4.5, 3.5, 1.3,
        "market_features.py\n3 指数 × 4 维 = 12 维\n(train 段 Z-score)",
        "#d4edda", "#28a745", fontsize=9)

    # 第四层: 模型
    box(ax, 0.3, 2.5, 2, 1.3, "LightGBM\n(基线树)", "#f8d7da", "#dc3545")
    box(ax, 2.5, 2.5, 2, 1.3, "MLP\n(DL 基线)", "#f8d7da", "#dc3545")
    box(ax, 4.7, 2.5, 2, 1.3, "GRU+Att\n(时序)", "#f8d7da", "#dc3545")
    box(ax, 6.9, 2.5, 2.4, 1.3, "MASTER v1\n(主模型, 123K)", "#f1a7af", "#a02030", fontsize=11)
    box(ax, 9.5, 2.5, 2, 1.3, "v2 (加深)\n反例", "#f8d7da", "#dc3545")
    box(ax, 11.7, 2.5, 2.4, 1.3, "v3 (multi-seed)\n减方差", "#f8d7da", "#dc3545")

    # 第五层: 回测 / 输出
    box(ax, 0.3, 0.5, 5, 1.3,
        "backtest/engine.py\n真实约束: T+1 + 涨跌停 + ST 剔除\n+ 双边 0.025% 佣金 + 卖出 0.1% 印花税",
        "#e2d9f3", "#6f42c1", fontsize=9)
    box(ax, 5.5, 0.5, 4, 1.3,
        "report/\nbuild_report + IC 分段 +\nloss curves + sensitivity + 归因",
        "#e2d9f3", "#6f42c1", fontsize=9)
    box(ax, 10, 0.5, 4.5, 1.3,
        "live/daily_signal.py\n模拟交易当日清单\n(n=10/k=2 默认, 等权满仓)",
        "#e2d9f3", "#6f42c1", fontsize=9)

    # 箭头
    for x_top in [1.3, 3.8, 6.3, 8.8, 11.3, 13.8]:
        arrow(ax, x_top, 8.3, x_top - 1 + 5.5, 7.5, color="#999")
    arrow(ax, 5.5, 6.5, 1.9, 5.8, color="#666")
    arrow(ax, 5.5, 6.5, 5.5, 5.8, color="#666")
    arrow(ax, 5.5, 6.5, 9, 5.8, color="#666")
    arrow(ax, 12.7, 6.5, 12.7, 5.8, color="#666")
    # 中间层 → 模型
    for x_mid in [1.3, 3.5, 5.7, 8.1, 10.5, 12.9]:
        arrow(ax, 6, 4.5, x_mid, 3.8, color="#999", style="->")
    arrow(ax, 12.7, 4.5, 8.1, 3.8, label="market gate (仅 MASTER)",
          color="#a02030", text_offset=(-1.5, 0.3))
    # 模型 → 回测
    arrow(ax, 8.1, 2.5, 2.8, 1.8, color="#666")
    arrow(ax, 8.1, 2.5, 7.5, 1.8, color="#666")
    arrow(ax, 8.1, 2.5, 12.2, 1.8, color="#666")

    ax.set_title("A 股趋势预测与模拟交易: 数据流程图", fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(FIGS / "data_flow.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGS / 'data_flow.png'}")


def plot_master_arch():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13); ax.set_ylim(0, 11)
    ax.axis("off")

    # Inputs
    box(ax, 0.5, 9.3, 5, 1.0, "X: 股票特征 [N stocks × T=20 天 × F=20 因子]",
        "#fff3cd", "#d4a000", fontsize=11)
    box(ax, 7, 9.3, 5, 1.0, "M: 市场状态 [T=20 天 × Fm=12 维]",
        "#ffd6cc", "#cc6644", fontsize=11)

    # Market guidance
    box(ax, 7, 7.5, 5, 1.0, "市场投影: M → Linear → [T, H]",
        "#ffd6cc", "#cc6644", fontsize=10)
    box(ax, 7, 5.7, 5, 1.0, "门控: sigmoid(Linear(H → F)) → gate [T, F]",
        "#ffd6cc", "#cc6644", fontsize=10)

    # 主干: gate × X
    box(ax, 0.5, 7.5, 5, 1.0, "门控调制: X = X ⊙ gate (逐时序 broadcast)",
        "#cce5ff", "#2266cc", fontsize=10)

    # Stock projection + positional encoding
    box(ax, 0.5, 6.0, 5, 1.0, "股票投影: X → Linear(F→H=64) + Pos Enc",
        "#cce5ff", "#2266cc", fontsize=10)

    # Intra-stock TX
    box(ax, 0.5, 4.3, 5, 1.2,
        "Intra-stock Transformer × 2 层\n(每只股 self-attention 时序轴 → [N, T, H])",
        "#a8d4f8", "#1144aa", fontsize=10)

    # Temporal aggregation
    box(ax, 0.5, 2.8, 5, 1.0,
        "时间聚合: Query attention pooling → [N, H]",
        "#a8d4f8", "#1144aa", fontsize=10)

    # Inter-stock TX
    box(ax, 0.5, 1.3, 5, 1.2,
        "Inter-stock Transformer × 1 层\n(跨股票 self-attention 横截面轴 → [N, H])",
        "#7fb8ed", "#0a3580", fontsize=10)

    # Residual + Head
    box(ax, 6, 2.5, 6.5, 1.0,
        "Residual: Z_final = Z_temporal + Z_inter   →  Head: LayerNorm + MLP → score [N]",
        "#d4edda", "#28a745", fontsize=10)

    # 输出
    box(ax, 6, 0.3, 6.5, 1.0,
        "横截面 score 排序 → top-K 选股 → 5 日预测",
        "#fce4ec", "#c2185b", fontsize=11)

    # Loss
    box(ax, 0.5, 0.0, 5, 0.8,
        "Loss = 0.6 · IC loss + 0.4 · Top-K margin",
        "#e2d9f3", "#6f42c1", fontsize=10)

    # 箭头
    arrow(ax, 3, 9.3, 3, 8.5, color="#444")
    arrow(ax, 9.5, 9.3, 9.5, 8.5, color="#cc6644")
    arrow(ax, 9.5, 7.5, 9.5, 6.7, color="#cc6644")
    arrow(ax, 9.5, 5.7, 5.5, 8.0, color="#cc6644", label="modulate")  # gate → mul

    arrow(ax, 3, 7.5, 3, 7.0, color="#444")
    arrow(ax, 3, 6.0, 3, 5.5, color="#444")
    arrow(ax, 3, 4.3, 3, 3.8, color="#444")
    arrow(ax, 3, 2.8, 3, 2.5, color="#444")
    arrow(ax, 5.5, 1.8, 6.0, 2.7, color="#444")
    arrow(ax, 5.5, 3.3, 6.0, 3.0, color="#444", label="residual")
    arrow(ax, 9.5, 2.5, 9.5, 1.3, color="#444")
    arrow(ax, 5.5, 0.4, 6.0, 0.7, label="train signal", color="#6f42c1")

    ax.set_title("MASTER 模型架构 (AAAI 2024 简化版)\n"
                 "Market-guided gating + Intra-stock TX (时序) + Inter-stock TX (横截面) + 双 Listwise loss",
                 fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(FIGS / "master_arch.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGS / 'master_arch.png'}")


if __name__ == "__main__":
    plot_data_flow()
    plot_master_arch()
