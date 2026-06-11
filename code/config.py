"""
共享常量与配置 — 所有模块的单一数据源

FACTOR_COLS / WEIGHT_COLS 的规范定义在此, 其他文件从本模块 import,
不再各自复制字面量列表。新增因子只需修改此处。
"""
from pathlib import Path

# === 路径 ===
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"

# === 基础因子 (20个) ===
# 动量(4) + 反转(2) + 波动率(2) + 流动性(2) + 资金流(3) + 基本面(3) + 技术(4)
FACTOR_COLS = [
    "mom_5", "mom_20", "mom_60", "mom_120",
    "rev_1", "rev_5",
    "vol_20", "vol_60",
    "turnover_20", "amihud_20",
    "mf_net_5", "mf_lg_strength", "mf_elg_strength",
    "pe_ttm_rank", "pb_rank", "circ_mv_log",
    "rsi_14", "bias_20", "vwap_dev", "vol_zscore",
]

# === 权重通道因子 (不经过中性化, 穿透原始值) ===
WEIGHT_COLS = ["hs300_weight", "hs300_dweight", "cyb_weight"]

# === 行业轮动因子 (v4+) ===
INDUSTRY_FACTORS = ["ind_mom_20", "ind_rel_str"]

# === 数据切分 ===
T = 20                      # 回溯窗口 (交易日)
TRAIN_MAX = 20221231        # 训练段结束
VALID_MAX = 20231231        # 验证段结束

# === 派生 ===
N_FEAT = len(FACTOR_COLS)   # 20
N_WEIGHT = len(WEIGHT_COLS) # 3

# === 扩展因子集 (v4+) ===
FACTOR_COLS_V4 = FACTOR_COLS + INDUSTRY_FACTORS
N_FEAT_V4 = len(FACTOR_COLS_V4)

# === MASTER 模型默认超参数 ===
MASTER_H = 64
MASTER_NHEAD = 4
MASTER_DROPOUT = 0.2
MASTER_N_INTRA_LAYERS = 2
MASTER_N_INTER_LAYERS = 1
MASTER_LR = 5e-4
MASTER_WD = 1e-3
MASTER_ALPHA = 0.6
MASTER_TOPK_MARGIN = 0.1
