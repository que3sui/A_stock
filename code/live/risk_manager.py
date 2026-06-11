"""
风险管理: 止损, 仓位限制, 回撤熔断

三条铁律:
  1. 单方案周亏损 > 5% → 清仓 (止损)
  2. 单方案总回撤 > 20% → 清仓 (熔断)
  3. 单日全市场波动 > 3σ → 减半仓 (避险)

Usage:
  from code.live.risk_manager import check_risk, RiskState
  state = check_risk(plan_name, daily_ret, cumulative_dd, market_vol_z)
  if state == RiskState.NORMAL: ...
"""
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
RISK_FILE = OUTPUT / "monitor" / "risk_state.json"

WEEKLY_LOSS_LIMIT = -0.05   # 周亏损 > 5% → 清仓
TOTAL_DD_LIMIT = -0.20      # 总回撤 > 20% → 熔断
VOL_Z_THRESHOLD = 2.0       # 市场波动 > 2σ → 减仓


class RiskState(Enum):
    NORMAL = "normal"           # 正常
    REDUCE = "reduce"           # 减半仓 (高波动)
    STOP = "stop"               # 清仓 (止损/熔断)
    PAUSED = "paused"           # 暂停 (手动)


@dataclass
class RiskCheck:
    state: RiskState
    reason: str
    suggested_n: int  # 建议持仓数


def _load_state():
    if RISK_FILE.exists():
        return json.load(open(RISK_FILE))
    return {"plans": {}}


def _save_state(state):
    RISK_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(state, open(RISK_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def check_risk(plan_name, daily_ret, cumulative_dd, market_vol_z=0.0, weekly_returns=None):
    """
    综合风控检查

    Args:
        plan_name: A/B/C
        daily_ret: 当日收益率 (e.g. -0.03 = -3%)
        cumulative_dd: 累计最大回撤 (负数)
        market_vol_z: 市场波动 z-score
        weekly_returns: 最近5个交易日收益率列表 (用于周止损)

    Returns:
        RiskCheck: 当前状态和建议
    """
    state = _load_state()
    ps = state["plans"].get(plan_name, {"paused": False, "stop_triggered": False,
                                         "stop_date": None, "history": []})

    # 1. 手动暂停
    if ps.get("paused"):
        return RiskCheck(RiskState.PAUSED, "手动暂停", 0)

    # 2. 已触发止损 → 保持清仓状态
    if ps.get("stop_triggered"):
        return RiskCheck(RiskState.STOP,
                         f"止损触发于 {ps.get('stop_date', '?')}", 0)

    # 3. 周度止损检查
    if weekly_returns and len(weekly_returns) >= 5:
        week_ret = sum(weekly_returns[-5:])
        if week_ret < WEEKLY_LOSS_LIMIT:
            ps["stop_triggered"] = True
            ps["stop_date"] = str(ps.get("last_date", "?"))
            state["plans"][plan_name] = ps
            _save_state(state)
            return RiskCheck(RiskState.STOP,
                             f"周亏损 {week_ret*100:.1f}% > {WEEKLY_LOSS_LIMIT*100:.0f}%", 0)

    # 4. 总回撤熔断
    if cumulative_dd < TOTAL_DD_LIMIT:
        ps["stop_triggered"] = True
        ps["stop_date"] = str(ps.get("last_date", "?"))
        state["plans"][plan_name] = ps
        _save_state(state)
        return RiskCheck(RiskState.STOP,
                         f"总回撤 {cumulative_dd*100:.1f}% > {TOTAL_DD_LIMIT*100:.0f}%", 0)

    # 5. 高波动减仓
    if market_vol_z > VOL_Z_THRESHOLD:
        return RiskCheck(RiskState.REDUCE,
                         f"市场波动 {market_vol_z:.1f}σ > {VOL_Z_THRESHOLD}σ", n_reduce=4)

    # 6. 正常
    return RiskCheck(RiskState.NORMAL, "正常", suggested_n=8)


def pause_plan(plan_name):
    """手动暂停方案"""
    state = _load_state()
    state["plans"].setdefault(plan_name, {})["paused"] = True
    _save_state(state)


def resume_plan(plan_name):
    """手动恢复方案"""
    state = _load_state()
    if plan_name in state["plans"]:
        state["plans"][plan_name]["paused"] = False
        state["plans"][plan_name]["stop_triggered"] = False
        _save_state(state)


def get_risk_report():
    """生成风控报告"""
    state = _load_state()
    report = []
    for plan, ps in state.get("plans", {}).items():
        flags = []
        if ps.get("paused"): flags.append("暂停")
        if ps.get("stop_triggered"): flags.append(f"止损({ps.get('stop_date','?')})")
        report.append(f"  方案{plan}: {', '.join(flags) if flags else '正常'}")
    return "\n".join(report) if report else "  所有方案正常"
