"""
交易日志: 记录每笔决策的原因和结果

Usage:
  from code.live.trade_journal import log_entry
  log_entry(date, plan, action, stocks, reason, metadata)
"""
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
JOURNAL_DIR = ROOT / "output" / "journal"
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)


def log_entry(date, plan, action, stocks, reason, metadata=None):
    """记录一笔交易日志

    Args:
        date: 交易日期
        plan: A/B/C
        action: buy/sell/rebalance/stop
        stocks: [(ts_code, name, lots, price), ...]
        reason: 决策原因 (e.g. "模型top-8等权建仓")
        metadata: dict of extra info (model_version, scores, etc.)
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "date": date,
        "plan": plan,
        "action": action,
        "stocks": [{"code": s[0], "name": s[1], "lots": s[2], "price": s[3]} for s in stocks],
        "reason": reason,
        "metadata": metadata or {},
    }

    filepath = JOURNAL_DIR / f"journal_{date}.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_journal(date=None, plan=None, limit=20):
    """读取日志"""
    if date:
        filepath = JOURNAL_DIR / f"journal_{date}.jsonl"
        if not filepath.exists():
            return []
        entries = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line.strip())
                if plan is None or e.get("plan") == plan:
                    entries.append(e)
        return entries[-limit:]
    else:
        # Read all journals
        entries = []
        for f in sorted(JOURNAL_DIR.glob("journal_*.jsonl"))[-10:]:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    e = json.loads(line.strip())
                    if plan is None or e.get("plan") == plan:
                        entries.append(e)
        return entries[-limit:]


def performance_summary(days=10):
    """从日志生成简要绩效"""
    entries = read_journal(limit=1000)
    if not entries:
        return "无日志记录"

    plans = {}
    for e in entries:
        p = e["plan"]
        plans.setdefault(p, {"buys": 0, "sells": 0, "dates": set()})
        plans[p]["buys" if e["action"] == "buy" else "sells"] += 1
        plans[p]["dates"].add(e["date"])

    lines = []
    for p, stats in plans.items():
        lines.append(f"  方案{p}: {len(stats['dates'])}天, "
                     f"{stats['buys']}买/{stats['sells']}卖")
    return "\n".join(lines) if lines else "无交易记录"
