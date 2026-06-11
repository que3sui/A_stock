"""
共享损失函数: Pearson IC loss + Top-K margin loss

从 code.models.master 提取, 消除 gru_att/mlp/master_v2 中的重复定义。
"""
import torch


def ic_loss(scores, labels):
    """Negative Pearson correlation (listwise ranking loss)"""
    s = scores - scores.mean()
    l = labels - labels.mean()
    num = (s * l).sum()
    den = torch.sqrt((s ** 2).sum() * (l ** 2).sum() + 1e-12)
    return -num / den


def topk_margin_loss(scores, labels, k_ratio=0.2, margin=0.1):
    """Top-K margin: 让真实 top-K 的预测分 > 真实 bottom-K 的预测分"""
    n = scores.size(0)
    k = max(int(n * k_ratio), 5)
    _, top_idx = torch.topk(labels, k)
    _, bot_idx = torch.topk(-labels, k)
    s_top = scores[top_idx].unsqueeze(1)  # [k, 1]
    s_bot = scores[bot_idx].unsqueeze(0)  # [1, k]
    margin_loss = torch.clamp(s_bot - s_top + margin, min=0)
    return margin_loss.mean()


def combined_loss(scores, labels, alpha=0.6):
    return alpha * ic_loss(scores, labels) + (1 - alpha) * topk_margin_loss(scores, labels)
