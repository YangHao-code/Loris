# coding=utf-8
"""
LORIS - Dynamic Router / Model Selection (双后端实现)
=====================================================
给定 n 个预训练模型的候选池 M，为文档集 D 选择最合适的 K 个模型，
以最大化多数投票准确率 (Majority-vote accuracy)。

核心: Hard Top-K 梯度几乎处处为零，采用 **随机平滑 (Stochastic Smoothing)**
的可导松弛，在反向传播时用蒙特卡洛采样近似 Jacobian。

提供两种可微分 Top-K 后端:
  - "custom"    : 自研 torch.autograd.Function (完全自控，便于调参)
  - "perturbed" : 复用 perturbations.py 的 perturbed_special (与 selec.py 一致)

模块组成:
  1. DifferentiableTopK   — 自研 autograd.Function (方案 A)
  2. _make_topk_layer     — 工厂函数，按 backend 创建可微分 Top-K 层
  3. SelectionNetwork      — 选择网络 g_θ (MLP → 分数 → Top-K 掩码)
  4. HybridLoss            — 混合训练目标 (imitate + task + entropy)
  5. FinalSelector          — 推理阶段全局聚合
  6. build_oracle_mask     — 构造 Oracle 伪标签
"""

import os as _os
import sys as _sys
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 路径设置: 确保项目根目录在 sys.path 中 ----------
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)


# ============================================================================
# 1. 方案 A: 自研可微分 Top-K (DifferentiableTopK)
# ============================================================================

class DifferentiableTopK(torch.autograd.Function):
    """
    基于随机平滑 (Stochastic Smoothing) 的可微分 Top-K 算子。

    前向传播:
        确定性 Hard Top-K — 得分最高的 K 个位置置 1，其余为 0。

    反向传播:
        蒙特卡洛采样近似 Jacobian:
            1. 生成 N 个高斯噪声 z^(l) ~ N(0, I)
            2. 对每个噪声计算扰动后掩码: m^(l) = TopK(s + σ·z^(l))
            3. 近似 Jacobian: G_N = (1/N) Σ [ m^(l) · (z^(l))^T ]
            4. 返回 G_N^T · grad_output 完成链式法则

    数学依据 (Stein's identity):
        ∂E[TopK(s + σz)] / ∂s ≈ (1/(Nσ)) Σ_l [ z^(l) · (m^(l) · grad)^T ]
    """

    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        k: int,
        num_samples: int,
        sigma: float,
    ) -> torch.Tensor:
        """
        前向: 确定性 Hard Top-K。

        Args:
            scores:      (B, n) 候选模型分数
            k:           选择数量
            num_samples: MC 采样次数 N (反向传播用)
            sigma:       噪声温度 σ

        Returns:
            mask: (B, n) 二值掩码
        """
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma
        ctx.save_for_backward(scores)

        _, topk_indices = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_indices, 1.0)
        return mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        反向: MC 采样近似梯度。

        实现:
            inner_l = Σ_i m_i^(l) * grad_output_i   →  (N, B)
            grad_j  = (1/(N·σ)) Σ_l z_j^(l) * inner_l  →  (B, n)
        """
        (scores,) = ctx.saved_tensors
        k, N, sigma = ctx.k, ctx.num_samples, ctx.sigma
        B, n = scores.shape
        device, dtype = scores.device, scores.dtype

        # z^(l) ~ N(0, I): (N, B, n)
        z = torch.randn(N, B, n, device=device, dtype=dtype)

        # 扰动分数 + Hard Top-K: (N, B, n)
        perturbed_scores = scores.unsqueeze(0) + sigma * z
        perturbed_flat = perturbed_scores.reshape(-1, n)
        _, topk_idx = torch.topk(perturbed_flat, k, dim=-1)
        m_flat = torch.zeros_like(perturbed_flat)
        m_flat.scatter_(-1, topk_idx, 1.0)
        m = m_flat.reshape(N, B, n)

        # einsum 高效计算梯度
        inner = torch.einsum("nbi,bi->nb", m, grad_output)   # (N, B)
        grad_scores = torch.einsum("nbj,nb->bj", z, inner)   # (B, n)
        grad_scores = grad_scores / (N * sigma)

        return grad_scores, None, None, None


# ============================================================================
# 2. 工厂函数: 按后端创建可微分 Top-K 层
# ============================================================================

def _make_topk_layer(
    k: int,
    num_samples: int = 1000,
    sigma: float = 0.1,
    device: Optional[torch.device] = None,
    backend: str = "custom",
):
    """
    工厂函数: 根据 backend 参数创建可微分 Top-K 层。

    两种后端返回值签名相同: layer(scores) -> mask
        scores: (B, n)  候选模型分数
        mask:   (B, n)  二值掩码

    Args:
        k:           选择的模型数
        num_samples: MC 采样次数
        sigma:       噪声温度
        device:      计算设备
        backend:     "custom" | "perturbed"
    """
    if backend == "custom":
        # ------ 方案 A: 自研 autograd.Function ------
        def custom_topk_layer(scores: torch.Tensor) -> torch.Tensor:
            return DifferentiableTopK.apply(scores, k, num_samples, sigma)
        return custom_topk_layer

    elif backend == "perturbed":
        # ------ 方案 B: 复用 perturbations.py ------
        import loris.selection.perturbations as _pert

        def batch_topk(scores: torch.Tensor) -> torch.Tensor:
            """闭包: 捕获 k，与 selec.py 中 batch_knapsack 相同模式。"""
            indices = torch.topk(scores, k).indices
            choice = torch.zeros_like(scores)
            choice.scatter_(
                1, indices,
                torch.ones(indices.shape, device=scores.device)
            )
            return choice

        topk_layer = _pert.perturbed_special(
            batch_topk,
            num_samples=num_samples,
            sigma=sigma,
            noise='normal',
            batched=True,
            device=device,
            hard_fwd=True,  # 前向: 确定性 Hard Top-K; 反向: MC 采样
        )
        return topk_layer

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Use 'custom' or 'perturbed'."
        )


# ============================================================================
# 3. 选择网络 (Selection Network g_θ)
# ============================================================================

class SelectionNetwork(nn.Module):
    """
    选择网络 g_θ: 文档特征 → n 个候选模型分数 → Top-K 二值掩码。

    结构: MLP (FC → ReLU → Dropout) × 2 → FC(n_models)
    路由: 调用可微分 Top-K 层输出离散掩码 m ∈ {0,1}^n。

    Args:
        input_dim:   输入特征维度 d (如 BERT 768)
        hidden_dim:  隐层维度
        n_models:    候选模型总数 n
        k:           选择的模型数 K
        num_samples: MC 采样次数 (反向传播用)
        sigma:       噪声温度 (平滑级别)
        dropout:     Dropout 比率
        device:      计算设备
        backend:     "custom" (自研) | "perturbed" (perturbations.py)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_models: int,
        k: int,
        num_samples: int = 1000,
        sigma: float = 0.1,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        backend: str = "custom",
    ):
        super().__init__()
        self.n_models = n_models
        self.k = k
        self.backend = backend

        # MLP: d → hidden → hidden → n
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_models),
        )

        # 可微分 Top-K 层 (函数, 无可学习参数)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._topk_layer = _make_topk_layer(
            k=k,
            num_samples=num_samples,
            sigma=sigma,
            device=device,
            backend=backend,
        )

    def forward(
        self,
        features: torch.Tensor,
        return_scores: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            features:      (B, d) 文档特征向量
            return_scores:  若为 True，同时返回原始分数

        Returns:
            mask:   (B, n) 二值掩码
            scores: (B, n) 原始分数 (仅当 return_scores=True)
        """
        scores = self.mlp(features)  # (B, n)
        mask = self._topk_layer(scores)
        if return_scores:
            return mask, scores
        return mask

    def get_scores(self, features: torch.Tensor) -> torch.Tensor:
        """仅返回原始分数 (不经过 Top-K)，用于单独计算 Loss。"""
        return self.mlp(features)


# ============================================================================
# 4. 混合训练目标 (Hybrid Training Objective)
# ============================================================================

class HybridLoss(nn.Module):
    """
    混合训练目标:
        L_total = L_imitate + λ_task · L_task − λ_ent · L_ent

    各分项:
        L_imitate : Weighted BCE — 预测分数 s 与 Oracle 掩码之间的损失。
                    Oracle 掩码指示哪些模型对当前文档最合适 (伪标签)。
        L_task    : Cross-Entropy — 被选中模型多数投票聚合预测 vs 真实标签。
                    直接优化下游任务性能。
        L_ent     : Entropy 正则化 — 鼓励探索 (最大化熵 → 分散选择)。
                    公式中用减号: -λ_ent · L_ent。

    Args:
        lambda_task: L_task 的权重
        lambda_ent:  L_ent 的权重
        pos_weight:  BCE 正样本权重, 建议设为 n/K - 1 以平衡正负样本
    """

    def __init__(
        self,
        lambda_task: float = 1.0,
        lambda_ent: float = 0.1,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.lambda_task = lambda_task
        self.lambda_ent = lambda_ent

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.ce = nn.CrossEntropyLoss()

    def imitation_loss(
        self, scores: torch.Tensor, oracle_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        L_imitate: Weighted Binary Cross Entropy.

        将 scores (logits) 与 oracle_mask (0/1 伪标签) 做 BCE。
        Oracle 掩码标识对每个文档的最优 K 个模型。

        Args:
            scores:      (B, n) 选择网络原始输出 (logits)
            oracle_mask: (B, n) Ground Truth 伪标签 {0, 1}
        """
        return self.bce(scores, oracle_mask)

    def task_loss(
        self,
        mask: torch.Tensor,
        model_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_task: 被选中模型多数投票后与真实标签的 Cross-Entropy。

        聚合: 对选中模型的 softmax 概率做加权平均 (权重为 mask),
              然后与真实标签计算 NLL Loss。

        Args:
            mask:         (B, n)      选择掩码
            model_logits: (n, B, C)   所有候选模型的 logits
            labels:       (B,)        真实标签
        """
        # 各模型 softmax 概率: (n, B, C) → (B, n, C)
        model_probs = F.softmax(model_logits, dim=-1).permute(1, 0, 2)

        # 加权聚合: mask (B, n, 1) * probs (B, n, C) → sum → (B, C)
        mask_expanded = mask.unsqueeze(-1)  # (B, n, 1)
        weighted = (model_probs * mask_expanded).sum(dim=1)  # (B, C)
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        ensemble_probs = weighted / denom  # (B, C)

        # NLL Loss (数值稳定)
        log_probs = torch.log(ensemble_probs + 1e-8)
        return F.nll_loss(log_probs, labels)

    def entropy_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        L_ent: 对分数 softmax 后计算熵正则化。

        H(p) = -Σ_i p_i log(p_i)
        最大化熵 → 探索; 最小化熵 → 确定性。
        公式中用 -λ_ent · H，即鼓励探索。

        Args:
            scores: (B, n) 选择网络原始输出
        """
        p = F.softmax(scores, dim=-1)
        log_p = F.log_softmax(scores, dim=-1)
        entropy = -(p * log_p).sum(dim=-1)  # (B,)
        return entropy.mean()

    def forward(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        oracle_mask: torch.Tensor,
        model_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算混合损失:
            L_total = L_imitate + λ_task · L_task − λ_ent · L_ent

        Args:
            scores:       (B, n)    选择网络原始输出
            mask:         (B, n)    Top-K 二值掩码
            oracle_mask:  (B, n)    Oracle 伪标签
            model_logits: (n, B, C) 候选模型 logits
            labels:       (B,)      真实标签

        Returns:
            total_loss: 标量
            details:    各分项损失字典 (用于日志)
        """
        l_imit = self.imitation_loss(scores, oracle_mask)
        l_task = self.task_loss(mask, model_logits, labels)
        l_ent = self.entropy_loss(scores)

        total = l_imit + self.lambda_task * l_task - self.lambda_ent * l_ent

        details = {
            "L_imitate": l_imit.item(),
            "L_task": l_task.item(),
            "L_ent": l_ent.item(),
            "L_total": total.item(),
        }
        return total, details


# ============================================================================
# 5. 最终模型聚合 (Final Selection)
# ============================================================================

class FinalSelector:
    """
    推理阶段: 遍历文档集 D，统计每个模型被 Top-K 选中的频次，
    返回全局频次最高的 K 个模型作为最终集合 S。

    流程:
        1. 对每篇文档 d_i → SelectionNetwork → mask_i ∈ {0,1}^n
        2. frequency[j] = Σ_i mask_i[j]
        3. S = Top-K(frequency)
    """

    @staticmethod
    @torch.no_grad()
    def select(
        selection_net: SelectionNetwork,
        features: torch.Tensor,
        k: int,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Args:
            selection_net: 训练好的选择网络
            features:      (D, d) 文档集所有文档的特征
            k:             最终选择的模型数

        Returns:
            selected_indices: 被选中模型的索引列表 (长度 K)
            frequency:        (n,) 每个模型的被选频次
        """
        selection_net.eval()
        masks = selection_net(features)  # (D, n)
        frequency = masks.sum(dim=0)     # (n,)
        _, top_indices = torch.topk(frequency, k)
        return top_indices.cpu().tolist(), frequency


# ============================================================================
# 6. 辅助: 构造 Oracle 掩码
# ============================================================================

def build_oracle_mask(
    model_logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    根据 Ground Truth 标签构造 Oracle 掩码 (伪标签)。

    策略: 对每个样本，找出在真实类别上 softmax 概率最高的 K 个模型，
    将对应位置置 1。

    Args:
        model_logits: (n, B, C) 所有候选模型的 logits
        labels:       (B,)      真实标签
        k:            选择数量

    Returns:
        oracle_mask: (B, n) 伪标签 {0, 1}
    """
    n, B, C = model_logits.shape

    # 各模型在真实类别上的概率
    probs = F.softmax(model_logits, dim=-1)                       # (n, B, C)
    label_exp = labels.unsqueeze(0).unsqueeze(-1).expand(n, B, 1) # (n, B, 1)
    gt_probs = torch.gather(probs, dim=-1, index=label_exp).squeeze(-1)  # (n, B)

    # 转置 + Top-K
    gt_probs_t = gt_probs.t()                  # (B, n)
    _, topk_idx = torch.topk(gt_probs_t, k, dim=-1)  # (B, k)

    oracle_mask = torch.zeros(B, n, device=model_logits.device)
    oracle_mask.scatter_(-1, topk_idx, 1.0)
    return oracle_mask


# ============================================================================
# 7. 测试流程
# ============================================================================

if __name__ == "__main__":
    # ----- 超参数 -----
    n_models = 10         # 候选模型数 n
    K = 3                 # 选择的模型数
    N_mc = 10             # 蒙特卡洛采样次数 (custom 方案用小值以便快速测试)
    sigma = 0.5           # 噪声温度
    input_dim = 768       # 文档特征维度
    hidden_dim = 256      # 选择网络隐层维度
    num_classes = 4       # 分类任务类别数
    batch_size = 16       # 批大小
    num_docs = 100        # 文档集大小 (Final Selection 测试)
    lr = 1e-3             # 学习率
    lambda_task = 1.0
    lambda_ent = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: n={n_models}, K={K}, N_mc={N_mc}, sigma={sigma}")
    print(f"        d={input_dim}, hidden={hidden_dim}, C={num_classes}, B={batch_size}")

    # ----- 模拟数据 (两个后端共用) -----
    torch.manual_seed(42)
    doc_features = torch.randn(batch_size, input_dim, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    model_logits = torch.randn(n_models, batch_size, num_classes, device=device)
    all_doc_features = torch.randn(num_docs, input_dim, device=device)

    # ========== 依次测试两种后端 ==========
    backends = ["custom", "perturbed"]
    for backend in backends:
        print("\n" + "=" * 70)
        print(f"  Backend: {backend}")
        print("=" * 70)

        # 实例化
        torch.manual_seed(42)  # 同样的初始化权重
        selection_net = SelectionNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_models=n_models,
            k=K,
            num_samples=N_mc,
            sigma=sigma,
            device=device,
            backend=backend,
        ).to(device)

        criterion = HybridLoss(
            lambda_task=lambda_task,
            lambda_ent=lambda_ent,
            pos_weight=n_models / K - 1,
        ).to(device)

        optimizer = torch.optim.Adam(selection_net.parameters(), lr=lr)

        # --- Oracle 掩码 ---
        oracle_mask = build_oracle_mask(model_logits, labels, K)
        print(f"\nOracle mask (sample 0): {oracle_mask[0].tolist()}")

        # --- 训练迭代 ---
        print(f"\n[Train] 前向 + 混合 Loss + 反向传播")
        selection_net.train()
        optimizer.zero_grad()

        mask, scores = selection_net(doc_features, return_scores=True)
        print(f"  Mask (sample 0):   {mask[0].tolist()}")
        print(f"  Models/sample:     {mask.sum(dim=-1).tolist()}")

        total_loss, details = criterion(
            scores=scores,
            mask=mask,
            oracle_mask=oracle_mask,
            model_logits=model_logits,
            labels=labels,
        )
        print(f"  Loss: {details}")

        total_loss.backward()

        # 检查梯度
        print(f"  Gradient norms:")
        for name, param in selection_net.named_parameters():
            if param.grad is not None:
                print(f"    {name}: {param.grad.norm().item():.6f}")

        optimizer.step()
        print(f"  Optimizer step OK.")

        # --- Final Selection ---
        print(f"\n[Inference] Final Selection")
        selected, freq = FinalSelector.select(selection_net, all_doc_features, K)
        print(f"  频次: {freq.cpu().tolist()}")
        print(f"  全局 Top-{K}: {selected}")

    print("\n" + "=" * 70)
    print("All backends tested successfully!")
    print("=" * 70)
