from typing import Optional

import torch
import torch.nn as nn


class RouterShiftRatio(nn.Module):

    def __init__(
        self,
        num_experts: int,
        gamma_min: float = 0.1,
        momentum: float = 0.9,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gamma_min = gamma_min
        self.momentum = momentum
        self.eps = eps

        # 历史路由分布缓冲区, 以均匀分布初始化
        init_dist = torch.full((num_experts,), 1.0 / num_experts)
        self.register_buffer("r_old", init_dist, persistent=False)

    @torch.no_grad()
    def _update_history(self, r_current: torch.Tensor):
        self.r_old.mul_(self.momentum).add_(r_current * (1 - self.momentum))

    def forward(self, token_to_expert_probs: torch.Tensor) -> torch.Tensor:  # type: ignore
 
        # 当前分布 r_current
        token_counts = token_to_expert_probs.sum(dim=0)  # [num_experts]
        if torch.distributed.is_initialized():
            # 同步不同进程, 保证全局分布一致
            torch.distributed.all_reduce(token_counts, op=torch.distributed.ReduceOp.SUM)
        r_current = token_counts / (token_counts.sum() + self.eps)

        # 计算 γ
        abs_log_diff = (r_current.clamp_min(self.eps).log() - self.r_old.clamp_min(self.eps).log()).abs()
        gamma = torch.exp(-abs_log_diff.mean())
        gamma = torch.clamp(gamma, min=self.gamma_min)

        # 更新历史分布
        self._update_history(r_current)

        return gamma.detach()
