# src/sparsemax.py
import torch

def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation (Martins & Astudillo, 2016).
    Returns a probability distribution with exact zeros.
    logits: (..., K)
    """
    z = logits
    z = z - z.max(dim=dim, keepdim=True).values  # stability

    # sort descending
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    k = z.size(dim)
    # 1..K
    r = torch.arange(1, k + 1, device=z.device, dtype=z.dtype).view(
        *([1] * (z.dim() - 1)), k
    )
    # cumsum
    z_cum = z_sorted.cumsum(dim)

    # determine k(z)
    # support: 1 + r*z_sorted > z_cum
    support = (1 + r * z_sorted) > z_cum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)

    # compute tau
    # tau = (sum_{j<=k} z_j - 1)/k
    z_cum_k = z_cum.gather(dim, k_z - 1)
    tau = (z_cum_k - 1) / k_z.to(z.dtype)

    # sparsemax
    p = torch.clamp(z - tau, min=0.0)
    return p
