# utils/kl_warmup.py
import math

def linear_warmup(ep: int, target: float, warmup_epochs: int, start: float = 0.0) -> float:
    """
    Linearly ramp from `start` to `target` over `warmup_epochs`.
    ep is 1-indexed epoch number.
    """
    if warmup_epochs <= 0:
        return float(target)
    x = min(max((ep - 1) / warmup_epochs, 0.0), 1.0)
    return float(start + (target - start) * x)

def cosine_warmup(ep: int, target: float, warmup_epochs: int, start: float = 0.0) -> float:
    """
    Cosine ramp from `start` to `target` over `warmup_epochs`.
    Smoother than linear near the start.
    """
    if warmup_epochs <= 0:
        return float(target)
    x = min(max((ep - 1) / warmup_epochs, 0.0), 1.0)   # 0..1
    w = 0.5 * (1.0 - math.cos(math.pi * x))            # 0..1
    return float(start + (target - start) * w)
