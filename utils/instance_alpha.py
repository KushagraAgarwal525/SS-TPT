"""Per-image stability–suitability trade-off α(x).

No spatial term, dataset logic, or attack logic. Person 3 adds γ * C_spatial
outside ``combine_ss``.
"""

import math

import torch

ALPHA_MODES = (
    "fixed",
    "margin",
    "entropy",
    "variance",
    "topk",
    "separation",
    "agreement",
)


def compute_instance_alpha(
    suit_n,
    stab_n,
    *,
    mode,
    fixed_alpha,
    temp=0.25,
    topk=3,
    eps=1e-8,
    suit_raw=None,
    stab_raw=None,
):
    """Return a 0-dim tensor on suit_n.device, detached, in [0, 1]."""
    alpha, _, _ = compute_instance_alpha_details(
        suit_n,
        stab_n,
        mode=mode,
        fixed_alpha=fixed_alpha,
        temp=temp,
        topk=topk,
        eps=eps,
        suit_raw=suit_raw,
        stab_raw=stab_raw,
    )
    return alpha


def compute_instance_alpha_details(
    suit_n,
    stab_n,
    *,
    mode,
    fixed_alpha,
    temp=0.25,
    topk=3,
    eps=1e-8,
    suit_raw=None,
    stab_raw=None,
):
    """Same as ``compute_instance_alpha``, plus the two reliabilities.

    Returns ``(alpha, r_suit, r_stab)``, each a detached 0-dim tensor.
    ``mode='fixed'`` ignores the vectors, so both reliabilities are 0.
    ``mode='agreement'`` reports the inner margin reliabilities.
    """
    if mode not in ALPHA_MODES:
        raise ValueError(f"Unsupported alpha mode: {mode}")
    if suit_n.ndim != 1 or stab_n.ndim != 1:
        raise ValueError("suit_n and stab_n must be 1-D")
    if suit_n.shape != stab_n.shape:
        raise ValueError("suit_n and stab_n must have the same shape")

    suit_n = suit_n.detach()
    stab_n = stab_n.detach()
    if suit_raw is not None:
        suit_raw = suit_raw.detach()
        if suit_raw.shape != suit_n.shape:
            raise ValueError("suit_raw must match suit_n")
    if stab_raw is not None:
        stab_raw = stab_raw.detach()
        if stab_raw.shape != stab_n.shape:
            raise ValueError("stab_raw must match stab_n")

    fixed = _scalar_on(fixed_alpha, suit_n).clamp(0, 1)
    zero = suit_n.new_zeros(())
    if mode == "fixed" or suit_n.numel() == 0:
        return fixed, zero, zero

    if _is_flat(suit_raw, suit_n, eps):
        r_suit = zero
    else:
        r_suit = _reliability(suit_n, mode=mode, temp=temp, topk=topk, eps=eps)
    if _is_flat(stab_raw, stab_n, eps):
        r_stab = zero
    else:
        r_stab = _reliability(stab_n, mode=mode, temp=temp, topk=topk, eps=eps)

    r_suit = torch.clamp(r_suit.reshape(()).to(dtype=suit_n.dtype), min=0)
    r_stab = torch.clamp(r_stab.reshape(()).to(dtype=suit_n.dtype), min=0)

    if float((r_suit + r_stab).item()) < eps:
        return fixed, r_suit.detach(), r_stab.detach()

    if mode == "agreement":
        rho = _spearman(suit_n, stab_n, eps).to(dtype=suit_n.dtype)
        rho_pos = torch.clamp(rho, min=0)
        alpha_margin = r_suit / (r_suit + r_stab + eps)
        alpha = rho_pos * fixed + (1.0 - rho_pos) * alpha_margin
    else:
        alpha = r_suit / (r_suit + r_stab + eps)

    alpha = alpha.clamp(0, 1).detach().reshape(())
    return alpha, r_suit.detach(), r_stab.detach()


def combine_ss(suit_n, stab_n, alpha):
    """Linear SS mix. ``alpha`` is a scalar and broadcasts over views."""
    return alpha * suit_n + (1.0 - alpha) * stab_n


def _scalar_on(value, like):
    if torch.is_tensor(value):
        return value.detach().to(device=like.device, dtype=like.dtype).reshape(())
    return like.new_tensor(float(value))


def _is_flat(raw, normalized, eps):
    src = normalized if raw is None else raw
    if src.numel() == 0:
        return True
    return float((src.max() - src.min()).item()) < eps


def _reliability(score, *, mode, temp, topk, eps):
    if mode in ("margin", "agreement"):
        return _margin(score)
    if mode == "entropy":
        return _entropy_reliability(score, temp, eps)
    if mode == "variance":
        return _variance(score)
    if mode == "topk":
        return _topk_reliability(score, temp, topk, eps)
    if mode == "separation":
        return _separation(score)
    raise ValueError(f"Unsupported alpha mode: {mode}")


def _margin(score):
    if score.numel() < 2:
        return score.new_zeros(())
    top2 = torch.topk(score, k=2).values
    return (top2[0] - top2[1]).reshape(())


def _entropy_reliability(score, temp, eps):
    n = score.numel()
    if n <= 1:
        return score.new_zeros(())
    temp_v = float(temp) if float(temp) > 0 else float(eps)
    probs = torch.softmax(score / temp_v, dim=0)
    log_p = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
    entropy = -(probs * log_p).sum()
    return (1.0 - entropy / math.log(n)).reshape(())


def _topk_reliability(score, temp, topk, eps):
    n = score.numel()
    k_eff = min(max(int(topk), 0), n)
    if k_eff == 0:
        return score.new_zeros(())
    temp_v = float(temp) if float(temp) > 0 else float(eps)
    probs = torch.softmax(score / temp_v, dim=0)
    idx = torch.topk(score, k=k_eff).indices
    return probs.index_select(0, idx).sum().reshape(())


def _separation(score):
    # Equal-sized halves. When the length is odd the middle view is unused.
    half = score.numel() // 2
    if half == 0:
        return score.new_zeros(())
    ordered = torch.sort(score, descending=True).values
    return (ordered[:half].mean() - ordered[-half:].mean()).reshape(())


def _variance(score):
    if score.numel() <= 1:
        return score.new_zeros(())
    return torch.var(score, unbiased=False).reshape(())


def _spearman(a, b, eps):
    if a.numel() < 2:
        return a.new_zeros(())
    rho = _pearson(_average_rank(a), _average_rank(b), eps)
    return rho.to(dtype=a.dtype, device=a.device)


def _average_rank(x):
    n = x.numel()
    x64 = x.detach().to(dtype=torch.float64)
    sorter = torch.argsort(x64, stable=True)
    ordered = x64[sorter]
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    start = 0
    while start < n:
        end = start + 1
        while end < n and ordered[end] == ordered[start]:
            end += 1
        ranks[sorter[start:end]] = 0.5 * (start + (end - 1))
        start = end
    return ranks


def _pearson(a, b, eps):
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum())
    if (not torch.isfinite(denom)) or float(denom) < eps:
        return a.new_zeros(())
    rho = (a * b).sum() / denom
    if not torch.isfinite(rho):
        return a.new_zeros(())
    return rho
