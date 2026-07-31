import math

import torch


def retained_token_robust_loss(
    token_error: torch.Tensor,
    ans_mask: torch.Tensor = None,
    vis_mask: torch.Tensor = None,
    tail_ratio: float = 0.20,
    tail_weight: float = 0.25,
) -> torch.Tensor:
    """
    Mean-plus-CVaR reconstruction loss over tokens retained after VTPQ pruning.

    ``token_error`` is expected to be [B, T].  When answer/visual masks are
    available, only their union contributes to the objective; otherwise every
    retained sequence position is used.  CVaR is implemented as the mean error
    of the largest ``tail_ratio`` fraction of valid tokens in each sample.
    """
    if token_error.dim() != 2:
        raise ValueError(
            "VTPQ robust scale loss expects token_error shaped [B, T], "
            f"got {tuple(token_error.shape)}."
        )
    if (
        not math.isfinite(float(tail_ratio))
        or not 0.0 < float(tail_ratio) <= 1.0
    ):
        raise ValueError("vtpq_tail_ratio must be in (0, 1].")
    if not math.isfinite(float(tail_weight)) or float(tail_weight) < 0.0:
        raise ValueError("vtpq_tail_weight must be non-negative.")

    if ans_mask is None and vis_mask is None:
        valid_mask = torch.ones_like(token_error, dtype=torch.bool)
    else:
        valid_mask = torch.zeros_like(token_error, dtype=torch.bool)
        for name, mask in (("ans_mask", ans_mask), ("vis_mask", vis_mask)):
            if mask is None:
                continue
            if tuple(mask.shape) != tuple(token_error.shape):
                raise ValueError(
                    f"VTPQ robust scale loss {name} shape {tuple(mask.shape)} "
                    f"does not match token error {tuple(token_error.shape)}."
                )
            valid_mask |= mask.to(token_error.device).bool()

    sample_losses = []
    for batch_idx in range(token_error.size(0)):
        errors = token_error[batch_idx][valid_mask[batch_idx]]
        if errors.numel() == 0:
            # Some model adapters do not expose answer/vision masks for every
            # intermediate tensor. Falling back to all retained positions keeps
            # scale search defined without silently producing a zero loss.
            errors = token_error[batch_idx].reshape(-1)

        mean_loss = errors.mean()
        tail_k = max(1, int(math.ceil(errors.numel() * float(tail_ratio))))
        tail_loss = torch.topk(errors, k=tail_k, largest=True).values.mean()
        sample_losses.append(mean_loss + float(tail_weight) * tail_loss)

    return torch.stack(sample_losses).mean()
