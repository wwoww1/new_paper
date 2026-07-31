import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def _quantize_with_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Symmetric fake quantization.

    x:     [T, C] or [T, C_sub]
    scale: [C] or [C_sub]
    """
    qmax = float(2 ** (bits - 1) - 1)
    scale = scale.float().clamp_min(eps).view(1, -1)

    x_int = torch.round(x.float() / scale).clamp(-qmax, qmax)
    return x_int * scale


def _scale_leave_one_out_gain(
    x: torch.Tensor,
    bits: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Estimate token-level negative calibration influence.

    For per-channel activation quantization, scale is usually controlled
    by the largest activation magnitude in each channel.

    If removing visual token i reduces the per-channel scale and lowers
    reconstruction error on the remaining tokens, token i is considered
    quantization-interfering.

    x: [Tv, C]

    Return:
        gain: [Tv]
        Larger gain means removing this token is likely beneficial
        for quantization calibration.
    """
    x = x.float()
    tv, hidden = x.shape

    if tv <= 1:
        return torch.zeros(tv, device=x.device, dtype=torch.float32)

    qmax = float(2 ** (bits - 1) - 1)

    abs_x = x.abs()

    top_vals, top_idx = torch.topk(abs_x, k=2, dim=0)

    top1_val = top_vals[0]          # [C]
    top2_val = top_vals[1]          # [C]
    top1_idx = top_idx[0].long()    # [C]

    full_scale = top1_val.clamp_min(eps) / qmax
    full_q = _quantize_with_scale(x, full_scale, bits=bits, eps=eps)
    full_res = (x - full_q).pow(2)  # [Tv, C]

    full_res_total = full_res.sum()
    full_res_per_token = full_res.sum(dim=1)
    base_protected_loss = full_res_total - full_res_per_token  # [Tv]

    new_protected_loss = base_protected_loss.clone()
    owners = torch.unique(top1_idx)

    for owner in owners:
        i = int(owner.item())

        cols = torch.nonzero(top1_idx == i, as_tuple=False).view(-1)
        if cols.numel() == 0:
            continue

        old_cols_res = full_res[:, cols]
        old_cols_loss = old_cols_res.sum() - old_cols_res[i].sum()

        new_scale_cols = top2_val[cols].clamp_min(eps) / qmax
        new_q_cols = _quantize_with_scale(x[:, cols], new_scale_cols, bits=bits, eps=eps)
        new_res_cols = (x[:, cols] - new_q_cols).pow(2)
        new_cols_loss = new_res_cols.sum() - new_res_cols[i].sum()

        new_protected_loss[i] = (
            base_protected_loss[i]
            - old_cols_loss
            + new_cols_loss
        )

    denom = float(max((tv - 1) * hidden, 1))
    gain = (base_protected_loss - new_protected_loss) / denom

    # gain > 0 means removing token i lowers remaining-token quantization error.
    return gain.float()


def _semantic_cost_by_redundancy(x: torch.Tensor) -> torch.Tensor:
    """
    Estimate semantic removal cost by visual-token redundancy.

    x: [Tv, C]

    If a token has a highly similar visual neighbor, removing it is safer.
    cost = 1 - max cosine similarity.
    Smaller cost means more redundant.
    """
    x = x.float()
    tv = x.size(0)

    if tv <= 1:
        return torch.ones(tv, device=x.device, dtype=torch.float32)

    x_norm = F.normalize(x, p=2, dim=-1, eps=1e-6)
    sim = x_norm @ x_norm.t()
    sim.fill_diagonal_(-1.0)

    max_sim = sim.max(dim=-1).values
    cost = 1.0 - max_sim
    return cost.clamp_min(0.0).float()


def _vtpq_prune_score(
    embed_x: torch.Tensor,
    bits: int = 8,
    semantic_lambda: float = 0.30,
    layer_xs: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    VTPQ pruning score.

    If layer_xs is provided, the quantization-interference gain is averaged
    over collected LLM-layer hidden states. Otherwise, it falls back to the
    original input-embedding proxy.
    """
    if layer_xs:
        gains = []
        for layer_x in layer_xs:
            gains.append(_scale_leave_one_out_gain(layer_x, bits=bits))
        gain = torch.stack(gains, dim=0).mean(dim=0)
    else:
        gain = _scale_leave_one_out_gain(embed_x, bits=bits)

    cost = _semantic_cost_by_redundancy(embed_x)
    score = _zscore(gain) - semantic_lambda * _zscore(cost)

    return score, {
        "gain": gain.detach().cpu(),
        "semantic_cost": cost.detach().cpu(),
        "score": score.detach().cpu(),
    }


def _pad_1d(x: torch.Tensor, target_len: int, key: str) -> torch.Tensor:
    cur = x.size(0)
    if cur == target_len:
        return x

    if cur > target_len:
        return x[:target_len]

    pad = target_len - cur

    if key == "labels":
        value = -100
    elif x.dtype == torch.bool:
        value = 0
    else:
        value = 0

    if x.dtype == torch.bool:
        y = F.pad(x.to(torch.uint8), (0, pad), value=value)
        return y.to(torch.bool)

    return F.pad(x, (0, pad), value=value)


def _pad_2d(x: torch.Tensor, target_len: int, key: str) -> torch.Tensor:
    cur = x.size(0)
    if cur == target_len:
        return x

    if cur > target_len:
        return x[:target_len]

    pad = target_len - cur
    return F.pad(x, (0, 0, 0, pad), value=0)


def _stack_sequence_items(items: List[torch.Tensor], key: str) -> torch.Tensor:
    target_len = max(x.size(0) for x in items)
    padded = []

    for x in items:
        if x.dim() == 1:
            padded.append(_pad_1d(x, target_len, key))
        elif x.dim() == 2:
            padded.append(_pad_2d(x, target_len, key))
        else:
            raise ValueError(f"Unsupported sequence item shape for key={key}: {x.shape}")

    return torch.stack(padded, dim=0)


def _select_indices_for_sample(
    embeds: torch.Tensor,
    vision_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_ratio: float,
    semantic_lambda: float,
    score_bit: int,
    protect_front: int,
    score_features: List[torch.Tensor] = None,
    positive_only: bool = True,
    min_prune_score: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Select sequence indices for one sample.

    Non-visual tokens are always kept.
    Visual tokens are pruned according to VTPQ score.
    """
    device = embeds.device

    valid_idx = torch.nonzero(attention_mask.bool(), as_tuple=False).view(-1)
    vision_idx = torch.nonzero(
        vision_mask.bool() & attention_mask.bool(),
        as_tuple=False,
    ).view(-1)

    if vision_idx.numel() == 0:
        return valid_idx.sort().values, {
            "num_visual": 0,
            "num_keep_visual": 0,
            "num_drop_visual": 0,
            "num_budget_drop_visual": 0,
            "num_eligible_prune_visual": 0,
            "mean_gain": 0.0,
            "mean_semantic_cost": 0.0,
            "mean_score": 0.0,
        }

    # keep all non-visual valid tokens
    non_vision_mask = ~vision_mask[valid_idx].bool()
    non_vision_idx = valid_idx[non_vision_mask]

    tv = int(vision_idx.numel())

    if positive_only:
        # keep_ratio is a lower bound on retained visual tokens, hence an
        # upper bound on deletions. ceil prevents small token counts from
        # exceeding that budget (e.g. ceil(6 * 0.75) == 5).
        num_keep = max(1, int(math.ceil(tv * keep_ratio)))
    else:
        # Preserve the old fixed-quota rounding for compatibility ablations.
        num_keep = max(1, int(round(tv * keep_ratio)))
    num_keep = min(num_keep, tv)
    num_drop = tv - num_keep

    if num_drop <= 0:
        return valid_idx.sort().values, {
            "num_visual": tv,
            "num_keep_visual": tv,
            "num_drop_visual": 0,
            "num_budget_drop_visual": 0,
            "num_eligible_prune_visual": 0,
            "mean_gain": 0.0,
            "mean_semantic_cost": 0.0,
            "mean_score": 0.0,
        }

    # Optionally protect the first few visual tokens.
    # Useful if model has special visual summary/register tokens.
    protect_front = min(int(protect_front), tv)
    protected_vis_idx = vision_idx[:protect_front]
    candidate_vis_idx = vision_idx[protect_front:]

    if candidate_vis_idx.numel() == 0:
        return valid_idx.sort().values, {
            "num_visual": tv,
            "num_keep_visual": tv,
            "num_drop_visual": 0,
            "num_budget_drop_visual": num_drop,
            "num_eligible_prune_visual": 0,
            "mean_gain": 0.0,
            "mean_semantic_cost": 0.0,
            "mean_score": 0.0,
        }

    candidate_embeds = embeds.index_select(0, candidate_vis_idx)
    candidate_score_features = None
    if score_features:
        candidate_score_features = [
            feature.index_select(0, candidate_vis_idx.to(feature.device)).to(candidate_embeds.device)
            for feature in score_features
        ]

    prune_score, aux = _vtpq_prune_score(
        candidate_embeds,
        bits=score_bit,
        semantic_lambda=semantic_lambda,
        layer_xs=candidate_score_features,
    )

    if positive_only:
        # ``gain > 0`` is the proxy's absolute prediction that removing the
        # token lowers quantization reconstruction error.  ``prune_score`` is
        # z-normalized and additionally accounts for semantic removal cost, so
        # use it as a conservative second gate rather than treating its ranking
        # alone as evidence that a token is beneficial to remove.
        raw_gain = aux["gain"].to(prune_score.device)
        eligible_mask = (raw_gain > 0.0) & (prune_score > float(min_prune_score))
    else:
        eligible_mask = torch.ones_like(prune_score, dtype=torch.bool)

    eligible_local_idx = torch.nonzero(
        eligible_mask,
        as_tuple=False,
    ).view(-1)
    actual_drop = min(num_drop, int(eligible_local_idx.numel()))

    if actual_drop > 0:
        eligible_scores = prune_score.index_select(0, eligible_local_idx)
        ranked_eligible_idx = torch.topk(
            eligible_scores,
            k=actual_drop,
            largest=True,
        ).indices
        drop_local_idx = eligible_local_idx.index_select(0, ranked_eligible_idx)
    else:
        drop_local_idx = torch.empty(
            0,
            device=prune_score.device,
            dtype=torch.long,
        )

    keep_candidate_mask = torch.ones(
        candidate_vis_idx.numel(),
        device=device,
        dtype=torch.bool,
    )
    keep_candidate_mask[drop_local_idx] = False

    kept_vis_idx = torch.cat(
        [
            protected_vis_idx,
            candidate_vis_idx[keep_candidate_mask],
        ],
        dim=0,
    )

    selected_idx = torch.cat(
        [
            non_vision_idx,
            kept_vis_idx,
        ],
        dim=0,
    ).sort().values

    info = {
        "num_visual": tv,
        "num_keep_visual": int(kept_vis_idx.numel()),
        "num_drop_visual": int(actual_drop),
        "num_budget_drop_visual": int(num_drop),
        "num_eligible_prune_visual": int(eligible_local_idx.numel()),
        "mean_gain": float(aux["gain"].mean()) if aux["gain"].numel() > 0 else 0.0,
        "mean_semantic_cost": float(aux["semantic_cost"].mean()) if aux["semantic_cost"].numel() > 0 else 0.0,
        "mean_score": float(aux["score"].mean()) if aux["score"].numel() > 0 else 0.0,
    }

    return selected_idx, info


def _is_sequence_tensor(x: torch.Tensor, seq_len: int) -> bool:
    return torch.is_tensor(x) and x.dim() >= 2 and x.size(1) == seq_len


def _reduce_tensor_by_indices(
    x: torch.Tensor,
    selected_indices: List[torch.Tensor],
    key: str,
    seq_len: int,
) -> torch.Tensor:
    """
    Reduce batch tensor whose dim=1 corresponds to sequence length.

    Supports:
        [B, N]
        [B, N, C]

    Other shapes are returned unchanged.
    """
    # Qwen2-VL uses multimodal RoPE position ids shaped [3, B, N], where
    # the leading axis stores temporal/height/width coordinates.  Sequence
    # tensors elsewhere in the project are [B, N] or [B, N, C], so handle
    # this layout explicitly before applying the regular dim-1 check.
    if (
        key == "position_ids"
        and x.dim() == 3
        and x.size(0) == 3
        and x.size(2) == seq_len
    ):
        items = []
        for b, idx in enumerate(selected_indices):
            idx = idx.to(x.device)
            items.append(x[:, b, :].index_select(1, idx))

        target_len = max(item.size(1) for item in items)
        padded = [
            F.pad(item, (0, target_len - item.size(1)), value=0)
            for item in items
        ]
        return torch.stack(padded, dim=1).to(x.device)

    if not _is_sequence_tensor(x, seq_len):
        return x

    if x.dim() not in [2, 3]:
        print(f"[VTPQ] warning: skip reducing unsupported tensor {key}: {tuple(x.shape)}")
        return x

    items = []
    for b, idx in enumerate(selected_indices):
        idx = idx.to(x.device)

        if x.dim() == 2:
            items.append(x[b].index_select(0, idx))
        elif x.dim() == 3:
            items.append(x[b].index_select(0, idx))

    return _stack_sequence_items(items, key=key).to(x.device)


def reduce_calibration_tokens(
    prompt_inputs: Dict[str, torch.Tensor],
    prompt_kwargs: Dict[str, torch.Tensor],
    keep_ratio: float = 0.75,
    semantic_lambda: float = 0.30,
    score_bit: int = 8,
    protect_front: int = 0,
    score_features=None,
    log_reduction: bool = True,
    positive_only: bool = True,
    min_prune_score: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Reduce visual tokens in calibration inputs.

    Required:
        prompt_inputs["inputs_embeds"]: [B, N, C]
        prompt_kwargs["attention_mask"]: [B, N]
        prompt_kwargs["vision_mask"]: [B, N]

    Return:
        reduced_inputs
        reduced_kwargs
        vtpq_info

    By default, ``keep_ratio`` is a maximum deletion budget: a visual token
    is removed only when its leave-one-out quantization gain is positive and
    its combined pruning score clears ``min_prune_score``.
    """
    if prompt_inputs is None or prompt_kwargs is None:
        raise ValueError("VTPQ requires prompt_inputs and prompt_kwargs.")

    if "inputs_embeds" not in prompt_inputs:
        raise KeyError("VTPQ requires prompt_inputs['inputs_embeds'].")

    if "attention_mask" not in prompt_kwargs:
        raise KeyError("VTPQ requires prompt_kwargs['attention_mask'].")

    if "vision_mask" not in prompt_kwargs:
        raise KeyError("VTPQ requires prompt_kwargs['vision_mask'].")

    embeds = prompt_inputs["inputs_embeds"]
    attention_mask = prompt_kwargs["attention_mask"]
    vision_mask = prompt_kwargs["vision_mask"]

    if embeds.dim() != 3:
        raise ValueError(f"inputs_embeds should be [B, N, C], got {tuple(embeds.shape)}")

    bsz, seq_len, _ = embeds.shape

    if score_features is not None:
        if not isinstance(score_features, (list, tuple)) or len(score_features) == 0:
            raise ValueError("score_features must be a non-empty list of [B, N, C] tensors.")
        for layer_idx, feature in enumerate(score_features):
            if not torch.is_tensor(feature) or feature.dim() != 3:
                raise ValueError(
                    f"score_features[{layer_idx}] should be [B, N, C], "
                    f"got {tuple(feature.shape) if torch.is_tensor(feature) else type(feature)}."
                )
            if feature.shape[:2] != embeds.shape[:2]:
                raise ValueError(
                    f"score_features[{layer_idx}] sequence shape {tuple(feature.shape[:2])} "
                    f"does not match inputs_embeds {tuple(embeds.shape[:2])}."
                )

    selected_indices = []
    sample_infos = []

    for b in range(bsz):
        sample_score_features = None
        if score_features is not None:
            sample_score_features = [feature[b] for feature in score_features]

        selected_idx, info = _select_indices_for_sample(
            embeds=embeds[b],
            vision_mask=vision_mask[b],
            attention_mask=attention_mask[b],
            keep_ratio=keep_ratio,
            semantic_lambda=semantic_lambda,
            score_bit=score_bit,
            protect_front=protect_front,
            score_features=sample_score_features,
            positive_only=positive_only,
            min_prune_score=min_prune_score,
        )
        selected_indices.append(selected_idx)
        sample_infos.append(info)

    reduced_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            reduced_inputs[key] = _reduce_tensor_by_indices(
                value,
                selected_indices,
                key=key,
                seq_len=seq_len,
            )
        else:
            reduced_inputs[key] = value

    reduced_kwargs = {}
    for key, value in prompt_kwargs.items():
        if torch.is_tensor(value):
            reduced_kwargs[key] = _reduce_tensor_by_indices(
                value,
                selected_indices,
                key=key,
                seq_len=seq_len,
            )
        else:
            reduced_kwargs[key] = value

    # Rebuild ordinary 1D position ids because compaction removes sequence gaps.
    # Preserve Qwen2-VL's [3, B, N] multimodal coordinates: gaps in those
    # coordinates are meaningful, and the Qwen adapter updates rope_deltas to
    # account for the shorter compacted cache.
    if "position_ids" in reduced_kwargs and "attention_mask" in reduced_kwargs:
        position_ids = reduced_kwargs["position_ids"]
        if position_ids.dim() == 2:
            attn = reduced_kwargs["attention_mask"].long()
            pos = attn.cumsum(dim=-1) - 1
            pos = pos.masked_fill(attn == 0, 0)
            reduced_kwargs["position_ids"] = pos

    total_visual = sum(x["num_visual"] for x in sample_infos)
    total_keep = sum(x["num_keep_visual"] for x in sample_infos)
    total_drop = sum(x["num_drop_visual"] for x in sample_infos)
    total_budget_drop = sum(x["num_budget_drop_visual"] for x in sample_infos)
    total_eligible = sum(x["num_eligible_prune_visual"] for x in sample_infos)

    vtpq_info = {
        "method": "VTPQ-v2-layerwise-scale-leave-one-out" if score_features is not None else "VTPQ-v1-scale-leave-one-out",
        "score_source": "layer" if score_features is not None else "embed",
        "num_score_layers": int(len(score_features)) if score_features is not None else 0,
        "keep_ratio": float(keep_ratio),
        "semantic_lambda": float(semantic_lambda),
        "score_bit": int(score_bit),
        "protect_front": int(protect_front),
        "positive_only": bool(positive_only),
        "budget_semantics": "maximum_drop" if positive_only else "fixed_quota",
        "min_prune_score": float(min_prune_score),
        "total_visual_tokens": int(total_visual),
        "kept_visual_tokens": int(total_keep),
        "dropped_visual_tokens": int(total_drop),
        "budgeted_drop_visual_tokens": int(total_budget_drop),
        "eligible_prune_visual_tokens": int(total_eligible),
        "actual_keep_ratio": float(total_keep / max(total_visual, 1)),
        "sample_infos": sample_infos,
    }

    if log_reduction:
        print(
            "[VTPQ] visual token reduction: "
            f"kept {total_keep}/{total_visual} "
            f"({vtpq_info['actual_keep_ratio']:.4f}), "
            f"dropped {total_drop}/{total_budget_drop} budgeted, "
            f"eligible {total_eligible}, "
            f"positive_only={bool(positive_only)}"
        )

    return reduced_inputs, reduced_kwargs, vtpq_info
# inference stage
@torch.inference_mode()
def reduce_inference_tokens(
    prompt_inputs,
    prompt_kwargs,
    keep_ratio=0.75,
    semantic_lambda=0.30,
    score_bit=8,
    protect_front=0,
    positive_only=True,
    min_prune_score=0.0,
):
    """
    Input-stage visual-token pruning for inference.

    The first implementation intentionally uses embedding-based scores.
    It requires no backward pass or calibration sample mask.
    """
    return reduce_calibration_tokens(
        prompt_inputs=prompt_inputs,
        prompt_kwargs=prompt_kwargs,
        keep_ratio=keep_ratio,
        semantic_lambda=semantic_lambda,
        score_bit=score_bit,
        protect_front=protect_front,
        score_features=None,
        log_reduction=False,
        positive_only=positive_only,
        min_prune_score=min_prune_score,
    )
