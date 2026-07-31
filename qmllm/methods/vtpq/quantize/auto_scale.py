import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm  # Used in apply_scale RMSNorm branch

from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from .robust_loss import retained_token_robust_loss

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_act_scale(x: torch.Tensor):
    """Compute per-channel activation scale: [B, T, C] -> [C]."""
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln: nn.Module, fcs, scales: torch.Tensor):
    """
    Apply channel-wise scaling across (Norm -> Linear(s)):
    - Divide LN/RMSNorm weights (and bias) by scales
    - Multiply following Linear weights by scales
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    # NaN safety checks
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    """
    Apply channel-wise scaling across (Linear -> Linear).
    Note: keeps the original behavior of only scaling the last `scales.size(0)` rows of fc1.
    """
    assert isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear)
    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    # NaN safety checks
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu: nn.Module, fc: nn.Linear, scales: torch.Tensor):
    """Apply channel-wise scaling across (GELU-like activation -> Linear)."""
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(
    module,
    module_kwargs,
    w_bit,
    q_config,
    input_feat,
    ans_mask,
    vis_mask,
    reweight_ratio_dict,
    loss_mode="mae",
    scale_loss="robust",
    tail_ratio=0.20,
    tail_weight=0.25,
):
    """
    Run adaptive scale search for a single Transformer block under weight-only quantization.
    Scale candidates are evaluated with VTPQ's retained-token mean-plus-tail
    robust objective.
    """
    if scale_loss != "robust":
        raise ValueError(
            f"Unsupported VTPQ scale loss {scale_loss!r}; expected 'robust'."
        )

    # === Weight-only quantization function ===
    if w_bit is not None:

        def w_quantize_func(p):
            return pseudo_quantize_tensor(p, n_bits=w_bit, **q_config).detach()

    else:

        def w_quantize_func(p):
            return p

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # === Search scales for a single module (with token-importance reweighting) ===
    def _search_module_scale(
        block,
        linears2scale: list,
        x,
        reweight_ratio=None,
        kwargs={},
    ):
        # x: [B, T, C]
        x = x.to(next(block.parameters()).device)

        # Compute full-precision output baseline
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        # ---- Grid search under the retained-token robust objective ----
        x_max = get_act_scale(x)
        best_error = float("inf")
        best_ratio = -1
        best_scales = None
        n_grid = 20
        history = []

        # Save/restore parameters for each grid candidate
        org_sd = {k: v.detach().cpu() for k, v in block.state_dict().items()}

        for ratio in range(n_grid):
            ratio = ratio / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            # Per-token error aggregated over the hidden dimension.
            if loss_mode == "mse":
                pos_change = (org_out - out).float().pow(2).mean(dim=-1)
            else:  # "mae"
                pos_change = (org_out - out).float().abs().mean(dim=-1)

            loss_val = retained_token_robust_loss(
                token_error=pos_change,
                ans_mask=ans_mask,
                vis_mask=vis_mask,
                tail_ratio=tail_ratio,
                tail_weight=tail_weight,
            )

            history.append((ratio, float(loss_val)))

            if float(loss_val) < best_error:
                best_error = float(loss_val)
                best_ratio = ratio
                best_scales = scales

            # Restore parameters
            block.load_state_dict(org_sd, strict=True)

        if best_ratio == -1 or best_scales is None:
            print("Scale search history:", history)
            raise RuntimeError("Failed to find best ratio.")

        return best_scales.view(-1).detach()

    # === Wrapper: produce (prev_op_name, layer_names, scales_cpu) ===
    def _auto_get_scale(
        prev_op,
        layers,
        inp,
        reweight_ratio=None,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale(
            module2inspect,
            layers,
            inp,
            reweight_ratio,
            kwargs,
        ).detach().cpu()

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    # -------------------------
    # Model-specific branches (including Qwen2 / InternLM2)
    # -------------------------

    def _get_feat(d, *keys):
        """Return the first existing key from input_feat, otherwise raise."""
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f"input_feat missing keys: {keys}")

    scales_list = []

    if isinstance(module, OPTDecoderLayer):
        # Attention input projections
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # Attention output projection
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
            )
        )
        # MLP fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # MLP fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
            )
        )

    elif isinstance(module, LlamaDecoderLayer):
        # Attention input projections
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=input_feat["self_attn.q_proj"],
                reweight_ratio=reweight_ratio_dict.get("attn", None),
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # Attention output projection (only if shapes match)
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                    reweight_ratio=reweight_ratio_dict.get("attn", None),
                )
            )
        # MLP input (gate + up)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                module2inspect=module.mlp,
            )
        )
        # MLP output
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
            )
        )

    elif isinstance(module, BloomBlock):
        # Attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attention.query_key_value],
                inp=input_feat["self_attention.query_key_value"],
                module2inspect=module.self_attention,
                kwargs=module_kwargs,
            )
        )
        # Attention output (only if shapes match)
        if module.self_attention.query_key_value.weight.shape == module.self_attention.dense.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attention.query_key_value,
                    layers=[module.self_attention.dense],
                    inp=input_feat["self_attention.dense"],
                )
            )
        # MLP fc1
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module,
                kwargs=module_kwargs,
            )
        )
        # MLP fc2
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )

    elif "mpt" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1,
                layers=[module.attn.Wqkv],
                inp=input_feat["attn.Wqkv"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv,
                layers=[module.attn.out_proj],
                inp=input_feat["attn.out_proj"],
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2,
                layers=[module.ffn.up_proj, module.ffn.gate_proj],
                inp=input_feat["ffn.up_proj"],
                module2inspect=module.ffn,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.up_proj,
                layers=[module.ffn.down_proj],
                inp=input_feat["ffn.down_proj"],
            )
        )

    elif "gptneox" in str(module.__class__).lower() or "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )
        if module.attention.query_key_value.weight.shape == module.attention.dense.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.attention.query_key_value,
                    layers=[module.attention.dense],
                    inp=input_feat["attention.dense"],
                )
            )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
                module2inspect=module.mlp,
            )
        )
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.dense_h_to_4h,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
            )
        )

    elif module.__class__.__name__ in ("Qwen2DecoderLayer", "Qwen2VLDecoderLayer"):
        # Attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                inp=_get_feat(input_feat, "self_attn.q_proj", "attention.q_proj"),
                reweight_ratio=reweight_ratio_dict.get("attn", None),
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # Attention output (only if shapes match)
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=_get_feat(input_feat, "self_attn.o_proj", "attention.o_proj"),
                    reweight_ratio=reweight_ratio_dict.get("attn", None),
                )
            )
        # MLP input (gate + up)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=_get_feat(input_feat, "mlp.gate_proj", "mlp.up_proj"),
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                module2inspect=module.mlp,
            )
        )
        # MLP output (down_proj)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=_get_feat(input_feat, "mlp.down_proj"),
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
            )
        )

    elif (
        ("internlm2" in str(module.__class__).lower())
        or ("internvl2" in str(module.__class__).lower())
        or (module.__class__.__name__ in ("InternLM2DecoderLayer", "InternVL2DecoderLayer"))
    ):
        # InternLM2 branch: fused wqkv/wo and FFN w1, w3, w2
        attn = getattr(module, "attention", getattr(module, "self_attn", None))
        feed_forward = getattr(module, "feed_forward", getattr(module, "mlp", None))
        attn_norm = getattr(module, "attention_norm", getattr(module, "input_layernorm", None))
        ffn_norm = getattr(module, "ffn_norm", getattr(module, "post_attention_layernorm", None))
        assert (
            attn is not None and feed_forward is not None and attn_norm is not None and ffn_norm is not None
        ), "Unexpected InternLM2 layer structure"

        # Attention input: wqkv
        scales_list.append(
            _auto_get_scale(
                prev_op=attn_norm,
                layers=[attn.wqkv],
                inp=_get_feat(input_feat, "attention.wqkv", "self_attn.wqkv", "attn.wqkv"),
                reweight_ratio=reweight_ratio_dict.get("attn", None),
                module2inspect=attn,
                kwargs=module_kwargs,
            )
        )
        # Attention output: wo (only if present and shapes match)
        if hasattr(attn, "wo") and attn.wqkv.weight.shape == attn.wo.weight.shape:
            scales_list.append(
                _auto_get_scale(
                    prev_op=attn.wqkv,
                    layers=[attn.wo],
                    inp=_get_feat(input_feat, "attention.wo", "self_attn.wo", "attn.wo"),
                    reweight_ratio=reweight_ratio_dict.get("attn", None),
                    module2inspect=attn.wo,
                )
            )
        # MLP input: w1 + w3 (gate/up)
        scales_list.append(
            _auto_get_scale(
                prev_op=ffn_norm,
                layers=[feed_forward.w1, feed_forward.w3],
                inp=_get_feat(input_feat, "feed_forward.w1", "mlp.w1", "mlp.gate_proj"),
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                module2inspect=feed_forward,
            )
        )
        # MLP output: w2 (down)
        scales_list.append(
            _auto_get_scale(
                prev_op=feed_forward.w3,
                layers=[feed_forward.w2],
                inp=_get_feat(input_feat, "feed_forward.w2", "mlp.w2", "mlp.down_proj"),
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
            )
        )

    else:
        raise NotImplementedError(f"Unsupported block type: {type(module)}")

    gc.collect()
    return scales_list


# =========================
# Apply scales (external interface unchanged)
# =========================
@torch.no_grad()
def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ in (
            "InternLM2RMSNorm",
            "Qwen2RMSNorm",
        ):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            # Lazy import to avoid circular dependencies
            from qmllm.methods.vtpq.quantize.qmodule import ScaledActivation  # noqa: WPS433

            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # Optionally apply scaling to cached input features
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
