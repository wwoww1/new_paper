import math
import os
import torch

from qmllm.methods.vtpq.quantize.pre_quant import run_vtpq, apply_vtpq
from qmllm.methods.vtpq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    pseudo_quantize_model_weight_act,
)


def _build_vtpq_cache_config(
    zero_point: bool,
    q_group_size: int,
    w_bit: int,
    a_bit: int,
    wa_quant: bool,
    reweight: bool,
    distort: bool,
    loss_mode: str,
    vtpq_keep_ratio: float,
    vtpq_lambda: float,
    vtpq_score_bit: int,
    vtpq_always_keep_front: int,
    vtpq_score_source: str,
    vtpq_score_layers: int,
    vtpq_positive_only: bool,
    vtpq_min_prune_score: float,
    vtpq_scale_loss: str,
    vtpq_tail_ratio: float,
    vtpq_tail_weight: float,
):
    return {
        "cache_version": 1,
        "method": "vtpq",
        "zero_point": bool(zero_point),
        "q_group_size": int(q_group_size),
        "w_bit": int(w_bit),
        "a_bit": int(a_bit),
        "wa_quant": bool(wa_quant),
        "reweight": bool(reweight),
        "distort": bool(distort),
        "loss_mode": str(loss_mode),
        "vtpq_keep_ratio": float(vtpq_keep_ratio),
        "vtpq_lambda": float(vtpq_lambda),
        "vtpq_score_bit": int(vtpq_score_bit),
        "vtpq_always_keep_front": int(vtpq_always_keep_front),
        "vtpq_score_source": str(vtpq_score_source),
        "vtpq_score_layers": int(vtpq_score_layers),
        "vtpq_positive_only": bool(vtpq_positive_only),
        "vtpq_min_prune_score": float(vtpq_min_prune_score),
        "vtpq_scale_loss": str(vtpq_scale_loss),
        "vtpq_tail_ratio": float(vtpq_tail_ratio),
        "vtpq_tail_weight": float(vtpq_tail_weight),
    }


def _format_config_diff(cached_config, expected_config):
    diffs = []
    for key in sorted(expected_config):
        cached_value = cached_config.get(key, "<missing>")
        expected_value = expected_config[key]
        if cached_value != expected_value:
            diffs.append(f"{key}: cached={cached_value!r}, expected={expected_value!r}")
    return "; ".join(diffs)


def _validate_vtpq_cache(vtpq_results, expected_config, scale_path):
    cached_config = vtpq_results.get("cache_config")
    if cached_config is None:
        raise ValueError(
            f"VTPQ scale cache at {scale_path} has no cache_config metadata. "
            "Regenerate it with the current code, or use a new scale_path."
        )

    diff = _format_config_diff(cached_config, expected_config)
    if diff:
        raise ValueError(
            f"VTPQ scale cache at {scale_path} was generated with different settings: "
            f"{diff}. Use a matching scale_path or regenerate the cache."
        )


def _require_tensor(container, key, owner):
    if container is None or key not in container or container[key] is None:
        raise ValueError(
            f"VTPQ calibration requires {owner}['{key}']. "
            "Use --calib_data coco, ocr_parquet, or mix_vl instead of pileval."
        )
    if not torch.is_tensor(container[key]):
        raise TypeError(f"VTPQ calibration expects {owner}['{key}'] to be a tensor.")
    return container[key]


def _validate_vtpq_calibration_inputs(prompt_inputs, prompt_kwargs, reweight):
    embeds = _require_tensor(prompt_inputs, "inputs_embeds", "prompt_inputs")
    attention_mask = _require_tensor(prompt_kwargs, "attention_mask", "prompt_kwargs")
    vision_mask = _require_tensor(prompt_kwargs, "vision_mask", "prompt_kwargs")

    if embeds.dim() != 3:
        raise ValueError(f"VTPQ expects inputs_embeds to be [B, N, C], got {tuple(embeds.shape)}.")

    expected_shape = embeds.shape[:2]
    if attention_mask.shape != expected_shape:
        raise ValueError(
            f"VTPQ expects attention_mask shape {tuple(expected_shape)}, "
            f"got {tuple(attention_mask.shape)}."
        )
    if vision_mask.shape != expected_shape:
        raise ValueError(
            f"VTPQ expects vision_mask shape {tuple(expected_shape)}, "
            f"got {tuple(vision_mask.shape)}."
        )
    if not vision_mask.bool().any():
        raise ValueError(
            "VTPQ calibration data contains no visual tokens. "
            "Use --calib_data coco, ocr_parquet, or mix_vl."
        )

    if reweight:
        caption_mask = _require_tensor(prompt_kwargs, "caption_mask", "prompt_kwargs")
        if caption_mask.shape != expected_shape:
            raise ValueError(
                f"VTPQ expects caption_mask shape {tuple(expected_shape)}, "
                f"got {tuple(caption_mask.shape)}."
            )
        if not caption_mask.bool().any():
            raise ValueError("VTPQ reweight requires at least one caption token.")


def vtpq_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_vtpq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: bool = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    a_bit: int = 16,
    wa_quant: bool = False,
    reweight: bool = False,
    distort: bool = False,
    loss_mode: str = "mae",
    vtpq_keep_ratio: float = 0.75,
    vtpq_lambda: float = 0.30,
    vtpq_score_bit: int = 8,
    vtpq_always_keep_front: int = 0,
    vtpq_score_source: str = "embed",
    vtpq_score_layers: int = 4,
    vtpq_positive_only: bool = True,
    vtpq_min_prune_score: float = 0.0,
    vtpq_scale_loss: str = "robust",
    vtpq_tail_ratio: float = 0.20,
    vtpq_tail_weight: float = 0.25,
):
    """
    VTPQ pipeline for quantization-interference-aware visual token reduction.

    Pipeline:
    1. reduce visual tokens in calibration inputs;
    2. run VTPQ calibration on reduced-token sequence;
    3. apply VTPQ scale and pseudo quantization.
    """

    if scale_path is None:
        raise ValueError("VTPQ requires scale_path.")
    if not 0.0 < float(vtpq_keep_ratio) <= 1.0:
        raise ValueError("vtpq_keep_ratio must be in (0, 1].")
    if float(vtpq_lambda) < 0.0 or not math.isfinite(float(vtpq_lambda)):
        raise ValueError("vtpq_lambda must be finite and non-negative.")
    if int(vtpq_score_bit) < 2:
        raise ValueError("vtpq_score_bit must be at least 2.")
    if int(vtpq_always_keep_front) < 0:
        raise ValueError("vtpq_always_keep_front must be non-negative.")
    if not math.isfinite(float(vtpq_min_prune_score)):
        raise ValueError("vtpq_min_prune_score must be finite.")
    if vtpq_scale_loss != "robust":
        raise ValueError("vtpq_scale_loss must be 'robust'.")
    if (
        not math.isfinite(float(vtpq_tail_ratio))
        or not 0.0 < float(vtpq_tail_ratio) <= 1.0
    ):
        raise ValueError("vtpq_tail_ratio must be in (0, 1].")
    if (
        not math.isfinite(float(vtpq_tail_weight))
        or float(vtpq_tail_weight) < 0.0
    ):
        raise ValueError("vtpq_tail_weight must be non-negative.")

    q_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
    }

    cache_config = _build_vtpq_cache_config(
        zero_point=zero_point,
        q_group_size=q_group_size,
        w_bit=w_bit,
        a_bit=a_bit,
        wa_quant=wa_quant,
        reweight=reweight,
        distort=distort,
        loss_mode=loss_mode,
        vtpq_keep_ratio=vtpq_keep_ratio,
        vtpq_lambda=vtpq_lambda,
        vtpq_score_bit=vtpq_score_bit,
        vtpq_always_keep_front=vtpq_always_keep_front,
        vtpq_score_source=vtpq_score_source,
        vtpq_score_layers=vtpq_score_layers,
        vtpq_positive_only=vtpq_positive_only,
        vtpq_min_prune_score=vtpq_min_prune_score,
        vtpq_scale_loss=vtpq_scale_loss,
        vtpq_tail_ratio=vtpq_tail_ratio,
        vtpq_tail_weight=vtpq_tail_weight,
    )

    scale_exist = os.path.exists(scale_path)
    vtpq_results = None

    if scale_exist and (run_vtpq_process or pseudo_quant):
        vtpq_results = torch.load(scale_path, map_location="cpu")
        _validate_vtpq_cache(vtpq_results, cache_config, scale_path)

    if run_vtpq_process and not scale_exist:
        _validate_vtpq_calibration_inputs(prompt_inputs, prompt_kwargs, reweight)
        model.to_cpu()

        vtpq_results = run_vtpq(
            model=model,
            prompt_inputs=prompt_inputs,
            prompt_kwargs=prompt_kwargs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            auto_scale=True,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            reweight=reweight,
            distort=distort,
            vtpq_keep_ratio=vtpq_keep_ratio,
            vtpq_lambda=vtpq_lambda,
            vtpq_score_bit=vtpq_score_bit,
            vtpq_always_keep_front=vtpq_always_keep_front,
            vtpq_score_source=vtpq_score_source,
            vtpq_score_layers=vtpq_score_layers,
            vtpq_positive_only=vtpq_positive_only,
            vtpq_min_prune_score=vtpq_min_prune_score,
            vtpq_scale_loss=vtpq_scale_loss,
            vtpq_tail_ratio=vtpq_tail_ratio,
            vtpq_tail_weight=vtpq_tail_weight,
        )

        vtpq_results["cache_config"] = cache_config

        dirpath = os.path.dirname(scale_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(vtpq_results, scale_path)
        print(f"VTPQ results saved at {scale_path}")

    if pseudo_quant:
        if vtpq_results is None:
            if not os.path.exists(scale_path):
                raise FileNotFoundError(f"VTPQ scale cache does not exist: {scale_path}")
            vtpq_results = torch.load(scale_path, map_location="cpu")
            _validate_vtpq_cache(vtpq_results, cache_config, scale_path)

        model.to_cpu()
        apply_vtpq(model.model, vtpq_results)

        if not wa_quant:
            pseudo_quantize_model_weight(
                model.model,
                w_bit=w_bit,
                q_config=q_config,
            )
        else:
            pseudo_quantize_model_weight_act(
                model.model,
                w_bit=w_bit,
                a_bit=a_bit,
            )

        model.to_cuda()

    return model
