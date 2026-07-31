import functools
import inspect
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from qmllm.methods.vtpq.quantize.token_reducer import reduce_inference_tokens


VTPQ_INFERENCE_PRUNING_MODELS = frozenset(
    {"internvl2", "qwen2_vl", "llava_onevision"}
)


@dataclass
class VTPQInferencePruningStats:
    model_type: str = "unknown"
    samples: int = 0
    total_visual_tokens: int = 0
    kept_visual_tokens: int = 0
    dropped_visual_tokens: int = 0
    budgeted_drop_visual_tokens: int = 0
    eligible_prune_visual_tokens: int = 0
    total_score_time_ms: float = 0.0
    max_score_time_ms: float = 0.0
    score_time_samples_ms: List[float] = field(default_factory=list)

    def update(self, info: Dict[str, Any], score_time_ms: float) -> None:
        self.samples += 1
        self.total_visual_tokens += int(info["total_visual_tokens"])
        self.kept_visual_tokens += int(info["kept_visual_tokens"])
        self.dropped_visual_tokens += int(info["dropped_visual_tokens"])
        self.budgeted_drop_visual_tokens += int(
            info.get("budgeted_drop_visual_tokens", info["dropped_visual_tokens"])
        )
        self.eligible_prune_visual_tokens += int(
            info.get("eligible_prune_visual_tokens", info["dropped_visual_tokens"])
        )
        self.total_score_time_ms += float(score_time_ms)
        self.max_score_time_ms = max(self.max_score_time_ms, float(score_time_ms))
        self.score_time_samples_ms.append(float(score_time_ms))

    def mean_score_time_window_ms(
        self,
        warmup_samples: int = 0,
        max_samples: Optional[int] = None,
    ) -> float:
        start = max(int(warmup_samples), 0)
        stop = None if max_samples is None else start + max(int(max_samples), 0)
        values = self.score_time_samples_ms[start:stop]
        return sum(values) / max(len(values), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "samples": self.samples,
            "total_visual_tokens": self.total_visual_tokens,
            "kept_visual_tokens": self.kept_visual_tokens,
            "dropped_visual_tokens": self.dropped_visual_tokens,
            "budgeted_drop_visual_tokens": self.budgeted_drop_visual_tokens,
            "eligible_prune_visual_tokens": self.eligible_prune_visual_tokens,
            "actual_keep_ratio": (
                self.kept_visual_tokens / max(self.total_visual_tokens, 1)
            ),
            "mean_score_time_ms": (
                self.total_score_time_ms / max(self.samples, 1)
            ),
            "max_score_time_ms": self.max_score_time_ms,
        }


def _normalize_model_type(model_type: str) -> str:
    return str(model_type).strip().lower().replace("-", "_")


def _extract_call_argument(original, args, kwargs, *names):
    for name in names:
        value = kwargs.get(name)
        if value is not None:
            return value

    try:
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None

    for name in names:
        value = bound.arguments.get(name)
        if value is not None:
            return value

    for parameter in inspect.signature(original).parameters.values():
        if parameter.kind != inspect.Parameter.VAR_KEYWORD:
            continue
        extras = bound.arguments.get(parameter.name, {})
        if isinstance(extras, dict):
            for name in names:
                value = extras.get(name)
                if value is not None:
                    return value
    return None


def _extract_input_ids(original_generate, args, kwargs) -> Optional[torch.Tensor]:
    input_ids = _extract_call_argument(
        original_generate,
        args,
        kwargs,
        "input_ids",
        "inputs",
    )
    return input_ids if torch.is_tensor(input_ids) else None


def _sync_if_cuda(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _prune_embedded_prompt(
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_mask: torch.Tensor,
    keep_ratio: float,
    semantic_lambda: float,
    score_bit: int,
    protect_front: int,
    positive_only: bool,
    min_prune_score: float,
    stats: VTPQInferencePruningStats,
    input_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    prompt_inputs = {"inputs_embeds": inputs_embeds}
    if torch.is_tensor(input_ids):
        prompt_inputs["input_ids"] = input_ids

    prompt_kwargs = {
        "attention_mask": attention_mask,
        "vision_mask": vision_mask.to(attention_mask.device),
    }
    if torch.is_tensor(position_ids):
        prompt_kwargs["position_ids"] = position_ids

    _sync_if_cuda(inputs_embeds)
    start = time.perf_counter()
    reduced_inputs, reduced_kwargs, info = reduce_inference_tokens(
        prompt_inputs=prompt_inputs,
        prompt_kwargs=prompt_kwargs,
        keep_ratio=keep_ratio,
        semantic_lambda=semantic_lambda,
        score_bit=score_bit,
        protect_front=protect_front,
        positive_only=positive_only,
        min_prune_score=min_prune_score,
    )
    _sync_if_cuda(inputs_embeds)
    score_time_ms = (time.perf_counter() - start) * 1000.0
    stats.update(info, score_time_ms)
    return reduced_inputs, reduced_kwargs, info


def _validate_installer_target(model, model_label: str) -> None:
    if getattr(model, "_vtpq_inference_pruning_installed", False):
        raise RuntimeError("VTPQ inference pruning is already installed on this model.")
    if not hasattr(model, "generate"):
        raise TypeError(f"{model_label} model has no generate method to wrap.")


def install_internvl2_vtpq_inference_pruning(
    model,
    keep_ratio: float = 0.75,
    semantic_lambda: float = 0.30,
    score_bit: int = 8,
    protect_front: int = 0,
    positive_only: bool = True,
    min_prune_score: float = 0.0,
) -> VTPQInferencePruningStats:
    """Install input-stage VTPQ pruning into InternVL2's chat/generate path.

    InternVL2 builds multimodal ``inputs_embeds`` inside its own ``generate``
    method, then calls ``language_model.generate``. The outer wrapper captures
    the IMG_CONTEXT positions from ``input_ids``; the inner wrapper compacts the
    matching embeddings immediately before the language-model prefill.
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("VTPQ inference keep_ratio must be in (0, 1].")
    _validate_installer_target(model, "InternVL2")
    if not hasattr(model, "language_model"):
        raise TypeError("InternVL2 model has no language_model to wrap.")
    if not hasattr(model.language_model, "generate"):
        raise TypeError("InternVL2 language_model has no generate method to wrap.")

    original_model_generate = model.generate
    original_lm_generate = model.language_model.generate
    runtime: Dict[str, Optional[torch.Tensor]] = {"vision_mask": None}
    stats = VTPQInferencePruningStats(model_type="internvl2")

    @functools.wraps(original_model_generate)
    def wrapped_model_generate(_self, *args, **kwargs):
        input_ids = _extract_input_ids(original_model_generate, args, kwargs)
        img_context_token_id = getattr(_self, "img_context_token_id", None)

        if input_ids is not None and img_context_token_id is not None:
            runtime["vision_mask"] = input_ids.eq(int(img_context_token_id))
        else:
            runtime["vision_mask"] = None

        try:
            return original_model_generate(*args, **kwargs)
        finally:
            runtime["vision_mask"] = None

    @functools.wraps(original_lm_generate)
    def wrapped_lm_generate(_self, *args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        attention_mask = kwargs.get("attention_mask")
        vision_mask = runtime.get("vision_mask")

        if (
            not torch.is_tensor(inputs_embeds)
            or not torch.is_tensor(attention_mask)
            or not torch.is_tensor(vision_mask)
            or not vision_mask.bool().any()
        ):
            return original_lm_generate(*args, **kwargs)

        expected_shape = tuple(inputs_embeds.shape[:2])
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                "InternVL2 VTPQ attention-mask shape mismatch: "
                f"inputs_embeds={expected_shape}, attention_mask={tuple(attention_mask.shape)}"
            )
        if tuple(vision_mask.shape) != expected_shape:
            raise ValueError(
                "InternVL2 VTPQ vision-mask shape mismatch: "
                f"inputs_embeds={expected_shape}, vision_mask={tuple(vision_mask.shape)}"
            )

        reduced_inputs, reduced_kwargs, _ = _prune_embedded_prompt(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            vision_mask=vision_mask,
            input_ids=kwargs.get("input_ids"),
            position_ids=kwargs.get("position_ids"),
            keep_ratio=keep_ratio,
            semantic_lambda=semantic_lambda,
            score_bit=score_bit,
            protect_front=protect_front,
            positive_only=positive_only,
            min_prune_score=min_prune_score,
            stats=stats,
        )

        kwargs["inputs_embeds"] = reduced_inputs["inputs_embeds"]
        kwargs["attention_mask"] = reduced_kwargs["attention_mask"]
        if "input_ids" in reduced_inputs:
            kwargs["input_ids"] = reduced_inputs["input_ids"]
        if "position_ids" in reduced_kwargs:
            kwargs["position_ids"] = reduced_kwargs["position_ids"]

        return original_lm_generate(*args, **kwargs)

    model.generate = types.MethodType(wrapped_model_generate, model)
    model.language_model.generate = types.MethodType(
        wrapped_lm_generate,
        model.language_model,
    )
    model._vtpq_inference_pruning_installed = True
    model._vtpq_inference_pruning_stats = stats
    return stats


def _qwen2vl_scatter_visual_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    token_id: int,
    visual_embeds: torch.Tensor,
    modality: str,
) -> torch.Tensor:
    token_mask = input_ids.eq(int(token_id))
    num_tokens = int(token_mask.sum().item())
    if num_tokens != int(visual_embeds.shape[0]):
        raise ValueError(
            f"Qwen2-VL VTPQ {modality}-token mismatch: "
            f"tokens={num_tokens}, embeddings={int(visual_embeds.shape[0])}."
        )
    visual_embeds = visual_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    expanded_mask = token_mask.unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(expanded_mask, visual_embeds)


def _qwen2vl_compacted_rope_delta(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    deltas = []
    for batch_idx in range(attention_mask.size(0)):
        valid = attention_mask[batch_idx].bool()
        if valid.any():
            max_position = position_ids[:, batch_idx, valid].max()
        else:
            max_position = position_ids.new_tensor(-1)
        deltas.append(max_position + 1 - attention_mask.shape[-1])
    return torch.stack(deltas).reshape(-1, 1).to(position_ids.dtype)


def _restore_qwen_generation_prefix(result, original_input_ids, compact_input_ids):
    sequences = result if torch.is_tensor(result) else getattr(result, "sequences", None)
    if not torch.is_tensor(sequences):
        return result
    if sequences.size(0) % compact_input_ids.size(0) != 0:
        return result

    repeat = sequences.size(0) // compact_input_ids.size(0)
    compact_prefix = compact_input_ids.repeat_interleave(repeat, dim=0)
    original_prefix = original_input_ids.repeat_interleave(repeat, dim=0)

    if (
        sequences.size(1) >= compact_prefix.size(1)
        and torch.equal(sequences[:, : compact_prefix.size(1)], compact_prefix)
    ):
        generated = sequences[:, compact_prefix.size(1) :]
    else:
        # Transformers 4.49 can return only newly generated ids when generation
        # starts from inputs_embeds. Treat that output as the generated suffix.
        generated = sequences

    restored = torch.cat([original_prefix.to(sequences.device), generated], dim=1)
    if torch.is_tensor(result):
        return restored
    result.sequences = restored
    return result


def install_qwen2_vl_vtpq_inference_pruning(
    model,
    keep_ratio: float = 0.75,
    semantic_lambda: float = 0.30,
    score_bit: int = 8,
    protect_front: int = 0,
    positive_only: bool = True,
    min_prune_score: float = 0.0,
) -> VTPQInferencePruningStats:
    """Install VTPQ pruning after Qwen2-VL vision encoding and before prefill.

    The adapter preserves Qwen2-VL's three-axis multimodal RoPE coordinates,
    updates ``rope_deltas`` for the compacted KV cache, and restores the public
    generation result's original prompt prefix for lmms-eval compatibility.
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("VTPQ inference keep_ratio must be in (0, 1].")
    _validate_installer_target(model, "Qwen2-VL")
    for attribute in ("model", "visual", "config", "get_rope_index"):
        if not hasattr(model, attribute):
            raise TypeError(f"Qwen2-VL model has no {attribute} attribute.")
    if not hasattr(model.model, "embed_tokens"):
        raise TypeError("Qwen2-VL language model has no embed_tokens module.")

    original_generate = model.generate
    original_get_rope_index = model.get_rope_index
    runtime: Dict[str, Optional[torch.Tensor]] = {
        "position_ids": None,
        "rope_deltas": None,
    }
    stats = VTPQInferencePruningStats(model_type="qwen2_vl")

    @functools.wraps(original_get_rope_index)
    def wrapped_get_rope_index(_self, *args, **kwargs):
        if runtime["position_ids"] is not None:
            return runtime["position_ids"], runtime["rope_deltas"]
        return original_get_rope_index(*args, **kwargs)

    @functools.wraps(original_generate)
    @torch.no_grad()
    def wrapped_generate(_self, *args, **kwargs):
        input_ids = _extract_input_ids(original_generate, args, kwargs)
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")

        if (
            not torch.is_tensor(input_ids)
            or (pixel_values is None and pixel_values_videos is None)
        ):
            return original_generate(*args, **kwargs)
        if input_ids.size(0) != 1:
            raise NotImplementedError(
                "Qwen2-VL VTPQ inference pruning currently requires batch_size=1."
            )
        if kwargs.get("past_key_values") is not None:
            raise NotImplementedError(
                "Qwen2-VL VTPQ pruning supports fresh multimodal prefill only."
            )
        if not torch.is_tensor(attention_mask):
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        inputs_embeds = _self.model.embed_tokens(input_ids)
        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        if torch.is_tensor(pixel_values):
            image_token_id = int(_self.config.image_token_id)
            image_pixels = pixel_values.type(_self.visual.get_dtype())
            image_embeds = _self.visual(image_pixels, grid_thw=image_grid_thw)
            inputs_embeds = _qwen2vl_scatter_visual_embeddings(
                inputs_embeds,
                input_ids,
                image_token_id,
                image_embeds,
                "image",
            )
            vision_mask |= input_ids.eq(image_token_id)

        if torch.is_tensor(pixel_values_videos):
            video_token_id = int(_self.config.video_token_id)
            video_pixels = pixel_values_videos.type(_self.visual.get_dtype())
            video_embeds = _self.visual(video_pixels, grid_thw=video_grid_thw)
            inputs_embeds = _qwen2vl_scatter_visual_embeddings(
                inputs_embeds,
                input_ids,
                video_token_id,
                video_embeds,
                "video",
            )
            vision_mask |= input_ids.eq(video_token_id)

        if not vision_mask.any():
            return original_generate(*args, **kwargs)

        position_ids, _ = original_get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        reduced_inputs, reduced_kwargs, _ = _prune_embedded_prompt(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            vision_mask=vision_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            keep_ratio=keep_ratio,
            semantic_lambda=semantic_lambda,
            score_bit=score_bit,
            protect_front=protect_front,
            positive_only=positive_only,
            min_prune_score=min_prune_score,
            stats=stats,
        )

        compact_input_ids = reduced_inputs["input_ids"]
        compact_position_ids = reduced_kwargs["position_ids"]
        compact_attention_mask = reduced_kwargs["attention_mask"]
        compact_rope_deltas = _qwen2vl_compacted_rope_delta(
            compact_position_ids,
            compact_attention_mask,
        )

        call_args = list(args)
        call_kwargs = dict(kwargs)
        if "input_ids" in call_kwargs:
            call_kwargs["input_ids"] = compact_input_ids
        elif "inputs" in call_kwargs:
            call_kwargs["inputs"] = compact_input_ids
        elif call_args and torch.is_tensor(call_args[0]):
            call_args[0] = compact_input_ids
        else:
            call_kwargs["input_ids"] = compact_input_ids

        call_kwargs["inputs_embeds"] = reduced_inputs["inputs_embeds"]
        call_kwargs["attention_mask"] = compact_attention_mask
        call_kwargs.pop("position_ids", None)
        call_kwargs.pop("rope_deltas", None)
        call_kwargs.pop("pixel_values", None)
        call_kwargs.pop("pixel_values_videos", None)
        call_kwargs.pop("image_grid_thw", None)
        call_kwargs.pop("video_grid_thw", None)

        runtime["position_ids"] = compact_position_ids
        runtime["rope_deltas"] = compact_rope_deltas
        try:
            result = original_generate(*call_args, **call_kwargs)
        finally:
            runtime["position_ids"] = None
            runtime["rope_deltas"] = None

        return _restore_qwen_generation_prefix(
            result,
            original_input_ids=input_ids,
            compact_input_ids=compact_input_ids,
        )

    model.get_rope_index = types.MethodType(wrapped_get_rope_index, model)
    model.generate = types.MethodType(wrapped_generate, model)
    model._vtpq_inference_pruning_installed = True
    model._vtpq_inference_pruning_stats = stats
    return stats


def install_llava_onevision_vtpq_inference_pruning(
    model,
    keep_ratio: float = 0.75,
    semantic_lambda: float = 0.30,
    score_bit: int = 8,
    protect_front: int = 0,
    positive_only: bool = True,
    min_prune_score: float = 0.0,
) -> VTPQInferencePruningStats:
    """Install VTPQ pruning at LLaVA-OneVision's multimodal assembly boundary."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("VTPQ inference keep_ratio must be in (0, 1].")
    _validate_installer_target(model, "LLaVA-OneVision")
    if not hasattr(model, "prepare_inputs_labels_for_multimodal"):
        raise TypeError(
            "LLaVA-OneVision model has no prepare_inputs_labels_for_multimodal method."
        )

    original_prepare = model.prepare_inputs_labels_for_multimodal
    stats = VTPQInferencePruningStats(model_type="llava_onevision")

    @functools.wraps(original_prepare)
    def wrapped_prepare(_self, *args, **kwargs):
        try:
            bound = inspect.signature(original_prepare).bind_partial(*args, **kwargs)
        except (TypeError, ValueError):
            return original_prepare(*args, **kwargs)

        input_ids = bound.arguments.get("input_ids")
        labels = bound.arguments.get("labels")
        images = bound.arguments.get("images")
        attention_mask = bound.arguments.get("attention_mask")

        # Restrict pruning to generation. Log-likelihood/training calls provide
        # labels and must retain the full multimodal sequence.
        if (
            not torch.is_tensor(input_ids)
            or labels is not None
            or images is None
            or input_ids.shape[1] <= 1
        ):
            return original_prepare(*args, **kwargs)
        if input_ids.size(0) != 1:
            raise NotImplementedError(
                "LLaVA-OneVision VTPQ inference pruning currently requires batch_size=1."
            )

        if not torch.is_tensor(attention_mask):
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            bound.arguments["attention_mask"] = attention_mask

        # LLaVA fills inserted visual locations with IGNORE_INDEX. Supplying a
        # zero marker label for every text token lets us recover an exact visual
        # mask after variable-length any-resolution image expansion.
        bound.arguments["labels"] = torch.zeros_like(input_ids)
        prepared = original_prepare(**bound.arguments)
        if not isinstance(prepared, (tuple, list)) or len(prepared) != 6:
            raise RuntimeError(
                "Unexpected LLaVA multimodal preparation result; expected six values."
            )

        (
            prepared_input_ids,
            position_ids,
            prepared_attention_mask,
            past_key_values,
            inputs_embeds,
            marker_labels,
        ) = prepared
        if not torch.is_tensor(inputs_embeds) or not torch.is_tensor(marker_labels):
            return prepared_input_ids, position_ids, prepared_attention_mask, past_key_values, inputs_embeds, None
        if not torch.is_tensor(prepared_attention_mask):
            prepared_attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        vision_mask = marker_labels.eq(-100) & prepared_attention_mask.bool()
        if not vision_mask.any():
            return prepared_input_ids, position_ids, prepared_attention_mask, past_key_values, inputs_embeds, None

        reduced_inputs, reduced_kwargs, _ = _prune_embedded_prompt(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared_attention_mask,
            vision_mask=vision_mask,
            position_ids=position_ids,
            keep_ratio=keep_ratio,
            semantic_lambda=semantic_lambda,
            score_bit=score_bit,
            protect_front=protect_front,
            positive_only=positive_only,
            min_prune_score=min_prune_score,
            stats=stats,
        )

        return (
            prepared_input_ids,
            reduced_kwargs.get("position_ids"),
            reduced_kwargs["attention_mask"],
            past_key_values,
            reduced_inputs["inputs_embeds"],
            None,
        )

    model.prepare_inputs_labels_for_multimodal = types.MethodType(
        wrapped_prepare,
        model,
    )
    model._vtpq_inference_pruning_installed = True
    model._vtpq_inference_pruning_stats = stats
    return stats


def install_vtpq_inference_pruning(
    model_type: str,
    model,
    keep_ratio: float = 0.75,
    semantic_lambda: float = 0.30,
    score_bit: int = 8,
    protect_front: int = 0,
    positive_only: bool = True,
    min_prune_score: float = 0.0,
) -> VTPQInferencePruningStats:
    """Dispatch VTPQ inference pruning to the selected MLLM adapter."""
    normalized = _normalize_model_type(model_type)
    installers = {
        "internvl2": install_internvl2_vtpq_inference_pruning,
        "qwen2_vl": install_qwen2_vl_vtpq_inference_pruning,
        "llava_onevision": install_llava_onevision_vtpq_inference_pruning,
    }
    if normalized not in installers:
        supported = ", ".join(sorted(installers))
        raise NotImplementedError(
            f"VTPQ inference pruning does not support model={model_type!r}. "
            f"Supported models: {supported}."
        )
    return installers[normalized](
        model=model,
        keep_ratio=keep_ratio,
        semantic_lambda=semantic_lambda,
        score_bit=score_bit,
        protect_front=protect_front,
        positive_only=positive_only,
        min_prune_score=min_prune_score,
    )
