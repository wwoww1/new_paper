import functools
import inspect
import math
import statistics
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

import torch


PREFILL_PROFILE_MODELS = frozenset(
    {"internvl2", "qwen2_vl", "llava_onevision"}
)


def _normalize_model_type(model_type: str) -> str:
    return str(model_type).strip().lower().replace("-", "_")


def _extract_call_argument(original, args, kwargs, name):
    if name in kwargs:
        return kwargs[name]
    try:
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    return bound.arguments.get(name)


def _cache_sequence_length(cache: Any) -> int:
    if cache is None:
        return 0

    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length())
        except (TypeError, ValueError):
            pass

    key_cache = getattr(cache, "key_cache", None)
    if key_cache is not None and len(key_cache) > 0:
        first = key_cache[0]
        if torch.is_tensor(first) and first.ndim >= 2:
            return int(first.shape[-2])

    if isinstance(cache, (tuple, list)) and cache:
        first_layer = cache[0]
        if isinstance(first_layer, (tuple, list)) and first_layer:
            first = first_layer[0]
            if torch.is_tensor(first) and first.ndim >= 2:
                return int(first.shape[-2])
    return 0


def _iter_cache_tensors(cache: Any) -> Iterable[torch.Tensor]:
    if cache is None:
        return

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None or value_cache is not None:
        for collection in (key_cache, value_cache):
            if collection is None:
                continue
            for tensor in collection:
                if torch.is_tensor(tensor):
                    yield tensor
        return

    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values", "key", "value"):
                tensor = getattr(layer, name, None)
                if torch.is_tensor(tensor):
                    yield tensor
        return

    if isinstance(cache, dict):
        for value in cache.values():
            yield from _iter_cache_tensors(value)
        return

    if isinstance(cache, (tuple, list)):
        for value in cache:
            if torch.is_tensor(value):
                yield value
            else:
                yield from _iter_cache_tensors(value)


def kv_cache_bytes(cache: Any) -> int:
    total = 0
    seen: Set[int] = set()
    for tensor in _iter_cache_tensors(cache):
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class PrefillProfilerStats:
    model_type: str
    warmup_samples: int = 10
    max_samples: int = 200
    seen_prefills: int = 0
    prefill_time_ms: List[float] = field(default_factory=list)
    kv_cache_mib: List[float] = field(default_factory=list)
    prompt_tokens: List[int] = field(default_factory=list)

    @property
    def samples(self) -> int:
        return len(self.prefill_time_ms)

    @property
    def complete(self) -> bool:
        return self.max_samples > 0 and self.samples >= self.max_samples

    def should_measure(self) -> bool:
        return not self.complete

    def update(self, elapsed_ms: float, cache: Any, fallback_tokens: int) -> None:
        self.seen_prefills += 1
        if self.seen_prefills <= self.warmup_samples:
            return
        if self.complete:
            return

        cache_bytes = kv_cache_bytes(cache)
        sequence_length = _cache_sequence_length(cache) or int(fallback_tokens)
        self.prefill_time_ms.append(float(elapsed_ms))
        self.kv_cache_mib.append(float(cache_bytes) / (1024.0 ** 2))
        self.prompt_tokens.append(sequence_length)

    def to_dict(self, mean_score_time_ms: float = 0.0) -> Dict[str, Any]:
        mean_prefill = (
            statistics.fmean(self.prefill_time_ms)
            if self.prefill_time_ms
            else 0.0
        )
        mean_cache = (
            statistics.fmean(self.kv_cache_mib)
            if self.kv_cache_mib
            else 0.0
        )
        mean_tokens = (
            statistics.fmean(self.prompt_tokens)
            if self.prompt_tokens
            else 0.0
        )
        return {
            "model_type": self.model_type,
            "warmup_samples": self.warmup_samples,
            "requested_samples": self.max_samples,
            "samples": self.samples,
            "cache_samples": sum(value > 0.0 for value in self.kv_cache_mib),
            "mean_llm_prefill_ms": mean_prefill,
            "median_llm_prefill_ms": (
                statistics.median(self.prefill_time_ms)
                if self.prefill_time_ms
                else 0.0
            ),
            "p90_llm_prefill_ms": _percentile(self.prefill_time_ms, 0.90),
            "mean_score_time_ms": float(mean_score_time_ms),
            "mean_prefill_with_score_ms": mean_prefill + float(mean_score_time_ms),
            "mean_kv_cache_mib": mean_cache,
            "median_kv_cache_mib": (
                statistics.median(self.kv_cache_mib)
                if self.kv_cache_mib
                else 0.0
            ),
            "mean_prompt_tokens": mean_tokens,
        }


def _locate_llm_backbone(model_type: str, model):
    normalized = _normalize_model_type(model_type)
    if normalized == "qwen2_vl":
        target = getattr(model, "model", None)
    elif normalized == "internvl2":
        language_model = getattr(model, "language_model", None)
        target = getattr(language_model, "model", language_model)
    elif normalized == "llava_onevision":
        target = getattr(model, "model", None)
    else:
        supported = ", ".join(sorted(PREFILL_PROFILE_MODELS))
        raise NotImplementedError(
            f"Prefill profiling does not support model={model_type!r}. "
            f"Supported models: {supported}."
        )

    if target is None or not callable(getattr(target, "forward", None)):
        raise TypeError(
            f"Cannot locate the LLM backbone forward method for {model_type}."
        )
    return target


def _module_cuda_devices(module) -> List[torch.device]:
    devices = set()
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if parameter.is_cuda:
                devices.add(parameter.device)
    return sorted(devices, key=lambda device: (device.type, device.index or 0))


def _synchronize(devices: Iterable[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _input_sequence_length(original, args, kwargs) -> int:
    for name in ("inputs_embeds", "input_ids"):
        value = _extract_call_argument(original, args, kwargs, name)
        if torch.is_tensor(value) and value.ndim >= 2:
            return int(value.shape[-2] if name == "inputs_embeds" else value.shape[-1])
    return 0


def _is_prefill_call(original, args, kwargs) -> bool:
    use_cache = _extract_call_argument(original, args, kwargs, "use_cache")
    if use_cache is False:
        return False
    past_key_values = _extract_call_argument(
        original, args, kwargs, "past_key_values"
    )
    return _cache_sequence_length(past_key_values) == 0 and _input_sequence_length(
        original, args, kwargs
    ) > 0


def install_prefill_profiler(
    model_type: str,
    model,
    warmup_samples: int = 10,
    max_samples: int = 200,
) -> PrefillProfilerStats:
    if warmup_samples < 0:
        raise ValueError("warmup_samples must be non-negative.")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    target = _locate_llm_backbone(model_type, model)
    if getattr(target, "_vtpq_prefill_profiler_installed", False):
        raise RuntimeError("A prefill profiler is already installed on this model.")

    stats = PrefillProfilerStats(
        model_type=_normalize_model_type(model_type),
        warmup_samples=int(warmup_samples),
        max_samples=int(max_samples),
    )
    original_forward = target.forward
    module_devices = _module_cuda_devices(target)

    @functools.wraps(original_forward)
    def wrapped_forward(_self, *args, **kwargs):
        profile_this_call = stats.should_measure() and _is_prefill_call(
            original_forward, args, kwargs
        )
        if not profile_this_call:
            return original_forward(*args, **kwargs)

        devices = module_devices
        if not devices:
            for name in ("inputs_embeds", "input_ids"):
                tensor = _extract_call_argument(original_forward, args, kwargs, name)
                if torch.is_tensor(tensor) and tensor.is_cuda:
                    devices = [tensor.device]
                    break

        _synchronize(devices)
        start = time.perf_counter()
        output = original_forward(*args, **kwargs)
        _synchronize(devices)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        cache = getattr(output, "past_key_values", None)
        if cache is None and isinstance(output, (tuple, list)) and len(output) > 1:
            cache = output[1]
        stats.update(
            elapsed_ms=elapsed_ms,
            cache=cache,
            fallback_tokens=_input_sequence_length(original_forward, args, kwargs),
        )
        return output

    target.forward = types.MethodType(wrapped_forward, target)
    target._vtpq_prefill_profiler_installed = True
    target._vtpq_prefill_profiler_stats = stats
    return stats
