import argparse
import datetime
import importlib
import json
import os
import re
import sys
import time
import traceback
import warnings
from functools import partial
from PIL import Image
import torch
import numpy as np
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from typing import Union, List, Dict, Any

from lmms_eval.models import get_model

from qmllm.quantization.quant_wrapper import qwrapper
from qmllm.models import get_process_model
from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN
except Exception:
    DEFAULT_IMAGE_TOKEN = "<image>"

from qmllm.methods.vtpq.quantize.token_reducer import (
    reduce_inference_tokens,
)
from qmllm.methods.vtpq.inference_pruning import (
    VTPQ_INFERENCE_PRUNING_MODELS,
    install_vtpq_inference_pruning,
)


def parse_quant_infer_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    # calibration parameters
    parser.add_argument("--calib_data", default="pileval", choices=["pileval", "coco", None])
    parser.add_argument("--n_samples", default=128, type=int)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)
    parser.add_argument("--method", default="awq", choices=["awq", "smoothquant", "mbq", "vtpq", "rtn", None])
    parser.add_argument("--w_bit", default=8, type=int)
    parser.add_argument("--a_bit", default=16, type=int)
    parser.add_argument("--w_group", default=128, type=int)
    parser.add_argument("--alpha", default=0.5, type=int)
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--distort", action="store_true")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"])
    parser.add_argument("--scale_path", default=None, type=str)
    parser.add_argument("--run_process", action="store_true")
    parser.add_argument("--pseudo_quant", action="store_true")
    parser.add_argument("--vtpq_keep_ratio", default=0.75, type=float)
    parser.add_argument("--vtpq_lambda", default=0.30, type=float)
    parser.add_argument("--vtpq_score_bit", default=8, type=int)
    parser.add_argument("--vtpq_always_keep_front", default=0, type=int)
    vtpq_gate_group = parser.add_mutually_exclusive_group()
    vtpq_gate_group.add_argument(
        "--vtpq_positive_only",
        dest="vtpq_positive_only",
        action="store_true",
        help=(
            "Use keep_ratio as a maximum calibration-pruning budget and "
            "drop only tokens with positive predicted quantization gain."
        ),
    )
    vtpq_gate_group.add_argument(
        "--vtpq_force_pruning_quota",
        dest="vtpq_positive_only",
        action="store_false",
        help="Always consume the full calibration pruning quota.",
    )
    parser.set_defaults(vtpq_positive_only=True)
    parser.add_argument(
        "--vtpq_min_prune_score",
        default=0.0,
        type=float,
        help="Minimum normalized prune score used for calibration positive-only gating.",
    )
    parser.add_argument(
        "--vtpq_scale_loss",
        default="robust",
        choices=["robust"],
    )
    parser.add_argument("--vtpq_tail_ratio", default=0.20, type=float)
    parser.add_argument("--vtpq_tail_weight", default=0.25, type=float)
    vtpq_infer_gate_group = parser.add_mutually_exclusive_group()
    vtpq_infer_gate_group.add_argument(
        "--vtpq_infer_positive_only",
        dest="vtpq_infer_positive_only",
        action="store_true",
        help=(
            "Treat vtpq_keep_ratio as a maximum inference-pruning budget and "
            "drop only tokens with positive predicted gain and prune score "
            "above vtpq_infer_min_score."
        ),
    )
    vtpq_infer_gate_group.add_argument(
        "--vtpq_infer_force_pruning_quota",
        dest="vtpq_infer_positive_only",
        action="store_false",
        help="Always consume the full inference pruning quota.",
    )
    parser.set_defaults(vtpq_infer_positive_only=True)
    parser.add_argument("--vtpq_infer_min_score", default=0.0, type=float)
    parser.add_argument(
        "--vtpq_score_source",
        default="embed",
        choices=["embed", "layer"],
    )
    parser.add_argument("--vtpq_score_layers", default=4, type=int)
    
    # Independent inference-pruning switch.
    # Keeping this independent from --method allows the 2x2 ablation.
    parser.add_argument("--vtpq_infer_prune", action="store_true") 
    ## inference parameters
    parser.add_argument("--infer_pairs", default=None, type=str,
                        help="Path to a JSON or JSONL file. Each item: "
                             '{"images": ["a.jpg", "b.png"] or "a.jpg", "question": "...", "id": "optional"}')
    parser.add_argument("--save_path", default=None, type=str,
                        help="Where to save inference outputs as a JSON list.")
    parser.add_argument("--max_new_tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--do_sample", action="store_true")

    
    args = parser.parse_args()
    return args


def cli_quant(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_quant_infer_args()

    args_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    for args in args_list:
        cli_quant_single(args)


def cli_quant_single(args: Union[argparse.Namespace, None] = None) -> None:
    # here we load MLLMs outside of the evaluator.
    if args.model_args is None:
        args.model_args = ""

    if not 0.0 < args.vtpq_keep_ratio <= 1.0:
        raise ValueError("--vtpq_keep_ratio must be in (0, 1].")
    normalized_model = args.model.lower().replace("-", "_")
    if (
        args.vtpq_infer_prune
        and normalized_model not in VTPQ_INFERENCE_PRUNING_MODELS
    ):
        supported = ", ".join(sorted(VTPQ_INFERENCE_PRUNING_MODELS))
        raise NotImplementedError(
            "Inference-stage VTPQ does not support "
            f"--model {args.model}. Supported models: {supported}."
        )
    if args.vtpq_infer_prune and args.vtpq_score_source != "embed":
        raise NotImplementedError(
            "Inference-stage VTPQ currently supports --vtpq_score_source embed only. "
            "Layer-based scoring would require a dense early-layer forward pass."
        )

    vtpq_infer_config = {
        "enabled": bool(args.vtpq_infer_prune),
        "keep_ratio": args.vtpq_keep_ratio,
        "semantic_lambda": args.vtpq_lambda,
        "score_bit": args.vtpq_score_bit,
        "protect_front": args.vtpq_always_keep_front,
        "positive_only": bool(args.vtpq_infer_positive_only),
        "min_prune_score": float(args.vtpq_infer_min_score),
    }  
    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": args.batch_size,
            "device": args.device,
            # "use_flash_attention_2": False,
        },
    )

    # Preprocess the MLLM here, use "lm._model" to get the fp16 mllm.
    Process_ModelClass = get_process_model(args.model)
    process_model = Process_ModelClass(lm._model, 
                                       lm._tokenizer, 
                                       lm.processor if hasattr(lm, 'processor') else None)

    # Generate the calibration tokens.
    prompt_inputs = None
    prompt_kwargs = None

    if args.run_process:
        if args.calib_data == "pileval":
            prompt_inputs, prompt_kwargs = get_calib_dataset(
                data_path=args.data_path,
                tokenizer=lm._tokenizer,
                n_samples=args.n_samples,
            )
        elif args.calib_data == "coco":
            prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(
                data_path=args.data_path,
                image_folder=args.image_folder,
                model=process_model,
                n_samples=args.n_samples,
                few_shot_format=args.few_shot_format,
                interleave_format=args.interleave_format,
                text_data_path=args.text_data_path,
            )

    # Wrapper the quantized model.
    qwrapper(process_model, prompt_inputs, prompt_kwargs, args)

    # The standalone InternVL2 runner already prunes the prepared embeddings
    # directly. Qwen2-VL and LLaVA-OneVision use the same model hooks as
    # main.py so pruning happens after vision encoding and before LLM prefill.
    if args.vtpq_infer_prune and normalized_model != "internvl2":
        install_vtpq_inference_pruning(
            model_type=normalized_model,
            model=lm._model,
            keep_ratio=args.vtpq_keep_ratio,
            semantic_lambda=args.vtpq_lambda,
            score_bit=args.vtpq_score_bit,
            protect_front=args.vtpq_always_keep_front,
            positive_only=args.vtpq_infer_positive_only,
            min_prune_score=args.vtpq_infer_min_score,
        )
    
    
    if args.infer_pairs and args.save_path:
        quant_meta = {
            "method": args.method,
            "w_bit": args.w_bit,
            "a_bit": args.a_bit,
            "w_group": args.w_group,
            "alpha": args.alpha,
            "reweight": bool(args.reweight),
            "distort": bool(args.distort),
            "loss_mode": args.loss_mode,
            "scale_path": args.scale_path,
            "pseudo_quant": bool(args.pseudo_quant),
            "vtpq_keep_ratio": args.vtpq_keep_ratio,
            "vtpq_lambda": args.vtpq_lambda,
            "vtpq_score_bit": args.vtpq_score_bit,
            "vtpq_score_source": args.vtpq_score_source,
            "vtpq_positive_only": bool(args.vtpq_positive_only),
            "vtpq_min_prune_score": float(args.vtpq_min_prune_score),
            "vtpq_scale_loss": args.vtpq_scale_loss,
            "vtpq_tail_ratio": args.vtpq_tail_ratio,
            "vtpq_tail_weight": args.vtpq_tail_weight,
            "vtpq_infer_prune": bool(args.vtpq_infer_prune),
            "vtpq_infer_positive_only": bool(args.vtpq_infer_positive_only),
            "vtpq_infer_min_score": float(args.vtpq_infer_min_score),
        }

        run_inference(
            arch=args.model,
            process_model=process_model, 
            lm_model=lm._model,
            tokenizer=lm._tokenizer,
            infer_pairs=args.infer_pairs,
            save_path=args.save_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            model_args_str=args.model_args,
            quant_meta=quant_meta,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            vtpq_infer_config=vtpq_infer_config,
        )

#################### inferece ######################
def load_pairs(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"--infer_pairs not found: {path}")

    def normalize_item(it, idx):
        if "images" not in it and "image" in it:
            it["images"] = it.pop("image")
        imgs = it.get("images", [])
        if isinstance(imgs, (str, os.PathLike)):
            imgs = [str(imgs)]
        elif imgs is None:
            imgs = []
        elif isinstance(imgs, tuple):
            imgs = list(imgs)
        it["images"] = imgs

        q = it.get("question", "")
        if not isinstance(q, str):
            raise ValueError(f"[pairs[{idx}]] 'question' must be str, got {type(q)}")
        q = re.sub(r"\s*<image>\s*", " ", q, flags=re.IGNORECASE).strip()
        it["question"] = q

        it.setdefault("id", idx)
        return it

    items = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    items.append(normalize_item(json.loads(line), i))
        return items

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items = data if isinstance(data, list) else data.get("data", [])
    if not isinstance(raw_items, list):
        raise TypeError(f"Unsupported JSON root type: {type(data)}")

    return [normalize_item(obj, i) for i, obj in enumerate(raw_items)]


def ensure_list_images(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def open_pils(paths: List[str]):
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs

def pick_language_model(top_model):
    m = top_model
    base = getattr(m, "model", None) or getattr(m, "_model", None) or m
    for name in ["language_model", "llm", "lm", "transformer"]:
        llm = getattr(base, name, None)
        if llm is not None and hasattr(llm, "generate"):
            return llm
    if hasattr(m, "generate"):
        return m
    raise RuntimeError("Cannot locate underlying language model (ForCausalLM) for generate().")


def is_dispatched_model(model):
    return bool(getattr(model, "hf_device_map", None))


@torch.inference_mode()
def maybe_vtpq_prune_inputs(
    prompt_inputs,
    prompt_kwargs,
    enabled=False,
    keep_ratio=0.75,
    semantic_lambda=0.30,
    score_bit=8,
    protect_front=0,
    positive_only=True,
    min_prune_score=0.0,
):
    if not enabled:
        return prompt_inputs, prompt_kwargs, None

    embeds = prompt_inputs["inputs_embeds"]

    if embeds.is_cuda:
        torch.cuda.synchronize(embeds.device)
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

    if embeds.is_cuda:
        torch.cuda.synchronize(embeds.device)

    info["score_time_ms"] = (time.perf_counter() - start) * 1000.0
    return reduced_inputs, reduced_kwargs, info


@torch.inference_mode()
def safe_generate_with_embeds(
    top_model,
    tokenizer,
    inputs_embeds,
    attention_mask,
    **gen_kwargs,
):
    generate_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        **gen_kwargs,
    }

    # LLaVA's custom generate() explicitly rejects inputs_embeds.
    # Bypass only that override and call its parent generation method.
    if "llava" in type(top_model).__name__.lower():
        parent_generate = super(type(top_model), top_model).generate
        return parent_generate(**generate_kwargs)

    llm = pick_language_model(top_model)
    if llm is not top_model:
        if (
            hasattr(llm, "to")
            and not is_dispatched_model(top_model)
            and not is_dispatched_model(llm)
        ):
            llm.to(inputs_embeds.device)
        llm.eval()
        return llm.generate(**generate_kwargs)

    return top_model.generate(**generate_kwargs)

# =========================
# llava
# =========================
@torch.inference_mode()
def infer_llava_onevision(
    proc_model,
    raw_model,
    tokenizer,
    device: str,
    pairs: List[Dict[str, Any]],
    max_new_tokens=256,
    temperature=0.2,
    do_sample=False,
) -> List[Dict[str, Any]]:
    raw_model.eval()
    if hasattr(raw_model, "to"):
        raw_model.to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    results = []

    for i, item in enumerate(pairs):
        q = item.get("question", "")
        img_paths = ensure_list_images(item.get("images"))
        pils = open_pils(img_paths) if img_paths else None
        n_img = len(pils) if pils else 0

        if n_img > 0:
            img_tokens = (DEFAULT_IMAGE_TOKEN + "\n") * n_img
            human_value = f"{img_tokens}{q}".strip()
        else:
            human_value = q

        data_item = {
            "id": item.get("id", i),
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt",   "value": ""}
            ],
            "image": img_paths if pils else None
        }

        data_dict = proc_model.preprocess_data(pils, data_item)
        batch = proc_model.data_collator([data_dict])

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        param_dtype = next(raw_model.parameters()).dtype
        images = [im.to(device=device, dtype=param_dtype) for im in batch["images"]] if "images" in batch else None
        image_sizes = batch.get("image_sizes", None)
        modalities = batch.get("modalities", None)

        gen_ids = raw_model.generate(
            inputs=input_ids,
            images=images,
            image_sizes=image_sizes,
            modalities=modalities,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=True,
            eos_token_id=getattr(raw_model.generation_config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=pad_id,
        )

        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        if q and full_text.startswith(q):
            answer = full_text[len(q):].strip()
        else:
            prompt_len = int(input_ids[0].ne(pad_id).sum().item())
            tail_ids = gen_ids[0][prompt_len:] if gen_ids.shape[1] > prompt_len else gen_ids[0]
            answer = tokenizer.decode(tail_ids, skip_special_tokens=True).strip()

        results.append({
            "id": data_item["id"],
            "question": q,
            "images": img_paths,
            "answer": answer,
        })

    return results

# =========================
# InternVL2
# =========================
@torch.inference_mode()
def infer_internvl2(
    proc_model,
    raw_model,
    tokenizer,
    device: str,
    pairs: List[Dict[str, Any]],
    max_new_tokens=256,
    temperature=0.2,
    do_sample=False,
    vtpq_infer_config=None,
) -> List[Dict[str, Any]]:
    raw_model.eval()
    if hasattr(raw_model, "to") and not is_dispatched_model(raw_model):
        raw_model.to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    results = []
    prune_config = vtpq_infer_config or {}

    for i, item in enumerate(pairs):
        q = item.get("question", "")
        img_paths = ensure_list_images(item.get("images"))
        if bool(prune_config.get("enabled", False)) and len(img_paths) > 1:
            raise NotImplementedError(
                "InternVL2 inference-stage VTPQ currently supports one source image per sample. "
                "Dynamic tiling of that image is supported, but multiple independent images "
                "need per-image token-budget constraints."
            )
        pils = open_pils(img_paths) if img_paths else None

        data_item = {
            "id": item.get("id", i),
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt",   "value": ""}
            ],
            "image": img_paths if pils else None
        }

        data_dict = proc_model.preprocess_data(pils, data_item)
        batch = proc_model.data_collator([data_dict])
        prompt_inputs, prompt_kwargs = proc_model.generate_input(batch)

        prompt_inputs, prompt_kwargs, prune_info = maybe_vtpq_prune_inputs(
            prompt_inputs=prompt_inputs,
            prompt_kwargs=prompt_kwargs,
            enabled=bool(prune_config.get("enabled", False)),
            keep_ratio=float(prune_config.get("keep_ratio", 0.75)),
            semantic_lambda=float(prune_config.get("semantic_lambda", 0.30)),
            score_bit=int(prune_config.get("score_bit", 8)),
            protect_front=int(prune_config.get("protect_front", 0)),
            positive_only=bool(prune_config.get("positive_only", False)),
            min_prune_score=float(prune_config.get("min_prune_score", 0.0)),
        )

        inputs_embeds = prompt_inputs["inputs_embeds"].to(device)
        attention_mask = prompt_kwargs["attention_mask"].to(device)

        gen_ids = safe_generate_with_embeds(
            top_model=raw_model,
            tokenizer=tokenizer,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=True,
            eos_token_id=getattr(raw_model.generation_config, "eos_token_id", tokenizer.eos_token_id),
            pad_token_id=pad_id,
        )

        # With decoder-only generation from inputs_embeds, Transformers 4.49
        # returns generated token ids without the original prompt prefix.
        ans = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        result = {
            "id": data_item["id"],
            "question": q,
            "images": img_paths,
            "answer": ans,
        }

        if prune_info is not None:
            result["pruning"] = {
                "total_visual_tokens": prune_info["total_visual_tokens"],
                "kept_visual_tokens": prune_info["kept_visual_tokens"],
                "dropped_visual_tokens": prune_info["dropped_visual_tokens"],
                "actual_keep_ratio": prune_info["actual_keep_ratio"],
                "score_time_ms": prune_info["score_time_ms"],
            }

        results.append(result)

    return results

# =========================
# Qwen2-VL
# =========================
@torch.inference_mode()
def infer_qwen2_vl(
    proc_model, raw_model, tokenizer, device: str, pairs, 
    max_new_tokens=256, temperature=0.2, do_sample=False
):
    raw_model.eval()
    if hasattr(raw_model, "to"):
        raw_model.to(device)

    processor = proc_model.processor
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    results = []

    for i, item in enumerate(pairs):
        q = item.get("question", "")
        img_paths = ensure_list_images(item.get("images"))
        pils = open_pils(img_paths) if img_paths else None

        user_content = []
        if pils:
            for _ in pils:
                user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": q})
        chat_item = [{"role": "user", "content": user_content}]

        prompt_text = processor.apply_chat_template(
            chat_item, tokenize=False, add_generation_prompt=True
        )

        data_dict = processor(
            text=prompt_text, images=pils, videos=None,
            padding=True, return_tensors="pt",
        )

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 1 and v.size(0) == 1:
                data_dict[k] = v.squeeze(0)

        samples = {
            "input_ids":        data_dict["input_ids"].unsqueeze(0),
            "attention_mask":   data_dict["attention_mask"].unsqueeze(0),
            "labels":           torch.full_like(data_dict["input_ids"].unsqueeze(0), -100),
            "pixel_values":     data_dict["pixel_values"],            # [B_total, C, H, W]
            "image_grid_thw":   data_dict["image_grid_thw"].unsqueeze(0),
        }

        if getattr(raw_model, "_vtpq_inference_pruning_installed", False):
            input_ids = samples["input_ids"].to(device)
            attention_mask = samples["attention_mask"].to(device)
            pixel_values = samples["pixel_values"].to(device)
            image_grid_thw = samples["image_grid_thw"].to(device)
            gen_ids = raw_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                temperature=temperature,
                do_sample=do_sample,
                use_cache=True,
                eos_token_id=getattr(raw_model.generation_config, "eos_token_id", tokenizer.eos_token_id),
                pad_token_id=pad_id,
            )
        else:
            prompt_inps, prompt_kwargs = proc_model.generate_input(samples)
            inputs_embeds = prompt_inps["inputs_embeds"].to(device)
            attention_mask = prompt_kwargs["attention_mask"].to(device)

            gen_ids = safe_generate_with_embeds(
                top_model=raw_model,
                tokenizer=tokenizer,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                temperature=temperature,
                do_sample=do_sample,
                use_cache=True,
                eos_token_id=getattr(raw_model.generation_config, "eos_token_id", tokenizer.eos_token_id),
                pad_token_id=pad_id,
            )

        prompt_len = int(attention_mask[0].sum().item())
        ans_ids = gen_ids[0][prompt_len:] if gen_ids.shape[1] > prompt_len else gen_ids[0]
        ans = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()

        if ans.lower().startswith("assistant"):
            ans = ans[len("assistant"):].lstrip(":： \n\t")

        results.append({
            "id": item.get("id", i),
            "question": q,
            "images": img_paths,
            "answer": ans
        })
    return results


def run_inference(
    arch: str,               # "llava_onevision" | "internvl2" | "qwen2_vl"
    process_model,
    lm_model,
    tokenizer,
    infer_pairs: str,        # input data（.json/.jsonl）
    save_path: str,          # save path
    max_new_tokens=256,
    temperature=0.2,
    do_sample=False,
    model_args_str: str = "",
    quant_meta: Dict[str, Any] = None,
    device: str = None,
    vtpq_infer_config=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(infer_pairs)

    if arch.lower() == "llava_onevision":
        outputs = infer_llava_onevision(
            proc_model=process_model,
            raw_model=lm_model,
            tokenizer=tokenizer,
            device=device,
            pairs=pairs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
    elif arch.lower() == "internvl2":
        outputs = infer_internvl2(
            proc_model=process_model,
            raw_model=lm_model,
            tokenizer=tokenizer,
            device=device,
            pairs=pairs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            vtpq_infer_config=vtpq_infer_config,
        )
    elif arch.lower() == "qwen2_vl":
        outputs = infer_qwen2_vl(
            proc_model=process_model,
            raw_model=lm_model,
            tokenizer=tokenizer,
            device=device,
            pairs=pairs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    export = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "arch": arch,
            "model_args": model_args_str,
            "quant": quant_meta or {},
            "inference_pruning": vtpq_infer_config or {"enabled": False},
        },
        "results": outputs
    }
    pruning_stats = getattr(lm_model, "_vtpq_inference_pruning_stats", None)
    if pruning_stats is not None:
        export["meta"]["inference_pruning_stats"] = pruning_stats.to_dict()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {len(outputs)} results to: {save_path}")



if __name__ == "__main__":
    model = cli_quant()
    
