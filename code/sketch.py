# -*- coding: utf-8 -*-
"""Scalable DV-Lens using output-metric sketches and joint layer VJPs.

For every module weight W this computes the J-dominant coordinates

    R H^(1/2) E[J_module] W,

where H is the unembedding/output metric and R is a frozen Gaussian JL
projection.  All target modules in a layer group share one forward graph and
one set of VJP passes.  This preserves the q/k/v/o + FFN-down objects and
raw-space scoring contract while avoiding explicit d_model x d_out Jacobians.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry import (
    atomic_json_save,
    atomic_torch_save,
    cluster_labels,
    relative_sse_for_labels,
    torch_load_cpu,
)
from exact import (
    ATTN_CATEGORIES,
    FFN_CATEGORIES,
    _hash_manifest,
    _mean,
    _safe_filename,
    _sha256_file,
    _tensor,
    aggregate_paths,
    build_fixed_length_jsonl_corpus,
    discover_target_modules,
    resolve_model_parts,
    validate_module_set,
)

METHOD = "joint_layer_output_metric_sketch_jwl_qkvo_ffn_ab"
IMPLEMENTATION_VERSION = "v2_joint_vjp_jl128_gpu_accum"


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def target_key(target: dict) -> str:
    return f"L{target['layer']}:{target['key']}"


def discover_all_targets(layers: Sequence[torch.nn.Module], selected: Sequence[int]) -> list[dict]:
    result = []
    for layer_index in selected:
        local = discover_target_modules(layers[layer_index])
        validate_module_set(local)
        for target in local:
            result.append({**target, "layer": int(layer_index)})
    return result


@torch.no_grad()
def build_output_sketch(
    *, unembed: torch.nn.Module, final_norm: torch.nn.Module, d_model: int,
    sketch_dim: int, sketch_seed: int, device: str,
) -> tuple[torch.Tensor, dict]:
    """Return frozen cotangent rows R H^(1/2) in final-residual coordinates."""
    gamma = (
        final_norm.weight.detach().to(device=device, dtype=torch.float32)
        if hasattr(final_norm, "weight") and final_norm.weight is not None
        else torch.ones(d_model, device=device, dtype=torch.float32)
    )
    wu = unembed.weight.detach().to(device=device, dtype=torch.float32)
    weighted = wu * gamma.unsqueeze(0)
    h_metric = weighted.t() @ weighted
    h_metric = (h_metric + h_metric.t()) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(h_metric)
    eigenvalues = eigenvalues.clamp_min(0)
    scale = eigenvalues.sqrt()
    maximum = scale.max().clamp_min(torch.finfo(scale.dtype).tiny)
    generator = torch.Generator(device="cpu").manual_seed(int(sketch_seed))
    random_rows = torch.randn(sketch_dim, d_model, generator=generator, dtype=torch.float32)
    random_rows.div_(math.sqrt(float(sketch_dim)))
    random_rows = random_rows.to(device)
    output_sqrt = (scale / maximum).unsqueeze(1) * eigenvectors.t()
    cotangents = (random_rows @ output_sqrt).contiguous()
    metadata = {
        "sketch_dim": int(sketch_dim),
        "sketch_seed": int(sketch_seed),
        "output_metric_rank": int((eigenvalues > 0).sum().item()),
        "output_metric_max_eigenvalue": float(eigenvalues.max().item()),
        "normalization": "sqrt_eigenvalue_div_sqrt_max_then_gaussian_1_over_sqrt_r",
    }
    del wu, weighted, h_metric, eigenvalues, eigenvectors, scale, random_rows, output_sqrt
    return cotangents, metadata


def save_running(path: str, *, sums: dict[str, torch.Tensor], next_prompt: int,
                 n_done: int, prompt_count: int, sketch_dim: int) -> None:
    atomic_torch_save(
        {
            "version": "v2_joint_sketch_sum",
            "prompt_count": int(prompt_count),
            "sketch_dim": int(sketch_dim),
            "next_prompt": int(next_prompt),
            "n_done": int(n_done),
            "sums": {key: value.detach().float().cpu() for key, value in sums.items()},
        },
        path,
    )


def fit_group(
    *, decoder: torch.nn.Module, embedding: torch.nn.Module,
    layers: Sequence[torch.nn.Module], targets: Sequence[dict],
    tokenized: Sequence[torch.Tensor], cotangent_rows: torch.Tensor,
    device: str, dim_batch: int, skip_first: int,
    checkpoint_path: str, checkpoint_every: int,
) -> tuple[dict[str, torch.Tensor], int]:
    sketch_dim, d_model = map(int, cotangent_rows.shape)
    keys = [target_key(target) for target in targets]
    sums = {
        key: torch.zeros(
            sketch_dim,
            int(target["module"].weight.shape[0]),
            dtype=torch.float32,
            device=device,
        )
        for key, target in zip(keys, targets)
    }
    n_done = 0
    next_prompt = 0
    if os.path.isfile(checkpoint_path):
        state = torch_load_cpu(checkpoint_path)
        if state.get("version") != "v2_joint_sketch_sum":
            raise RuntimeError(f"unknown checkpoint: {checkpoint_path}")
        if int(state.get("prompt_count", -1)) != len(tokenized):
            raise RuntimeError(f"prompt-count mismatch: {checkpoint_path}")
        if int(state.get("sketch_dim", -1)) != sketch_dim:
            raise RuntimeError(f"sketch-dim mismatch: {checkpoint_path}")
        if set(state.get("sums", {})) != set(sums):
            raise RuntimeError(f"target-set mismatch: {checkpoint_path}")
        for key in sums:
            sums[key].copy_(state["sums"][key].float())
        n_done = int(state["n_done"])
        next_prompt = int(state["next_prompt"])
        print(f"  resume group prompt={next_prompt}/{len(tokenized)} n_done={n_done}", flush=True)

    captured: dict[str, torch.Tensor] = {}
    handles = []

    def embedding_hook(_module, _inputs, output):
        _tensor(output).requires_grad_(True)

    def final_hook(_module, _inputs, output):
        captured["__final__"] = _tensor(output)

    handles.append(embedding.register_forward_hook(embedding_hook))
    handles.append(layers[-1].register_forward_hook(final_hook))
    for key, target in zip(keys, targets):
        def module_hook(_module, _inputs, output, *, capture_key=key):
            captured[capture_key] = _tensor(output)
        handles.append(target["module"].register_forward_hook(module_hook))

    try:
        for prompt_index in range(next_prompt, len(tokenized)):
            ids = tokenized[prompt_index].to(device, non_blocking=True)
            seq_len = int(ids.numel())
            valid = torch.arange(skip_first, seq_len - 1, device=device)
            if valid.numel() == 0:
                next_prompt = prompt_index + 1
                continue
            captured.clear()
            with torch.enable_grad():
                decoder(input_ids=ids.unsqueeze(0).expand(dim_batch, -1), use_cache=False)
                final_activation = captured["__final__"]
                missing = [key for key in keys if key not in captured]
                if missing:
                    raise RuntimeError(f"target modules not called: {missing}")
                module_outputs = [captured[key] for key in keys]
                cotangent = torch.zeros_like(final_activation)
                n_passes = math.ceil(sketch_dim / dim_batch)
                for pass_index, start in enumerate(range(0, sketch_dim, dim_batch)):
                    n_this = min(dim_batch, sketch_dim - start)
                    cotangent.zero_()
                    directions = cotangent_rows[start:start + n_this].to(cotangent.dtype)
                    cotangent[:n_this, valid, :] = directions[:, None, :]
                    grads = torch.autograd.grad(
                        outputs=final_activation,
                        inputs=module_outputs,
                        grad_outputs=cotangent,
                        retain_graph=pass_index < n_passes - 1,
                        allow_unused=True,
                    )
                    for key, grad in zip(keys, grads):
                        if grad is None:
                            raise RuntimeError(f"unused module output: {key}")
                        rows = grad[:n_this, valid, :].float().mean(dim=1)
                        sums[key][start:start + n_this].add_(rows)
                    del grads, directions
            n_done += 1
            next_prompt = prompt_index + 1
            if next_prompt == 1 or next_prompt % 10 == 0 or next_prompt == len(tokenized):
                print(f"  group prompt {next_prompt}/{len(tokenized)} seq={seq_len} valid={valid.numel()}", flush=True)
            if next_prompt % max(1, checkpoint_every) == 0 or next_prompt == len(tokenized):
                save_running(
                    checkpoint_path,
                    sums=sums,
                    next_prompt=next_prompt,
                    n_done=n_done,
                    prompt_count=len(tokenized),
                    sketch_dim=sketch_dim,
                )
                print(f"  checkpoint joint sketch at prompt {next_prompt}", flush=True)
            del captured["__final__"]
    finally:
        for handle in handles:
            handle.remove()

    if n_done <= 0:
        raise RuntimeError("no valid prompt for joint sketch")
    return {key: value.div(float(n_done)).detach().cpu() for key, value in sums.items()}, n_done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--safe", required=True)
    ap.add_argument("--fit_corpus_jsonl", required=True)
    ap.add_argument("--fit_n", type=int, default=2000)
    ap.add_argument("--n_layers", type=int, default=0)
    ap.add_argument("--layer_indices", default="", help="comma-separated explicit layer indices; overrides --n_layers")
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--sketch_dim", type=int, default=128)
    ap.add_argument("--sketch_seed", type=int, default=314159)
    ap.add_argument("--layer_group_size", type=int, default=8)
    ap.add_argument("--dim_batch", type=int, default=16)
    ap.add_argument("--checkpoint_every", type=int, default=100)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jlens_path", default="jacobian-lens")
    ap.add_argument("--cache_dir", default="outputs/cache/sketch")
    ap.add_argument("--out", default="outputs/sketch.jsonl")
    args = ap.parse_args()
    if args.fit_n <= 0 or args.k < 2 or args.iters <= 0:
        ap.error("fit_n and iters must be positive; k must be >= 2")
    if args.skip_first < 0 or args.max_seq_len <= args.skip_first + 1:
        ap.error("require 0 <= skip_first < max_seq_len - 1")
    if args.sketch_dim <= 0 or args.dim_batch <= 0 or args.layer_group_size <= 0:
        raise ValueError("sketch_dim, dim_batch and layer_group_size must be positive")

    sys.path.insert(0, args.jlens_path)
    import transformers

    model_dtype = dtype_from_name(args.dtype)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    ).to(args.device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for parameter in hf.parameters():
        parameter.requires_grad_(False)
    decoder, embedding, layers, final_norm, unembed = resolve_model_parts(hf, tok)
    n_layers_total = len(layers)
    d_model = int(unembed.weight.shape[1])
    if args.layer_indices:
        selected_layers = sorted(set(int(x.strip()) for x in args.layer_indices.split(",") if x.strip()))
        if not selected_layers or min(selected_layers) < 0 or max(selected_layers) >= n_layers_total - 1:
            raise ValueError(f"invalid --layer_indices={args.layer_indices!r} for {n_layers_total} layers")
    elif args.n_layers <= 0 or args.n_layers >= n_layers_total - 1:
        selected_layers = list(range(n_layers_total - 1))
    else:
        selected_layers = sorted(set(int(x) for x in np.linspace(0, n_layers_total - 2, args.n_layers)))
    all_targets = discover_all_targets(layers, selected_layers)

    corpus_sha = _sha256_file(args.fit_corpus_jsonl)
    manifest = {
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "safe": args.safe,
        "model": args.model,
        "selected_layers": selected_layers,
        "fit_n": args.fit_n,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "sketch_dim": args.sketch_dim,
        "sketch_seed": args.sketch_seed,
        "layer_group_size": args.layer_group_size,
        "dim_batch": args.dim_batch,
        "dtype": args.dtype,
        "fit_corpus_sha256": corpus_sha,
        "module_set": "q/k/v/o + ffn_down",
        "jacobian_target": "final_block_output",
    }
    cache_path = os.path.join(args.cache_dir, args.safe, _hash_manifest(manifest))
    os.makedirs(cache_path, exist_ok=True)
    atomic_json_save(manifest, os.path.join(cache_path, "manifest.json"))

    sketch_file = os.path.join(cache_path, "output_metric_sketch.pt")
    if os.path.isfile(sketch_file):
        sketch_state = torch_load_cpu(sketch_file)
        cotangent_rows = sketch_state["cotangent_rows"].to(args.device)
        sketch_metadata = sketch_state["metadata"]
        print("output-metric sketch cache hit", flush=True)
    else:
        cotangent_rows, sketch_metadata = build_output_sketch(
            unembed=unembed,
            final_norm=final_norm,
            d_model=d_model,
            sketch_dim=args.sketch_dim,
            sketch_seed=args.sketch_seed,
            device=args.device,
        )
        atomic_torch_save(
            {"cotangent_rows": cotangent_rows.cpu(), "metadata": sketch_metadata},
            sketch_file,
        )

    prompts = build_fixed_length_jsonl_corpus(args.fit_corpus_jsonl, args.fit_n, tok, args.max_seq_len)
    tokenized = [
        tok(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids[0].cpu()
        for text in prompts
    ]
    print(
        f"[{args.safe}] layers={len(selected_layers)} fit_n={args.fit_n} sketch={args.sketch_dim} "
        f"group={args.layer_group_size} dim_batch={args.dim_batch} dtype={args.dtype}",
        flush=True,
    )

    sketch_jacobians: dict[str, torch.Tensor] = {}
    prompts_used: dict[str, int] = {}
    for group_start in range(0, len(selected_layers), args.layer_group_size):
        group_layers = selected_layers[group_start:group_start + args.layer_group_size]
        targets = [target for target in all_targets if target["layer"] in set(group_layers)]
        group_name = f"L{group_layers[0]}-{group_layers[-1]}"
        print(f"group {group_name}: layers={group_layers} targets={len(targets)}", flush=True)
        final_paths = {
            target_key(target): os.path.join(cache_path, f"S_{_safe_filename(target_key(target))}.pt")
            for target in targets
        }
        checkpoint_path = os.path.join(cache_path, f"S_{group_name}_running.pt")
        if all(os.path.isfile(path) for path in final_paths.values()):
            group_values = {key: torch_load_cpu(path).float() for key, path in final_paths.items()}
            n_used = args.fit_n
            print(f"  group cache hit ({len(group_values)})", flush=True)
        else:
            group_values, n_used = fit_group(
                decoder=decoder,
                embedding=embedding,
                layers=layers,
                targets=targets,
                tokenized=tokenized,
                cotangent_rows=cotangent_rows,
                device=args.device,
                dim_batch=args.dim_batch,
                skip_first=args.skip_first,
                checkpoint_path=checkpoint_path,
                checkpoint_every=args.checkpoint_every,
            )
            for key, path in final_paths.items():
                atomic_torch_save(group_values[key].float(), path)
            if os.path.isfile(checkpoint_path):
                os.unlink(checkpoint_path)
        sketch_jacobians.update(group_values)
        prompts_used.update({key: n_used for key in group_values})
        torch.cuda.empty_cache()

    matrix_records = []
    for target in all_targets:
        key = target_key(target)
        weight = target["module"].weight.detach().to(args.device, dtype=torch.float32)
        raw_columns = weight.t().contiguous()
        lens_columns = (sketch_jacobians[key].to(args.device) @ weight).t().contiguous()
        raw_labels, raw_k = cluster_labels(raw_columns, args.k, seed=args.seed, iters=args.iters)
        j_labels, j_k = cluster_labels(lens_columns, args.k, seed=args.seed, iters=args.iters)
        km_raw = relative_sse_for_labels(raw_columns, raw_labels, raw_k)
        km_jassign_raw = relative_sse_for_labels(raw_columns, j_labels, j_k)
        km_jassign_lens = relative_sse_for_labels(lens_columns, j_labels, j_k)
        record = {
            "layer": target["layer"],
            "module": target["path"],
            "category": target["category"],
            "shape": list(weight.shape),
            "n_prompts_used": prompts_used[key],
            "sketch_dim": args.sketch_dim,
            "km_raw": km_raw,
            "km_jassign_raw": km_jassign_raw,
            "km_jassign_lens": km_jassign_lens,
            "absdiff_jassign_raw": abs(km_jassign_raw - km_raw),
        }
        matrix_records.append(record)
        print(
            f"L{target['layer']} {target['category']:<3} {target['path']:<32} "
            f"raw={km_raw:.8f} J->R={km_jassign_raw:.8f} Jlens={km_jassign_lens:.8f}",
            flush=True,
        )
        del weight, raw_columns, lens_columns, raw_labels, j_labels

    aggregate = aggregate_paths(matrix_records)
    result = {
        "safe": args.safe,
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "nL": n_layers_total,
        "d": d_model,
        "layers": selected_layers,
        "fit_n": args.fit_n,
        "fit_corpus_kind": "fixed_length_jsonl",
        "fit_corpus_sha256": corpus_sha,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "sketch_dim": args.sketch_dim,
        "sketch_seed": args.sketch_seed,
        "layer_group_size": args.layer_group_size,
        "dim_batch": args.dim_batch,
        "dtype": args.dtype,
        "k": args.k,
        "iters": args.iters,
        "aggregation": "arithmetic_mean_of_per_matrix_relative_sse",
        "path_a": "sketched_module_JWL_assignment_on_qkvo_and_ffn_down_scored_raw",
        "path_b": "raw_kmeans_qkvo_plus_sketched_JWL_assignment_ffn_down_scored_raw",
        "cache_path": cache_path,
        "output_sketch": sketch_metadata,
        "matrix_records": matrix_records,
        **aggregate,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
