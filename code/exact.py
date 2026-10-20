"""Dense module-specific downstream Jacobians and DV-Complexity.

Fit the average mapping from native projection outputs to the final block
output. Cluster weight columns in the induced vocabulary metric, then score
the fixed partition by relative reconstruction SSE in the original weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Sequence
from typing import Optional

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


METHOD = "module_jacobian_jwl_qkvo_ffn_ab"
IMPLEMENTATION_VERSION = "v1_module_output_to_final_residual"
ATTN_CATEGORIES = frozenset(("q", "k", "v", "o", "qkv"))
FFN_CATEGORIES = frozenset(("ffn",))


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _tensor(output):
    return output if torch.is_tensor(output) else output[0]


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _hash_manifest(manifest: dict) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixed_length_jsonl_corpus(
    path: str,
    n: int,
    tokenizer,
    max_seq_len: int,
) -> list[str]:
    """Pack a deterministic JSONL text stream into N full-length sequences.

    Packing is tokenizer-specific so every retained string reaches exactly
    ``max_seq_len`` after the same truncating encode used by the estimator.
    JSON objects may contain a string ``text`` or a list of strings ``texts``.
    """
    if n <= 0 or max_seq_len <= 0:
        raise ValueError((n, max_seq_len))
    prompts: list[str] = []
    pending: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            values = row.get("texts") if isinstance(row, dict) else None
            if values is None:
                values = [row.get("text")] if isinstance(row, dict) else []
            if not isinstance(values, list):
                raise ValueError(f"{path}:{line_number}: texts must be a list")
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    continue
                pending.append(value.strip())
                candidate = "\n\n".join(pending)
                encoded = tokenizer(
                    candidate,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                ).input_ids
                if int(encoded.shape[1]) == max_seq_len:
                    prompts.append(candidate)
                    pending = []
                    if len(prompts) == n:
                        return prompts
    raise RuntimeError(
        f"corpus {path} produced only {len(prompts)}/{n} full "
        f"{max_seq_len}-token sequences"
    )


def discover_target_modules(block: torch.nn.Module) -> list[dict]:
    """Return supported q/k/v/o and FFN-down linear modules in one block.

    Split q/k/v projections are preferred.  Fused QKV projections are kept as
    one matrix, matching the object exposed by the checkpoint.  FFN means the
    residual-writing down projection, used by this method.
    """
    targets = []
    seen = set()
    for path, module in block.named_modules():
        if not isinstance(module, torch.nn.Linear) or id(module) in seen:
            continue
        leaf = path.split(".")[-1].lower()
        lower = path.lower()
        category = None
        if leaf in {"q_proj", "wq", "query"}:
            category = "q"
        elif leaf in {"k_proj", "wk", "key"}:
            category = "k"
        elif leaf in {"v_proj", "wv", "value"}:
            category = "v"
        elif leaf in {"qkv_proj", "query_key_value", "wqkv", "w_qkv"}:
            category = "qkv"
        elif leaf in {"o_proj", "wo", "out_proj"}:
            category = "o"
        elif leaf == "dense" and any(x in lower for x in ("attn", "attention")):
            category = "o"
        elif leaf in {"down_proj", "w2", "dense_4h_to_h", "fc2"}:
            category = "ffn"
        if category is None:
            continue
        seen.add(id(module))
        targets.append(
            {
                "key": f"{category}:{path}",
                "category": category,
                "path": path,
                "module": module,
            }
        )
    return targets


def validate_module_set(targets: Sequence[dict]) -> None:
    categories = {target["category"] for target in targets}
    if not (categories & ATTN_CATEGORIES):
        raise RuntimeError("no supported q/k/v/o projection found")
    if "ffn" not in categories:
        raise RuntimeError("no supported FFN down projection found")


def aggregate_paths(matrix_records: Sequence[dict]) -> dict:
    """Arithmetic-mean aggregation for raw, path A, and path B."""
    if not matrix_records:
        raise ValueError("no matrix records")
    raw_all = [record["km_raw"] for record in matrix_records]
    path_a = [record["km_jassign_raw"] for record in matrix_records]
    path_b = [
        record["km_jassign_raw"]
        if record["category"] in FFN_CATEGORIES
        else record["km_raw"]
        for record in matrix_records
    ]
    qkvo = [record for record in matrix_records if record["category"] in ATTN_CATEGORIES]
    ffn = [record for record in matrix_records if record["category"] in FFN_CATEGORIES]
    return {
        "km_raw_qkvo_ffn": _mean(raw_all),
        "km_path_a_jwl_qkvo_ffn": _mean(path_a),
        "km_path_b_ffn_jwl_qkvo_raw": _mean(path_b),
        "km_raw_qkvo": _mean([record["km_raw"] for record in qkvo]),
        "km_jwl_qkvo": _mean([record["km_jassign_raw"] for record in qkvo]),
        "km_raw_ffn": _mean([record["km_raw"] for record in ffn]),
        "km_jwl_ffn": _mean([record["km_jassign_raw"] for record in ffn]),
        "km_jwl_output_qkvo_ffn": _mean([record["km_jassign_lens"] for record in matrix_records]),
        "km_jwl_output_qkvo": _mean([record["km_jassign_lens"] for record in qkvo]),
        "km_jwl_output_ffn": _mean([record["km_jassign_lens"] for record in ffn]),
        "n_matrices": len(matrix_records),
        "n_qkvo_matrices": len(qkvo),
        "n_ffn_matrices": len(ffn),
    }


@torch.no_grad()
def output_metric_coordinates(
    weight: torch.Tensor,
    module_jacobian: torch.Tensor,
    h_eigenvectors: torch.Tensor,
    h_eigenvalues: torch.Tensor,
) -> torch.Tensor:
    """Coordinates of weight columns in W_U Gamma J_module output geometry."""
    transported = module_jacobian.float() @ weight.float()
    rotated = h_eigenvectors.float().t() @ transported
    scale = h_eigenvalues.float().clamp_min(0).sqrt()
    maximum = scale.max().clamp_min(torch.finfo(scale.dtype).tiny)
    return ((scale / maximum).unsqueeze(1) * rotated).t().contiguous()


def fit_module_jacobians(
    *,
    decoder: torch.nn.Module,
    embedding: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    layer_index: int,
    targets: Sequence[dict],
    tokenizer,
    prompts: Sequence[str],
    d_model: int,
    device: str,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 25,
    target_module: Optional[torch.nn.Module] = None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Fit d(target activation)/d(module output) for all targets in one layer.

    ``target_module`` defaults to the final decoder block. Callers can pass
    another target module explicitly, for example the final normalization.
    """
    sums = {
        target["key"]: torch.zeros(
            d_model,
            int(target["module"].weight.shape[0]),
            dtype=torch.float32,
        )
        for target in targets
    }
    n_done = 0
    next_prompt = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch_load_cpu(checkpoint_path)
        if state.get("version") != "v1_module_j_sum":
            raise RuntimeError(f"unknown checkpoint version: {checkpoint_path}")
        if state.get("prompt_count") != len(prompts):
            raise RuntimeError(f"prompt-count mismatch in {checkpoint_path}")
        loaded = state.get("sums", {})
        if set(loaded) != set(sums):
            raise RuntimeError(f"module-set mismatch in {checkpoint_path}")
        for key in sums:
            if tuple(loaded[key].shape) != tuple(sums[key].shape):
                raise RuntimeError(f"shape mismatch for {key} in {checkpoint_path}")
            sums[key].copy_(loaded[key].float())
        n_done = int(state["n_done"])
        next_prompt = int(state["next_prompt"])
        if not 0 <= next_prompt <= len(prompts) or not 0 <= n_done <= next_prompt:
            raise RuntimeError(f"invalid counters in {checkpoint_path}")
        print(
            f"  resume module-J prompt={next_prompt}/{len(prompts)} n_done={n_done}",
            flush=True,
        )

    for prompt_index in range(next_prompt, len(prompts)):
        prompt = prompts[prompt_index]
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        ).input_ids.to(device)
        seq_len = int(encoded.shape[1])
        valid = torch.arange(skip_first, seq_len - 1, device=device)
        if valid.numel() == 0:
            print(
                f"  skip prompt {prompt_index + 1}: seq_len={seq_len} <= {skip_first + 1}",
                flush=True,
            )
            next_prompt = prompt_index + 1
            continue

        captured: dict[str, torch.Tensor] = {}
        handles = []

        def embedding_hook(_module, _inputs, output):
            tensor = _tensor(output)
            tensor.requires_grad_(True)

        def target_hook(_module, _inputs, output):
            captured["__target__"] = _tensor(output)

        handles.append(embedding.register_forward_hook(embedding_hook))
        handles.append((target_module or layers[-1]).register_forward_hook(target_hook))
        for target in targets:
            key = target["key"]

            def module_hook(_module, _inputs, output, *, capture_key=key):
                captured[capture_key] = _tensor(output)

            handles.append(target["module"].register_forward_hook(module_hook))

        try:
            with torch.enable_grad():
                decoder(input_ids=encoded.expand(dim_batch, -1), use_cache=False)
                target_activation = captured["__target__"]
                missing = [target["key"] for target in targets if target["key"] not in captured]
                if missing:
                    raise RuntimeError(f"target modules were not called: {missing}")
                module_outputs = [captured[target["key"]] for target in targets]
                cotangent = torch.zeros_like(target_activation)
                batch_indices = torch.arange(dim_batch, device=target_activation.device)
                n_passes = math.ceil(d_model / dim_batch)

                for pass_index, dim_start in enumerate(range(0, d_model, dim_batch)):
                    n_this = min(dim_batch, d_model - dim_start)
                    cotangent.zero_()
                    cotangent[
                        batch_indices[:n_this, None],
                        valid[None, :],
                        dim_start + batch_indices[:n_this, None],
                    ] = 1.0
                    grads = torch.autograd.grad(
                        outputs=target_activation,
                        inputs=module_outputs,
                        grad_outputs=cotangent,
                        retain_graph=pass_index < n_passes - 1,
                        allow_unused=True,
                    )
                    if len(targets) != len(grads):
                        raise RuntimeError(
                            "autograd returned a different number of gradients: "
                            f"targets={len(targets)} grads={len(grads)}"
                        )
                    # Hope's pinned runtime is Python 3.9.  Keep the explicit
                    # length check above instead of Python 3.10's zip(strict=True).
                    for target, grad in zip(targets, grads):
                        if grad is None:
                            raise RuntimeError(f"unused module output: {target['key']}")
                        if grad.ndim != 3:
                            raise RuntimeError(
                                f"unexpected grad shape for {target['key']}: {tuple(grad.shape)}"
                            )
                        rows = grad[:n_this, valid, :].float().mean(dim=1).cpu()
                        sums[target["key"]][dim_start : dim_start + n_this] += rows
                    del grads
        finally:
            for handle in handles:
                handle.remove()
        n_done += 1
        next_prompt = prompt_index + 1
        print(
            f"  L{layer_index} prompt {prompt_index + 1}/{len(prompts)} "
            f"seq={seq_len} valid={int(valid.numel())}",
            flush=True,
        )
        del captured
        torch.cuda.empty_cache()

        if checkpoint_path and (
            next_prompt % max(1, checkpoint_every) == 0
            or next_prompt == len(prompts)
        ):
            atomic_torch_save(
                {
                    "version": "v1_module_j_sum",
                    "prompt_count": len(prompts),
                    "next_prompt": next_prompt,
                    "n_done": n_done,
                    "sums": sums,
                },
                checkpoint_path,
            )
            print(f"  checkpoint module-J at prompt {next_prompt}", flush=True)

    if n_done == 0:
        raise RuntimeError("no prompt produced a valid module Jacobian")
    return {key: value / n_done for key, value in sums.items()}, n_done


def explicit_layout_spec(hf):
    """Return non-standard jlens Layout kwargs for known decoder structures."""
    decoder = getattr(hf, "model", None)
    if (
        decoder is not None
        and all(hasattr(decoder, name) for name in ("layers", "norm", "tok_embeddings"))
        and hasattr(hf, "output")
    ):
        return {"path": "model", "embed": "tok_embeddings", "lm_head": "output"}

    decoder = getattr(hf, "transformer", None)
    if (
        decoder is not None
        and all(hasattr(decoder, name) for name in ("h", "ln_f", "word_embeddings"))
        and hasattr(hf, "lm_head")
    ):
        return {
            "path": "transformer",
            "layers": "h",
            "norm": "ln_f",
            "embed": "word_embeddings",
        }
    return None


def resolve_model_parts(hf, tok):
    """Reuse jlens' audited HF layout resolver and expose the needed modules."""
    import jlens

    spec = explicit_layout_spec(hf)
    layout = None
    if spec is not None:
        from jlens.hf import Layout

        layout = Layout(**spec)

    wrapped = jlens.from_hf(hf, tok, layout=layout)
    return (
        wrapped._text_module,
        wrapped._embed_tokens,
        wrapped.layers,
        wrapped._final_norm,
        wrapped._lm_head,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--safe", required=True)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--layer_indices", default="", help="comma-separated explicit layer indices; overrides --n_layers")
    ap.add_argument("--fit_n", type=int, default=4)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim_batch", type=int, default=8)
    ap.add_argument("--skip_first", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument(
        "--fit_corpus_jsonl",
        required=True,
        help="reference JSONL text; deterministically packed to full sequences",
    )
    ap.add_argument("--checkpoint_every", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help="model/activation dtype; Jacobian accumulators and metric algebra remain fp32",
    )
    ap.add_argument("--jlens_path", default="jacobian-lens")
    ap.add_argument("--cache_dir", default="outputs/cache/exact")
    ap.add_argument("--out", default="outputs/exact.jsonl")
    args = ap.parse_args()
    if args.fit_n <= 0 or args.dim_batch <= 0 or args.k < 2 or args.iters <= 0:
        ap.error("fit_n, dim_batch and iters must be positive; k must be >= 2")
    if args.skip_first < 0 or args.max_seq_len <= args.skip_first + 1:
        ap.error("require 0 <= skip_first < max_seq_len - 1")

    sys.path.insert(0, args.jlens_path)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import transformers

    dev = args.device
    model_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    ).to(dev).eval()
    tok = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
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
        selected_layers = sorted(
            set(int(x) for x in np.linspace(0, n_layers_total - 2, args.n_layers))
        )

    gamma = (
        final_norm.weight.detach().to(dev).float()
        if hasattr(final_norm, "weight") and final_norm.weight is not None
        else torch.ones(d_model, device=dev)
    )
    wu = unembed.weight.detach().to(dev).float()
    h_metric = (gamma.unsqueeze(1) * (wu.t() @ wu) * gamma.unsqueeze(0)).double()
    h_metric = (h_metric + h_metric.t()) * 0.5
    h_eigenvalues, h_eigenvectors = torch.linalg.eigh(h_metric)
    h_eigenvalues = h_eigenvalues.clamp_min(0).float()
    h_eigenvectors = h_eigenvectors.float()
    del h_metric, wu
    torch.cuda.empty_cache()

    corpus_sha256 = (
        _sha256_file(args.fit_corpus_jsonl) if args.fit_corpus_jsonl else None
    )
    manifest = {
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "safe": args.safe,
        "model": args.model,
        "selected_layers": selected_layers,
        "fit_n": args.fit_n,
        "model_dtype": args.dtype,
        "dim_batch": args.dim_batch,
        "skip_first": args.skip_first,
        "max_seq_len": args.max_seq_len,
        "fit_corpus_kind": "fixed_length_jsonl" if args.fit_corpus_jsonl else "mixed",
        "fit_corpus_sha256": corpus_sha256,
        "module_set": "q/k/v/o + ffn_down",
        "jacobian_target": "final_block_output",
        "jacobian_representation": "dense_unsketched",
    }
    cache_path = os.path.join(args.cache_dir, args.safe, _hash_manifest(manifest))
    manifest_path = os.path.join(cache_path, "manifest.json")
    os.makedirs(cache_path, exist_ok=True)
    atomic_json_save(manifest, manifest_path)
    atomic_torch_save(
        {"eigenvalues": h_eigenvalues.cpu(), "eigenvectors": h_eigenvectors.cpu()},
        os.path.join(cache_path, "output_metric_eig.pt"),
    )

    prompts = build_fixed_length_jsonl_corpus(
        args.fit_corpus_jsonl, args.fit_n, tok, args.max_seq_len
    )
    matrix_records = []
    print(
        f"[{args.safe}] layers={selected_layers} fit_n={args.fit_n} "
        f"module_set=qkvo+down",
        flush=True,
    )

    for layer_index in selected_layers:
        targets = discover_target_modules(layers[layer_index])
        validate_module_set(targets)
        print(
            f"L{layer_index} targets="
            + ", ".join(f"{x['category']}:{x['path']}" for x in targets),
            flush=True,
        )
        jacobian_paths = {
            target["key"]: os.path.join(
                cache_path,
                f"J_L{layer_index}_{_safe_filename(target['key'])}.pt",
            )
            for target in targets
        }
        checkpoint_path = os.path.join(cache_path, f"J_L{layer_index}_running.pt")
        if all(os.path.isfile(path) for path in jacobian_paths.values()):
            jacobians = {
                key: torch_load_cpu(path).float()
                for key, path in jacobian_paths.items()
            }
            n_prompts_used = args.fit_n
            print(f"  module-J cache hit ({len(jacobians)})", flush=True)
        else:
            jacobians, n_prompts_used = fit_module_jacobians(
                decoder=decoder,
                embedding=embedding,
                layers=layers,
                layer_index=layer_index,
                targets=targets,
                tokenizer=tok,
                prompts=prompts,
                d_model=d_model,
                device=dev,
                dim_batch=args.dim_batch,
                max_seq_len=args.max_seq_len,
                skip_first=args.skip_first,
                checkpoint_path=checkpoint_path,
                checkpoint_every=args.checkpoint_every,
            )
            for key, path in jacobian_paths.items():
                atomic_torch_save(jacobians[key].float(), path)
            # The running sum is only a recovery artifact.  Once every final
            # per-module J has been committed atomically it would duplicate a
            # full layer of cache, so remove it to keep paper-scale storage
            # bounded.  A later rerun uses the final J files directly.
            if os.path.isfile(checkpoint_path):
                os.unlink(checkpoint_path)

        for target in targets:
            key = target["key"]
            weight = target["module"].weight.detach().to(dev).float()
            raw_columns = weight.t().contiguous()
            lens_columns = output_metric_coordinates(
                weight,
                jacobians[key].to(dev),
                h_eigenvectors,
                h_eigenvalues,
            )
            raw_labels, raw_k = cluster_labels(
                raw_columns, args.k, seed=args.seed, iters=args.iters
            )
            j_labels, j_k = cluster_labels(
                lens_columns, args.k, seed=args.seed, iters=args.iters
            )
            km_raw = relative_sse_for_labels(raw_columns, raw_labels, raw_k)
            km_jassign_raw = relative_sse_for_labels(raw_columns, j_labels, j_k)
            km_jassign_lens = relative_sse_for_labels(lens_columns, j_labels, j_k)
            record = {
                "layer": layer_index,
                "module": target["path"],
                "category": target["category"],
                "shape": list(weight.shape),
                "n_prompts_used": n_prompts_used,
                "km_raw": km_raw,
                "km_jassign_raw": km_jassign_raw,
                "km_jassign_lens": km_jassign_lens,
                "absdiff_jassign_raw": abs(km_jassign_raw - km_raw),
            }
            matrix_records.append(record)
            print(
                f"  {target['category']:<3} {target['path']:<32} "
                f"raw={km_raw:.8f} J->R={km_jassign_raw:.8f} "
                f"Jlens={km_jassign_lens:.8f}",
                flush=True,
            )
            del weight, raw_columns, lens_columns, raw_labels, j_labels
        del jacobians
        torch.cuda.empty_cache()

    aggregate = aggregate_paths(matrix_records)
    result = {
        "schema": "dv_lens_dense_v1",
        "safe": args.safe,
        "method": METHOD,
        "implementation_version": IMPLEMENTATION_VERSION,
        "nL": n_layers_total,
        "d": d_model,
        "layers": selected_layers,
        "fit_n": args.fit_n,
        "model_dtype": args.dtype,
        "jacobian_representation": "dense_unsketched",
        "fit_corpus_kind": manifest["fit_corpus_kind"],
        "fit_corpus_sha256": corpus_sha256,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "k": args.k,
        "iters": args.iters,
        "aggregation": "arithmetic_mean_of_per_matrix_relative_sse",
        "path_a": "module_JWL_assignment_on_qkvo_and_ffn_down_scored_raw",
        "path_b": "raw_kmeans_qkvo_plus_JWL_assignment_ffn_down_scored_raw",
        "cache_path": cache_path,
        "matrix_records": matrix_records,
        **aggregate,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print("\n" + json.dumps(aggregate, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
