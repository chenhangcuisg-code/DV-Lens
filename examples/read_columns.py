"""Export signed vocabulary tails for selected columns from a dense fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import torch
from exact import _safe_filename, discover_target_modules, resolve_model_parts
from geometry import atomic_json_save, torch_load_cpu, vocabulary_readout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", type=Path, required=True, help="dense cache directory containing manifest.json")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--module", required=True, help="path from matrix_records, e.g. self_attn.q_proj")
    ap.add_argument("--columns", default="0", help="comma-separated native input column indices")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/readout.json")
    args = ap.parse_args()
    if args.topk < 1:
        ap.error("topk must be positive")
    with (args.cache / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("jacobian_representation") != "dense_unsketched":
        ap.error("signed readout requires a dense cache, not a sketch cache")
    if manifest["model"] != args.model:
        ap.error("--model must match the dense fit's model identifier or path")
    if args.layer not in manifest["selected_layers"]:
        ap.error("requested layer is not in this cache")

    import transformers
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[manifest["model_dtype"]]
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(args.device).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _, _, layers, norm, unembed = resolve_model_parts(model, tokenizer)
    targets = [t for t in discover_target_modules(layers[args.layer]) if t["path"] == args.module]
    if len(targets) != 1:
        ap.error("module path must identify exactly one supported projection")
    target = targets[0]
    columns = [int(c.strip()) for c in args.columns.split(",")]
    weight = target["module"].weight.detach()
    if not columns or min(columns) < 0 or max(columns) >= weight.shape[1]:
        ap.error("column index is outside the native weight matrix")
    path = args.cache / f"J_L{args.layer}_{_safe_filename(target['key'])}.pt"
    jacobian = torch_load_cpu(str(path)).to(args.device)
    gamma = (norm.weight.detach().to(args.device) if getattr(norm, "weight", None) is not None
             else torch.ones(unembed.weight.shape[1], device=args.device))
    records = []
    # One selected column at a time bounds vocabulary readout memory.
    for column in columns:
        values = vocabulary_readout(weight, jacobian, unembed.weight.detach(), gamma, [column])[0]
        k = min(args.topk, values.numel())
        def tail(largest):
            scores, ids = torch.topk(values, k, largest=largest)
            return [{"token_id": int(i), "token": tokenizer.decode([int(i)]), "value": float(v)}
                    for i, v in zip(ids.cpu(), scores.cpu())]
        records.append({"column": column, "positive_tail": tail(True), "negative_tail": tail(False)})
    atomic_json_save({"model": args.model, "layer": args.layer, "module": args.module,
                      "readout": "W_U Gamma mean_J weight_column", "columns": records}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
