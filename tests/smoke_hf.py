"""Offline end-to-end CLI check with a freshly initialized tiny Llama."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, timeout=180)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        model_path = tmp / "model"
        vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, **{f"word{i}": i + 3 for i in range(29)}}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                            bos_token="[BOS]", eos_token="[EOS]")
        tokenizer.save_pretrained(model_path)
        torch.manual_seed(3)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                                            num_hidden_layers=2, num_attention_heads=2,
                                            num_key_value_heads=2, max_position_embeddings=64))
        model.save_pretrained(model_path)
        corpus = tmp / "reference.jsonl"
        corpus.write_text("\n".join(json.dumps({"text": " ".join(f"word{i % 29}" for i in range(40))})
                                   for _ in range(4)), encoding="utf-8")
        common = ["--model", model_path, "--safe", "tiny", "--fit_corpus_jsonl", corpus,
                  "--fit_n", "2", "--layer_indices", "0", "--max_seq_len", "16",
                  "--skip_first", "1", "--dim_batch", "2", "--dtype", "fp32",
                  "--device", "cpu", "--k", "2", "--iters", "3"]
        for method in ("exact", "sketch"):
            extra = ["--sketch_dim", "4", "--layer_group_size", "1"] if method == "sketch" else []
            run(f"code/{method}.py", *common, *extra, "--cache_dir", tmp / method,
                "--out", tmp / f"{method}.jsonl")
            result = json.loads((tmp / f"{method}.jsonl").read_text(encoding="utf-8"))
            assert result["n_matrices"] == 5
            assert 0 <= result["km_path_a_jwl_qkvo_ffn"] <= 1.000001
            assert {row["category"] for row in result["matrix_records"]} == {"q", "k", "v", "o", "ffn"}
        cache = next((tmp / "exact" / "tiny").glob("*/manifest.json")).parent
        run("examples/read_columns.py", "--model", model_path, "--cache", cache, "--layer", "0",
            "--module", "self_attn.q_proj", "--columns", "0,1", "--topk", "3", "--out", tmp / "readout.json")
        readout = json.loads((tmp / "readout.json").read_text(encoding="utf-8"))
        assert len(readout["columns"]) == 2
        assert len(readout["columns"][0]["positive_tail"]) == 3
        print("PASS: offline tiny-Llama dense fit, sketch fit, and signed readout")


if __name__ == "__main__":
    main()
