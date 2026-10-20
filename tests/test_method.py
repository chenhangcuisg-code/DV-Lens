"""Numerical contracts for the released method; no checkpoint download."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import torch
from exact import output_metric_coordinates, fit_module_jacobians, discover_target_modules, validate_module_set
from geometry import relative_sse_for_labels, vocabulary_readout, cluster_labels
from sketch import fit_group, target_key


class MethodTests(unittest.TestCase):
    def test_metric_matches_explicit_vocabulary_distances(self):
        torch.manual_seed(4)
        weight, jacobian = torch.randn(4, 16), torch.randn(6, 4)
        unembedding, gamma = torch.randn(40, 6), torch.randn(6)
        metric = (unembedding * gamma).t() @ (unembedding * gamma)
        eigenvalues, eigenvectors = torch.linalg.eigh(metric)
        coordinates = output_metric_coordinates(weight, jacobian, eigenvectors, eigenvalues)
        readout = vocabulary_readout(weight, jacobian, unembedding, gamma, list(range(16)))
        torch.testing.assert_close(torch.cdist(coordinates, coordinates),
                                   torch.cdist(readout, readout) / eigenvalues.max().sqrt(),
                                   rtol=2e-4, atol=2e-4)

    def test_raw_score_uses_fixed_partition(self):
        columns = torch.tensor([[1., 0.], [3., 0.], [0., 2.], [0., 4.]])
        labels = torch.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(relative_sse_for_labels(columns, labels, 2), 4 / 30)
        self.assertEqual(relative_sse_for_labels(columns, torch.arange(4), 4), 0.)
        a, k = cluster_labels(columns, 2, seed=7)
        b, _ = cluster_labels(columns, 2, seed=7)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(k, 2)

    def test_dense_and_joint_vjp_agree(self):
        # A causal, token-local linear decoder has an analytic downstream map.
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4, bias=False)
                self.down_proj = torch.nn.Linear(4, 4, bias=False)
            def forward(self, x):
                return x + self.down_proj(self.q_proj(x))
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(10, 4)
                self.layers = torch.nn.ModuleList([Block(), Block()])
            def forward(self, input_ids, use_cache=False):
                x = self.embed(input_ids)
                for block in self.layers:
                    x = block(x)
                return x
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return type("Encoding", (), {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})()
        torch.manual_seed(12)
        decoder = Decoder().eval()
        decoder.requires_grad_(False)
        targets = discover_target_modules(decoder.layers[0])
        validate_module_set(targets)
        with tempfile.TemporaryDirectory() as temporary:
            dense, n = fit_module_jacobians(
                decoder=decoder, embedding=decoder.embed, layers=decoder.layers,
                layer_index=0, targets=targets, tokenizer=Tokenizer(), prompts=["a", "b"],
                d_model=4, device="cpu", dim_batch=2, max_seq_len=5, skip_first=1)
            rows = torch.randn(3, 4) / math.sqrt(3)
            tagged = [{**target, "layer": 0} for target in targets]
            sketches, ns = fit_group(
                decoder=decoder, embedding=decoder.embed, layers=decoder.layers,
                targets=tagged, tokenized=[torch.tensor([1, 2, 3, 4, 5])] * 2,
                cotangent_rows=rows, device="cpu", dim_batch=2, skip_first=1,
                checkpoint_path=str(Path(temporary) / "running.pt"), checkpoint_every=1)
            final = decoder.layers[1]
            downstream = torch.eye(4) + final.down_proj.weight @ final.q_proj.weight
            for target in tagged:
                expected = downstream
                if target["category"] == "q":
                    expected = downstream @ decoder.layers[0].down_proj.weight
                torch.testing.assert_close(dense[target["key"]], expected)
                torch.testing.assert_close(sketches[target_key(target)], rows @ expected)
            self.assertEqual((n, ns), (2, 2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
