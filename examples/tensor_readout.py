"""Minimal CPU example using synthetic matrices, without model downloads."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import torch
from exact import output_metric_coordinates
from geometry import cluster_labels, relative_sse_for_labels, vocabulary_readout

torch.manual_seed(0)
weight = torch.randn(8, 64)
jacobian = torch.randn(12, 8)
unembedding = torch.randn(128, 12)
gamma = torch.ones(12)
weighted = unembedding * gamma[None, :]
values, vectors = torch.linalg.eigh(weighted.t() @ weighted)
coordinates = output_metric_coordinates(weight, jacobian, vectors, values)
labels, k = cluster_labels(coordinates, k=8, seed=0, iters=12)
score = relative_sse_for_labels(weight.t(), labels, k)
readout = vocabulary_readout(weight, jacobian, unembedding, gamma, [0, 1])
print(f"DV-Complexity (synthetic): {score:.6f}")
print(f"Signed readout shape: {tuple(readout.shape)} [columns, vocabulary]")
print(f"Column 0: largest vocabulary coordinates {readout[0].topk(5).indices.tolist()}")
