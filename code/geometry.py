"""Clustering, raw-weight reconstruction scores, and signed vocabulary readout."""
import json
import os
import torch

def atomic_torch_save(value, path):
    """Durably publish one reusable tensor artifact without partial files."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_json_save(value, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def cluster_labels(cols, k, seed=0, iters=12):
    """Run seeded k-means and return (labels, effective_k)."""
    from kmeans import kmeans_channels

    kk = min(int(k), int(cols.shape[0]) // 2)
    if kk < 2:
        return None, None
    return kmeans_channels(cols, kk, iters=iters, seed=seed), kk


@torch.no_grad()
def relative_sse_for_labels(cols, labels, k=None):
    """Relative centroid reconstruction SSE for an already-fixed partition."""
    if labels is None or labels.numel() != cols.shape[0]:
        return None
    kk = int(k) if k is not None else int(labels.max().item()) + 1
    if kk < 1:
        return None
    sums = torch.zeros(kk, cols.shape[1], device=cols.device, dtype=cols.dtype)
    counts = torch.zeros(kk, device=cols.device, dtype=cols.dtype)
    sums.index_add_(0, labels, cols)
    counts.index_add_(
        0,
        labels,
        torch.ones(labels.numel(), device=cols.device, dtype=cols.dtype),
    )
    centers = sums / counts.clamp_min(1).unsqueeze(1)
    numerator = (cols - centers[labels]).double().square().sum()
    denominator = cols.double().square().sum().clamp_min(1e-300)
    return float((numerator / denominator).item())


@torch.no_grad()
def vocabulary_readout(weight, jacobian, unembedding, gamma, column_ids):
    """Read selected native columns in the frozen output metric.

    Shapes: weight [module_out, module_in], jacobian [hidden, module_out],
    unembedding [vocab, hidden], gamma [hidden]. Returns [columns, vocab].
    The Jacobian target is the final block output; gamma is the final norm's
    learned scale. This is a local linear readout, not a probability or the
    exact derivative through the input-dependent final normalization.
    """
    columns = weight[:, column_ids].float()
    transported = jacobian.float() @ columns
    return (unembedding.float() @ (gamma.float()[:, None] * transported)).t()
