"""Seeded tensor k-means for rows of a weight-column coordinate matrix."""
import torch

@torch.no_grad()
def kmeans_channels(cols: torch.Tensor, k: int, iters: int = 15, seed: int = 0):
    """Vectorized K-Means over the ROWS of `cols` (each row = one channel of W).

    cols: [n_channels, dim]  (i.e. W.T)
    returns labels: LongTensor [n_channels] in [0, k)
    Fully vectorized: scatter_add centroid update + batched empty-cluster reseed.
    """
    n, d = cols.shape
    device = cols.device
    g = torch.Generator(device=device).manual_seed(seed)
    if k >= n:  # degenerate: every channel its own cluster
        return torch.arange(n, device=device)

    # --- init: random distinct points (fast; k-means++ not worth it here) ---
    perm = torch.randperm(n, generator=g, device=device)[:k]
    centers = cols[perm].clone()

    labels = torch.full((n,), -1, dtype=torch.long, device=device)
    for _ in range(iters):
        dists = torch.cdist(cols, centers)  # [n, k]
        new_labels = dists.argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        # vectorized centroid update via scatter_add
        sums = torch.zeros(k, d, device=device, dtype=cols.dtype)
        sums.index_add_(0, labels, cols)
        counts = torch.zeros(k, device=device, dtype=cols.dtype)
        counts.index_add_(0, labels, torch.ones(n, device=device, dtype=cols.dtype))
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        # reseed all empty clusters at once to the globally farthest points
        empties = (~nonempty).nonzero(as_tuple=True)[0]
        if empties.numel() > 0:
            far = torch.cdist(cols, centers).min(dim=1).values  # [n]
            pick = far.topk(empties.numel()).indices
            centers[empties] = cols[pick]
    return labels
