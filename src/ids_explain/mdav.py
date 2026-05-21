import numpy as np


def _sq_distances(X: np.ndarray, point: np.ndarray) -> np.ndarray:
    diff = X - point
    return np.einsum("ij,ij->i", diff, diff)


def _knn_indices(distances: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(distances, k - 1)[:k]


def mdav(X: np.ndarray, k: int) -> list[np.ndarray]:
    n = len(X)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"Dataset size {n} is smaller than k={k}")

    remaining: np.ndarray = np.arange(n)
    clusters: list[np.ndarray] = []

    while len(remaining) >= 3 * k:
        subset = X[remaining]

        q = subset.mean(axis=0)
        r_local: int = int(_sq_distances(subset, q).argmax())

        dist_xr = _sq_distances(subset, subset[r_local])
        s_local: int = int(dist_xr.argmax())

        gr_local = _knn_indices(dist_xr, k)
        gr_global = remaining[gr_local]
        clusters.append(gr_global)

        after_gr = np.setdiff1d(remaining, gr_global, assume_unique=True)
        subset_s = X[after_gr]
        dist_xs = _sq_distances(subset_s, X[remaining[s_local]])
        gs_local = _knn_indices(dist_xs, k)
        gs_global = after_gr[gs_local]
        clusters.append(gs_global)

        remaining = np.setdiff1d(after_gr, gs_global, assume_unique=True)

    if len(remaining) > 0:
        if len(clusters) >= 2:
            half = len(remaining) // 2
            clusters[-2] = np.concatenate([clusters[-2], remaining[:half]])
            clusters[-1] = np.concatenate([clusters[-1], remaining[half:]])
        elif len(clusters) == 1:
            clusters[-1] = np.concatenate([clusters[-1], remaining])
        else:
            clusters.append(remaining)

    return clusters
