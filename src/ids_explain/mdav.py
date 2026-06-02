import numpy as np


def _distance(X: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Returns the squared Euclidean distance. MDAV only ever feeds this into
    argmin / argmax, and squaring preserves ordering, so dropping the square
    root does not change any result.
    """
    diff = X - point
    return (diff**2).sum(axis=1)


def _mean_record(X: np.ndarray, members: np.ndarray) -> np.ndarray:
    """The average vector (centroid) of the current set X."""
    return X[members].mean(axis=0)


def _argmax_distance(X: np.ndarray, members: np.ndarray, point: np.ndarray) -> int:
    """argmax_{x_i in X} distance(x_i, point), returned as a global index."""
    return int(members[_distance(X[members], point).argmax()])


def _cluster(X: np.ndarray, members: np.ndarray, x: int, k: int) -> np.ndarray:
    """Algorithm 3: cluster(x, k, X).

    Builds a cluster of exactly k records around seed x. The reference grows
    C = {x} by repeatedly moving the record of X nearest to x into C until
    |C| = k. Because the distance is always measured to the fixed seed x,
    this is exactly the seed plus its k - 1 nearest neighbours, so we can take
    all k at once. Returns the global indices of the cluster's records.
    """
    dist = _distance(X[members], X[x])
    nearest = np.argpartition(dist, k - 1)[:k]
    return members[nearest]


def mdav(X: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Partitions X into clusters, each of size between k and 2k - 1, and returns
    one array of global indices per cluster. Structure mirrors the reference
    pseudocode line by line.
    """
    n = len(X)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"Dataset size {n} is smaller than k={k}")

    X_remaining = np.arange(n)
    C: list[np.ndarray] = []

    while len(X_remaining) >= 3 * k:
        x_c = _mean_record(X, X_remaining)
        x_r = _argmax_distance(X, X_remaining, x_c)
        x_s = _argmax_distance(X, X_remaining, X[x_r])

        C_r = _cluster(X, X_remaining, x_r, k)
        X_remaining = np.setdiff1d(X_remaining, C_r, assume_unique=True)
        C_s = _cluster(X, X_remaining, x_s, k)
        X_remaining = np.setdiff1d(X_remaining, C_s, assume_unique=True)

        C.append(C_r)
        C.append(C_s)

    if 2 * k <= len(X_remaining) < 3 * k:
        x_c = _mean_record(X, X_remaining)
        x_r = _argmax_distance(X, X_remaining, x_c)
        C_r = _cluster(X, X_remaining, x_r, k)
        X_remaining = np.setdiff1d(X_remaining, C_r, assume_unique=True)
        C.append(C_r)
    else:
        C.append(X_remaining)

    return C
