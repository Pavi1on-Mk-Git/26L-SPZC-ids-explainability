import numpy as np
from tqdm.auto import tqdm


def _distance(X: np.ndarray, point: np.ndarray) -> np.ndarray:
    diff = X - point
    return (diff**2).sum(axis=1)


def _centroid(X: np.ndarray, members: np.ndarray) -> np.ndarray:
    return X[members].mean(axis=0)


def _argmax_distance(X: np.ndarray, members: np.ndarray, point: np.ndarray) -> int:
    return int(members[_distance(X[members], point).argmax()])


def _cluster(X: np.ndarray, members: np.ndarray, x: int, k: int) -> np.ndarray:
    dist = _distance(X[members], X[x])
    nearest = np.argpartition(dist, k - 1)[:k]
    return members[nearest]


def mdav(X: np.ndarray, k: int, progress: bool = False) -> list[np.ndarray]:
    n = len(X)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"Dataset size {n} is smaller than k={k}")

    X_remaining = np.arange(n)
    C: list[np.ndarray] = []

    bar = tqdm(total=n, desc=f"MDAV clustering (k={k})", unit="rec", disable=not progress)
    while len(X_remaining) >= 3 * k:
        x_c = _centroid(X, X_remaining)
        x_r = _argmax_distance(X, X_remaining, x_c)
        x_s = _argmax_distance(X, X_remaining, X[x_r])

        C_r = _cluster(X, X_remaining, x_r, k)
        X_remaining = np.setdiff1d(X_remaining, C_r, assume_unique=True)
        C_s = _cluster(X, X_remaining, x_s, k)
        X_remaining = np.setdiff1d(X_remaining, C_s, assume_unique=True)

        C.append(C_r)
        C.append(C_s)
        bar.update(len(C_r) + len(C_s))
        bar.set_postfix(clusters=len(C))

    if 2 * k <= len(X_remaining) < 3 * k:
        x_c = _centroid(X, X_remaining)
        x_r = _argmax_distance(X, X_remaining, x_c)
        C_r = _cluster(X, X_remaining, x_r, k)
        X_remaining = np.setdiff1d(X_remaining, C_r, assume_unique=True)
        C.append(C_r)
        bar.update(len(C_r))

    C.append(X_remaining)
    bar.update(len(X_remaining))

    bar.set_postfix(clusters=len(C))
    bar.close()
    return C
