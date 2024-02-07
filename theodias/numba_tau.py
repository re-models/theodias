"""Auxiliary methods to speed up the handling of numpy arrays with numba."""

import numpy as np
from numba import jit, types, typed
from typing import List, Tuple


@jit(nopython=True)
def create_hash(pos: np.ndarray):
    s = 0
    for i in range(len(pos)):
        s += (pos[i]+1) * 5 ** i       # +1 to include suspended elements
    return s

@jit(nopython=True)
def gray(N:types.int8, K:types.int8):

    n = np.empty(K+1, dtype=types.int8)
    n.fill(N)

    g = np.zeros(K+1, dtype=types.int8)
    u = np.ones(K+1, dtype=types.int8)

    while g[K]==0:

        yield g[:-1]

        i = 0
        k = g[0] + u[0]
        while k >= n[i] or k < 0:
            u[i] = -u[i]
            i += 1
            k = g[i] + u[i]

        g[i] = k


@jit(nopython=True)
def direct_sub_arrays(arr: np.ndarray, n: int) -> List[types.int8[:]]:
    _subpositions = typed.List()
    for i in typed.List(*np.where(arr != 0)):
        submask = np.ones(n, dtype=types.int8)
        submask[i] = 0
        _subpositions.append(submask*arr)
    return _subpositions


@jit(nopython=True)
def to_int(arr: np.ndarray):
    s = 0
    for i in range(len(arr)):
        s += arr[i]*3**i
    return s


@jit(nopython=True)
def from_int(pos: int, n: int) -> types.int8[:]:
    arr = np.empty(n, dtype=types.int8)
    q = pos
    for i in range(n):
        q, arr[i] = divmod(q, 3)
    return arr


@jit(nopython=True)
def na_union(arrays: Tuple[np.ndarray], n: int):

    res = np.zeros(n, dtype=np.float32)
    arrays = np.vstack(arrays).T

    for i in range(n):
        if (1.0 in arrays[i]) & (2.0 in arrays[i]):
            return None
        else:
            res[i] = np.max(arrays[i])

    return res
