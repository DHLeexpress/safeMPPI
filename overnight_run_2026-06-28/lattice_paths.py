"""Enumerate the SAFE MONOTONE lattice paths for the grid-coverage experiment (the coverage denominator).

Grid = N x N cells, indices (i=col, j=row), start = (0,0) bottom-left, goal = (N-1,N-1) top-right.
A safe path: a sequence of cells from start to goal where every step is +x (right) or +y (up) [monotone /
goal-seeking], stays in the grid, and never enters the obstacle block. coverage = |discovered| / |all|.
"""
from __future__ import annotations
from functools import lru_cache
from typing import List, Set, Tuple


def block_cells(N=7, bsize=3) -> Set[Tuple[int, int]]:
    lo = (N - bsize) // 2
    return {(i, j) for i in range(lo, lo + bsize) for j in range(lo, lo + bsize)}


def count_safe_paths(N=7, bsize=3) -> int:
    blk = block_cells(N, bsize)

    @lru_cache(maxsize=None)
    def f(i, j):
        if (i, j) in blk:
            return 0
        if (i, j) == (N - 1, N - 1):
            return 1
        tot = 0
        if i + 1 < N:
            tot += f(i + 1, j)
        if j + 1 < N:
            tot += f(i, j + 1)
        return tot
    return f(0, 0)


def enumerate_safe_paths(N=7, bsize=3) -> List[Tuple[Tuple[int, int], ...]]:
    """All safe monotone start->goal cell-paths (as tuples of cells)."""
    blk = block_cells(N, bsize)
    out: List[Tuple[Tuple[int, int], ...]] = []

    def rec(i, j, acc):
        if (i, j) in blk:
            return
        acc = acc + [(i, j)]
        if (i, j) == (N - 1, N - 1):
            out.append(tuple(acc)); return
        if i + 1 < N:
            rec(i + 1, j, acc)
        if j + 1 < N:
            rec(i, j + 1, acc)
    rec(0, 0, [])
    return out


def path_signature(cells: List[Tuple[int, int]], N=7, bsize=3):
    """Map a sequence of visited cells (from a rolled-out trajectory) to a canonical safe monotone path,
    or None if it is unsafe (non-monotone / out-of-grid / hits the block / doesn't reach the goal)."""
    blk = block_cells(N, bsize)
    # de-duplicate consecutive repeats
    seq = [cells[0]]
    for c in cells[1:]:
        if c != seq[-1]:
            seq.append(c)
    if seq[0] != (0, 0) or seq[-1] != (N - 1, N - 1):
        return None
    for k in range(len(seq)):
        i, j = seq[k]
        if not (0 <= i < N and 0 <= j < N) or (i, j) in blk:
            return None
        if k > 0:
            di, dj = i - seq[k - 1][0], j - seq[k - 1][1]
            if (di, dj) not in ((1, 0), (0, 1)):   # must be a single +x or +y step (monotone)
                return None
    return tuple(seq)


if __name__ == "__main__":
    for N, b in [(7, 3), (9, 3), (5, 3)]:
        print(f"N={N} block={b}x{b} (center): safe monotone paths = {count_safe_paths(N, b)} "
              f"(total monotone w/o block = {__import__('math').comb(2*(N-1), N-1)})")
    paths = enumerate_safe_paths(7, 3)
    print("7x7/3x3 enumerated:", len(paths), "example:", paths[0])
