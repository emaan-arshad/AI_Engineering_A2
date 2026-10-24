"""Exact shortest-solution distance (half-turn metric) by meet-in-the-middle BFS.

Because the 18-move set is closed under inversion, dist(a->solved) equals
dist(solved->a). We BFS 3 layers from the solved cube once (~3.5k states) and
3 layers from the given state; the minimum d_a + d_b over common states is the
exact distance whenever the true distance is <= 6. That covers the warm-up
(3-4 turns) and primary (4-5 turns) suites. For longer scrambles we return
None (unknown) rather than a wrong number.
"""
from __future__ import annotations

from functools import lru_cache

from .cube import MOVES, SOLVED, apply_move

HALF_DEPTH = 3
MAX_EXACT = 2 * HALF_DEPTH


def _bfs(start: str, depth: int) -> dict[str, int]:
    dist = {start: 0}
    frontier = [start]
    for d in range(1, depth + 1):
        nxt = []
        for s in frontier:
            for m in MOVES:
                t = apply_move(s, m)
                if t not in dist:
                    dist[t] = d
                    nxt.append(t)
        frontier = nxt
    return dist


@lru_cache(maxsize=1)
def _solved_table() -> dict[str, int]:
    return _bfs(SOLVED, HALF_DEPTH)


def optimal_distance(state: str) -> int | None:
    """Exact shortest solution length in moves, or None if it is > 6."""
    table = _solved_table()
    best = None
    for s, da in _bfs(state, HALF_DEPTH).items():
        db = table.get(s)
        if db is not None and (best is None or da + db < best):
            best = da + db
    return best
