"""Deterministic 3x3 Rubik's Cube engine.

State = 54-character facelet string in the standard URFDLB order
(U 0-8, R 9-17, F 18-26, D 27-35, L 36-44, B 45-53). Each character is the
letter of the face whose centre the sticker belongs to, so SOLVED is
"UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB".

Move permutations are *derived from 3D geometry* (rotating sticker
positions), not hand typed, so every model uses exactly the same engine.
Move metric: half-turn metric (HTM) - 18 moves: U U' U2 R R' R2 ...
"""
from __future__ import annotations

import random
from operator import itemgetter

FACES = "URFDLB"
SOLVED = "".join(f * 9 for f in FACES)
MOVES = [f + s for f in FACES for s in ("", "'", "2")]

_AXIS = {
    "U": (0, 1, 0), "D": (0, -1, 0),
    "R": (1, 0, 0), "L": (-1, 0, 0),
    "F": (0, 0, 1), "B": (0, 0, -1),
}


def _sticker(face: str, i: int, j: int):
    """(cubie position, outward normal) of sticker (row i, col j) on a face."""
    if face == "U":
        return (j - 1, 1, i - 1), (0, 1, 0)
    if face == "R":
        return (1, 1 - i, 1 - j), (1, 0, 0)
    if face == "F":
        return (j - 1, 1 - i, 1), (0, 0, 1)
    if face == "D":
        return (j - 1, -1, 1 - i), (0, -1, 0)
    if face == "L":
        return (-1, 1 - i, j - 1), (-1, 0, 0)
    if face == "B":
        return (1 - j, 1 - i, -1), (0, 0, -1)
    raise ValueError(face)


_STICKERS = [_sticker(f, i, j) for f in FACES for i in range(3) for j in range(3)]
_INDEX = {s: k for k, s in enumerate(_STICKERS)}


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _rot_cw(axis, v):
    """Rotate v by -90 deg about `axis` (= clockwise seen from outside)."""
    c = _cross(axis, v)
    d = _dot(axis, v)
    return (-c[0] + axis[0] * d, -c[1] + axis[1] * d, -c[2] + axis[2] * d)


def _quarter_perm(face: str):
    a = _AXIS[face]
    perm = list(range(54))  # sticker at i moves to perm[i]
    for k, (pos, normal) in enumerate(_STICKERS):
        if _dot(pos, a) == 1:
            perm[k] = _INDEX[(_rot_cw(a, pos), _rot_cw(a, normal))]
    return perm


def _gather(perm):
    inv = [0] * 54
    for i, p in enumerate(perm):
        inv[p] = i
    return itemgetter(*inv)  # new[j] = old[inv[j]]


_MOVE_FN = {}
for _f in FACES:
    _p1 = _quarter_perm(_f)
    _p2 = [_p1[_p1[i]] for i in range(54)]
    _p3 = [_p1[_p2[i]] for i in range(54)]
    _MOVE_FN[_f] = _gather(_p1)
    _MOVE_FN[_f + "2"] = _gather(_p2)
    _MOVE_FN[_f + "'"] = _gather(_p3)


def apply_move(state: str, move: str) -> str:
    if move not in _MOVE_FN:
        raise ValueError(f"illegal move {move!r}")
    return "".join(_MOVE_FN[move](state))


def apply_moves(state: str, moves) -> str:
    for m in moves:
        state = apply_move(state, m)
    return state


def is_solved(state: str) -> bool:
    """All six faces uniform. Centres never move, so this means solved."""
    return all(len(set(state[i * 9:(i + 1) * 9])) == 1 for i in range(6))


def is_valid_state(state: str) -> bool:
    return len(state) == 54 and all(state.count(f) == 9 for f in FACES)


def generate_scramble(n: int, rng: random.Random) -> list[str]:
    """n random moves, never turning the same face twice in a row and never
    three turns on one axis in a row (keeps the scramble non-redundant)."""
    axis_of = {"U": 0, "D": 0, "R": 1, "L": 1, "F": 2, "B": 2}
    seq: list[str] = []
    while len(seq) < n:
        m = rng.choice(MOVES)
        f = m[0]
        if seq and seq[-1][0] == f:
            continue
        if len(seq) >= 2 and axis_of[seq[-1][0]] == axis_of[seq[-2][0]] == axis_of[f]:
            continue
        seq.append(m)
    return seq


def inverse_move(move: str) -> str:
    return move if move.endswith("2") else (move[0] if move.endswith("'") else move + "'")


_NAMES = dict(zip(FACES, ["upper", "right", "front", "down", "left", "back"]))
MOVE_DESCRIPTIONS = {}
for _f in FACES:
    MOVE_DESCRIPTIONS[_f] = f"Turn the {_NAMES[_f]} face clockwise by 90 degrees"
    MOVE_DESCRIPTIONS[_f + "'"] = f"Turn the {_NAMES[_f]} face counterclockwise by 90 degrees"
    MOVE_DESCRIPTIONS[_f + "2"] = f"Turn the {_NAMES[_f]} face by 180 degrees"

LAYOUT_NOTE = (
    "54 chars in order U(0-8) R(9-17) F(18-26) D(27-35) L(36-44) B(45-53); each face is read "
    "row by row. Each letter is the face whose centre that sticker belongs to. Solved = every "
    "face uniform: UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB. Clockwise is as seen "
    "looking at that face from outside the cube."
)


# ---------------------------------------------------------------------------
# Move definitions given to EVERY model so the cube can actually be reasoned about.
# ---------------------------------------------------------------------------
def _cycles(perm):
    seen, out = set(), []
    for i in range(54):
        if i in seen or perm[i] == i:
            continue
        c, j = [], i
        while j not in seen:
            seen.add(j)
            c.append(j)
            j = perm[j]
        out.append(c)
    return out


# For a clockwise quarter turn of face X: for each cycle [a,b,c,d] the sticker at index a moves
# to index b, b to c, c to d, d to a. X' is the reverse direction; X2 applies X twice.
MOVE_CYCLES = {f: _cycles(_quarter_perm(f)) for f in FACES}

INDEX_MAP = (
    "Each face has 9 stickers read row by row (top-left=0, top-middle=1, top-right=2, middle-left=3, "
    "centre=4, ..., bottom-right=8). Face offsets: U=0, R=9, F=18, D=27, L=36, B=45. "
    "Sticker index = face offset + position. Example: F's top-right sticker is index 20."
)
EXAMPLE_MOVE = "R"
EXAMPLE = {"move": EXAMPLE_MOVE, "before": SOLVED, "after": apply_move(SOLVED, EXAMPLE_MOVE)}
