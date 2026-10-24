import random

from backend.cube import (MOVES, SOLVED, apply_move, apply_moves, generate_scramble,
                          inverse_move, is_solved, is_valid_state)
from backend.solver import optimal_distance


def test_eighteen_moves():
    assert len(MOVES) == 18


def test_quarter_turn_has_order_four_and_inverse_cancels():
    for f in "URFDLB":
        assert apply_moves(SOLVED, [f] * 4) == SOLVED
        assert apply_moves(SOLVED, [f, f + "'"]) == SOLVED
        assert apply_move(SOLVED, f + "2") == apply_moves(SOLVED, [f, f])
        assert not is_solved(apply_move(SOLVED, f))


def test_u_move_geometry_matches_real_cube():
    s = apply_move(SOLVED, "U")  # U clockwise: R's top row goes to F
    assert s[18:21] == "RRR"
    assert s[9:12] == "BBB"  # B's top row goes to R


def test_r_move_geometry_matches_real_cube():
    s = apply_move(SOLVED, "R")  # R clockwise: F's right column goes up to U
    assert [s[2], s[5], s[8]] == ["F", "F", "F"]


def test_sexy_move_has_order_six():
    s = SOLVED
    for _ in range(6):
        s = apply_moves(s, ["R", "U", "R'", "U'"])
    assert s == SOLVED
    s = SOLVED
    for _ in range(5):
        s = apply_moves(s, ["R", "U", "R'", "U'"])
    assert s != SOLVED


def test_states_stay_valid():
    rng = random.Random(1)
    s = SOLVED
    for m in generate_scramble(40, rng):
        s = apply_move(s, m)
        assert is_valid_state(s)


def test_inverse_scramble_solves():
    rng = random.Random(7)
    scr = generate_scramble(18, rng)
    s = apply_moves(SOLVED, scr)
    assert not is_solved(s)
    assert is_solved(apply_moves(s, [inverse_move(m) for m in reversed(scr)]))


def test_optimal_distance_known_values():
    assert optimal_distance(SOLVED) == 0
    assert optimal_distance(apply_move(SOLVED, "R2")) == 1
    assert optimal_distance(apply_moves(SOLVED, ["R", "U"])) == 2
    assert optimal_distance(apply_moves(SOLVED, ["R", "U", "R'", "U'"])) == 4  # sexy move


def test_optimal_never_exceeds_scramble_length():
    rng = random.Random(3)
    for n in (3, 4, 5):
        for _ in range(10):
            scr = generate_scramble(n, rng)
            d = optimal_distance(apply_moves(SOLVED, scr))
            assert d is not None and 1 <= d <= n


def test_optimal_unknown_beyond_six_is_none_or_exact():
    rng = random.Random(5)
    s = apply_moves(SOLVED, generate_scramble(18, rng))
    d = optimal_distance(s)
    assert d is None or d <= 6
