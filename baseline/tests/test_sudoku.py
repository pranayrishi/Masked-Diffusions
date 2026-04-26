"""Tests for the Sudoku data path: 7-strategy solver, tokenization, synthetic helper."""

from __future__ import annotations

import numpy as np

from src.sudoku import (
    SUDOKU_MASK_TOKEN_ID,
    SUDOKU_SEQ_LEN,
    SUDOKU_VOCAB_SIZE,
    grid_to_tokens,
    solve_with_seven_strategies,
    sudoku_string_to_tokens,
    synthetic_easy_puzzle,
    tokens_to_grid,
    _CANONICAL_SOLVED,
    _is_valid_sudoku,
)


def test_constants():
    assert SUDOKU_SEQ_LEN == 81
    assert SUDOKU_VOCAB_SIZE == 10
    assert SUDOKU_MASK_TOKEN_ID == 0


def test_string_to_tokens_roundtrip():
    """Take the canonical solved board, stringify, parse back, compare."""
    canonical_str = "".join(str(_CANONICAL_SOLVED[r][c]) for r in range(9) for c in range(9))
    tokens = sudoku_string_to_tokens(canonical_str)
    grid = tokens_to_grid(tokens)
    assert grid == _CANONICAL_SOLVED
    # And tokens → grid → tokens is identity
    np.testing.assert_array_equal(grid_to_tokens(grid), tokens)


def test_seven_strategies_solves_a_minimally_obscured_puzzle():
    """Constructively solvable: take the canonical solved board and hide a handful
    of cells. Such a puzzle is trivially naked-singles solvable in one pass."""
    puzzle, solution = synthetic_easy_puzzle(num_clues=78, seed=0)   # hide only 3 cells
    n_empty = int((puzzle == 0).sum())
    assert n_empty == 3
    solved, order = solve_with_seven_strategies(puzzle.copy())
    assert solved, "near-fully-revealed puzzle must be solvable by naked singles"
    # The number of strategy-filled cells equals the number of empty cells
    assert len(order) == n_empty, f"order has {len(order)} entries; expected {n_empty}"
    # Each (r, c, d) in order is well-formed
    for r, c, d in order:
        assert 0 <= r < 9 and 0 <= c < 9
        assert 1 <= d <= 9
    # And the implied final grid matches the solution
    final = puzzle.copy()
    for r, c, d in order:
        final[r * 9 + c] = d
    np.testing.assert_array_equal(final, solution)


def test_seven_strategies_returns_false_on_an_unsolvable_with_just_strategies_puzzle():
    """A symmetric, near-empty puzzle that requires search/backtracking. The
    7-strategy solver should report 'not solvable' rather than infinite loop."""
    # A canonical 17-clue minimum-Sudoku-style puzzle (often requires backtracking)
    s = "000000010" "400000000" "020000000" "000050407" "008000300" "001090000" "300400200" "050100000" "000806000"
    puzzle = sudoku_string_to_tokens(s)
    solved, order = solve_with_seven_strategies(puzzle.copy())
    # We don't require a specific order length; we just require termination + solved=False.
    assert solved is False, "this puzzle should NOT be solvable by the 7 strategies alone"


def test_synthetic_easy_puzzle_is_consistent_with_solution():
    """Synthetic puzzles must be a subset of their solution (no contradictory clues)."""
    puzzle, solution = synthetic_easy_puzzle(num_clues=40, seed=0)
    assert puzzle.shape == (81,)
    assert solution.shape == (81,)
    # Wherever puzzle has a clue (nonzero), it must match the solution at that position
    clue_pos = puzzle != 0
    np.testing.assert_array_equal(puzzle[clue_pos], solution[clue_pos])
    # Solution must be a valid filled board
    assert _is_valid_sudoku(tokens_to_grid(solution))
