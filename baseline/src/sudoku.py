"""Sudoku data: representation, 7-strategy filter, and dataset construction.

Paper §9.3 / Appendix D.2:
  - Source dataset: Radcliffe (2020), Kaggle 'radcliffe/3-million-sudoku-puzzles-with-ratings'.
  - Train / easy-test: puzzles solvable by the 7 fixed strategies (no backtracking).
  - Hard test (Table 5): the complement (~1M puzzles) — requires backtracking
    or strategies outside the 7.

Representation:
  - 81 tokens (9×9 grid, row-major).
  - Token values: 1..9 for digits, 0 for empty/mask.
  - mask_token_id = 0 (no collision because legitimate digits never equal 0).
  - vocab_size = 10.

Status (Phase 4):
  - Data loading + 7-strategy filter implemented.
  - MDM training is supported via the same `src/train.py` entry point.
  - **ARM-with-ordering** baseline is *intentionally deferred*. The paper says
    (Appendix D.2): "we use the codebase of (Ye et al., 2024)" and the ARM-with-
    ordering baseline traces back to Shah et al. 2024's alternating-token format
    (paper-notes §12.9). Code-audit P0-2 / P0-3 documented why the existing
    Colab implementation is broken. The follow-up will:
      1. Clone https://github.com/HKUNLP/diffusion-vs-ar (Ye et al. 2024) and
         check whether they ship a baseline that includes the Shah ordering loader.
      2. If yes, vendor that adapter into baseline/src/.
      3. If no, vendor https://github.com/kulinshah98/logic-puzzles directly.
    Until that step, the baseline reports MDM Sudoku only (Table 2 row 'MDM
    (vanilla / Top probability / Top prob. margin)'), without the ARM rows.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUDOKU_SEQ_LEN = 81
SUDOKU_VOCAB_SIZE = 10
SUDOKU_MASK_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def sudoku_string_to_tokens(s: str) -> np.ndarray:
    """Convert an 81-char Sudoku string ('.' or '0' or digit) to int tokens (0=empty, 1-9=digit)."""
    if len(s) != 81:
        raise ValueError(f"Sudoku string must be length 81, got {len(s)}")
    out = np.empty(81, dtype=np.int64)
    for i, ch in enumerate(s):
        out[i] = 0 if ch in ".0" else int(ch)
    return out


def tokens_to_grid(tokens: np.ndarray) -> list[list[int]]:
    """Convert a length-81 token vector to a 9x9 nested list."""
    return [[int(tokens[r * 9 + c]) for c in range(9)] for r in range(9)]


def grid_to_tokens(grid: list[list[int]]) -> np.ndarray:
    return np.asarray([grid[r][c] for r in range(9) for c in range(9)], dtype=np.int64)


# ---------------------------------------------------------------------------
# 7-strategy solver / filter
# ---------------------------------------------------------------------------
# Paper Appendix D.2 / Shah et al. 2024: easy puzzles are those solvable with
# *only* these 7 strategies (no backtracking). The strategies, in the order we
# attempt them per inner iteration:
#   1. Naked Singles
#   2. Hidden Singles
#   3. Naked Pairs
#   4. Hidden Pairs
#   5. Pointing Pairs
#   6. Box / Line Reduction
#   7. Naked Triples
#
# Implementation ported from the prior Colab 1 cells (which were paper-correct
# in their logic), cleaned up and made deterministic.

def _units() -> list[list[tuple[int, int]]]:
    """All 27 units of a Sudoku: 9 rows + 9 cols + 9 boxes."""
    out: list[list[tuple[int, int]]] = []
    for i in range(9):
        out.append([(i, j) for j in range(9)])         # row i
        out.append([(j, i) for j in range(9)])         # col i
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            out.append([(br + r, bc + c) for r in range(3) for c in range(3)])
    return out


_UNITS = _units()


def _candidates(grid: list[list[int]]) -> list[list[set[int]]]:
    cands: list[list[set[int]]] = [[set() for _ in range(9)] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                continue
            used: set[int] = set(grid[r])               # row
            used |= {grid[rr][c] for rr in range(9)}   # col
            br, bc = 3 * (r // 3), 3 * (c // 3)
            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    used.add(grid[rr][cc])
            cands[r][c] = set(range(1, 10)) - used
    return cands


def _naked_singles(grid, cands) -> bool:
    progress = False
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0 and len(cands[r][c]) == 1:
                grid[r][c] = next(iter(cands[r][c]))
                cands[r][c] = set()
                progress = True
    return progress


def _hidden_singles(grid, cands) -> bool:
    progress = False
    for unit in _UNITS:
        for d in range(1, 10):
            positions = [(r, c) for r, c in unit if grid[r][c] == 0 and d in cands[r][c]]
            if len(positions) == 1:
                r, c = positions[0]
                grid[r][c] = d
                cands[r][c] = set()
                progress = True
    return progress


def _naked_pairs(grid, cands) -> bool:
    progress = False
    for unit in _UNITS:
        empty = [(r, c) for r, c in unit if grid[r][c] == 0]
        for i in range(len(empty)):
            for j in range(i + 1, len(empty)):
                r1, c1 = empty[i]
                r2, c2 = empty[j]
                if len(cands[r1][c1]) == 2 and cands[r1][c1] == cands[r2][c2]:
                    pair = cands[r1][c1]
                    for r, c in empty:
                        if (r, c) in {(r1, c1), (r2, c2)}:
                            continue
                        before = len(cands[r][c])
                        cands[r][c] -= pair
                        if len(cands[r][c]) < before:
                            progress = True
    return progress


def _hidden_pairs(grid, cands) -> bool:
    progress = False
    for unit in _UNITS:
        empty = [(r, c) for r, c in unit if grid[r][c] == 0]
        for d1 in range(1, 10):
            for d2 in range(d1 + 1, 10):
                pos_d1 = {(r, c) for r, c in empty if d1 in cands[r][c]}
                pos_d2 = {(r, c) for r, c in empty if d2 in cands[r][c]}
                if pos_d1 == pos_d2 and len(pos_d1) == 2:
                    pair = {d1, d2}
                    for r, c in pos_d1:
                        before = len(cands[r][c])
                        cands[r][c] &= pair
                        if len(cands[r][c]) < before:
                            progress = True
    return progress


def _pointing_pairs(grid, cands) -> bool:
    progress = False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [(br + r, bc + c) for r in range(3) for c in range(3)]
            box_set = set(box)
            for d in range(1, 10):
                positions = [(r, c) for r, c in box if grid[r][c] == 0 and d in cands[r][c]]
                if len(positions) < 2:
                    continue
                rows = {r for r, _ in positions}
                cols = {c for _, c in positions}
                if len(rows) == 1:
                    row = next(iter(rows))
                    for c in range(9):
                        if (row, c) in box_set or grid[row][c] != 0:
                            continue
                        if d in cands[row][c]:
                            cands[row][c].discard(d)
                            progress = True
                if len(cols) == 1:
                    col = next(iter(cols))
                    for r in range(9):
                        if (r, col) in box_set or grid[r][col] != 0:
                            continue
                        if d in cands[r][col]:
                            cands[r][col].discard(d)
                            progress = True
    return progress


def _box_line_reduction(grid, cands) -> bool:
    progress = False
    for i in range(9):
        # row i
        for d in range(1, 10):
            positions = [(i, c) for c in range(9) if grid[i][c] == 0 and d in cands[i][c]]
            if len(positions) < 2:
                continue
            boxes = {(i // 3, c // 3) for _, c in positions}
            if len(boxes) == 1:
                box_r, box_c = next(iter(boxes))
                for r in range(box_r * 3, box_r * 3 + 3):
                    for c in range(box_c * 3, box_c * 3 + 3):
                        if r != i and grid[r][c] == 0 and d in cands[r][c]:
                            cands[r][c].discard(d)
                            progress = True
        # col i
        for d in range(1, 10):
            positions = [(r, i) for r in range(9) if grid[r][i] == 0 and d in cands[r][i]]
            if len(positions) < 2:
                continue
            boxes = {(r // 3, i // 3) for r, _ in positions}
            if len(boxes) == 1:
                box_r, box_c = next(iter(boxes))
                for r in range(box_r * 3, box_r * 3 + 3):
                    for c in range(box_c * 3, box_c * 3 + 3):
                        if c != i and grid[r][c] == 0 and d in cands[r][c]:
                            cands[r][c].discard(d)
                            progress = True
    return progress


def _naked_triples(grid, cands) -> bool:
    progress = False
    for unit in _UNITS:
        empty = [(r, c) for r, c in unit if grid[r][c] == 0 and 1 <= len(cands[r][c]) <= 3]
        for combo in combinations(empty, 3):
            union: set[int] = set()
            for r, c in combo:
                union |= cands[r][c]
            if len(union) == 3:
                combo_set = set(combo)
                for r, c in unit:
                    if grid[r][c] == 0 and (r, c) not in combo_set:
                        before = len(cands[r][c])
                        cands[r][c] -= union
                        if len(cands[r][c]) < before:
                            progress = True
    return progress


_STRATEGIES = (
    _naked_singles,
    _hidden_singles,
    _naked_pairs,
    _hidden_pairs,
    _pointing_pairs,
    _box_line_reduction,
    _naked_triples,
)


def solve_with_seven_strategies(puzzle: np.ndarray, max_iters: int = 200) -> tuple[bool, list[tuple[int, int, int]]]:
    """Try to solve a Sudoku using only the 7 strategies (no backtracking).

    Returns:
        solved (bool): True iff every cell is filled and consistent.
        order (list[(row, col, digit)]): the cells filled in by the strategies,
            in the order they were determined. Shape: list of 81 - num_clues if solved.

    The `order` is the data needed for the ARM-with-ordering baseline (Shah et al. 2024).
    Even if solving fails, we return what we managed to fill in (useful for debugging).
    """
    grid = tokens_to_grid(puzzle)
    order: list[tuple[int, int, int]] = []

    for _ in range(max_iters):
        cands = _candidates(grid)
        empty_before = {(r, c) for r in range(9) for c in range(9) if grid[r][c] == 0}
        if not empty_before:
            return True, order

        any_progress = False
        for strategy in _STRATEGIES:
            if strategy(grid, cands):
                any_progress = True
                # Record any cells newly filled by this strategy
                still_empty = {(r, c) for r in range(9) for c in range(9) if grid[r][c] == 0}
                for r, c in sorted(empty_before - still_empty, key=lambda rc: rc[0] * 9 + rc[1]):
                    order.append((r, c, grid[r][c]))
                empty_before = still_empty
                cands = _candidates(grid)         # refresh after each strategy
        if not any_progress:
            break

    solved = all(grid[r][c] != 0 for r in range(9) for c in range(9))
    return solved, order


# ---------------------------------------------------------------------------
# Dataset construction (from a Kaggle CSV)
# ---------------------------------------------------------------------------

@dataclass
class SudokuSplits:
    train_clues: np.ndarray
    train_solutions: np.ndarray
    train_orders: list[list[tuple[int, int, int]]]
    test_easy_clues: np.ndarray
    test_easy_solutions: np.ndarray
    test_hard_clues: np.ndarray
    test_hard_solutions: np.ndarray


def filter_and_split(
    csv_path: str,
    *,
    test_easy_size: int = 10_000,
    seed: int = 42,
    max_puzzles: int | None = None,
    progress: bool = True,
) -> SudokuSplits:
    """Load Radcliffe 2020 Kaggle CSV, run 7-strategy filter, and produce splits.

    Returns a SudokuSplits dataclass with train / test_easy / test_hard arrays.
    """
    import pandas as pd  # imported lazily so non-Sudoku users don't need pandas

    df = pd.read_csv(csv_path)
    if max_puzzles is not None:
        df = df.head(max_puzzles)

    # The Kaggle CSV uses columns 'puzzle' and 'solution'; fall back to the first two
    # columns if the names differ.
    puzzle_col = "puzzle" if "puzzle" in df.columns else df.columns[0]
    solution_col = "solution" if "solution" in df.columns else df.columns[1]

    easy: list[tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]] = []
    hard: list[tuple[np.ndarray, np.ndarray]] = []

    iterator: Iterable = df.itertuples(index=False)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(df), desc="Filtering Sudoku puzzles")
        except ImportError:
            pass

    for row in iterator:
        puzzle_str = getattr(row, puzzle_col, None)
        solution_str = getattr(row, solution_col, None)
        if puzzle_str is None or solution_str is None:
            continue
        if len(puzzle_str) != 81 or len(solution_str) != 81:
            continue
        puzzle = sudoku_string_to_tokens(puzzle_str)
        solution = np.asarray([int(c) for c in solution_str], dtype=np.int64)
        ok, order = solve_with_seven_strategies(puzzle.copy())
        if ok:
            easy.append((puzzle, solution, order))
        else:
            hard.append((puzzle, solution))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(easy))
    test_easy_n = min(test_easy_size, len(easy) // 10)
    train_idx = perm[test_easy_n:]
    test_easy_idx = perm[:test_easy_n]

    train_clues = np.stack([easy[i][0] for i in train_idx]) if len(train_idx) else np.empty((0, 81), dtype=np.int64)
    train_solutions = np.stack([easy[i][1] for i in train_idx]) if len(train_idx) else np.empty((0, 81), dtype=np.int64)
    train_orders = [easy[i][2] for i in train_idx]

    te_clues = np.stack([easy[i][0] for i in test_easy_idx]) if len(test_easy_idx) else np.empty((0, 81), dtype=np.int64)
    te_solutions = np.stack([easy[i][1] for i in test_easy_idx]) if len(test_easy_idx) else np.empty((0, 81), dtype=np.int64)

    th_clues = np.stack([h[0] for h in hard]) if len(hard) else np.empty((0, 81), dtype=np.int64)
    th_solutions = np.stack([h[1] for h in hard]) if len(hard) else np.empty((0, 81), dtype=np.int64)

    return SudokuSplits(
        train_clues=train_clues,
        train_solutions=train_solutions,
        train_orders=train_orders,
        test_easy_clues=te_clues,
        test_easy_solutions=te_solutions,
        test_hard_clues=th_clues,
        test_hard_solutions=th_solutions,
    )


# ---------------------------------------------------------------------------
# Mini-synthetic Sudoku for testing without the Kaggle download
# ---------------------------------------------------------------------------

def _is_valid_sudoku(grid: list[list[int]]) -> bool:
    """Return True iff the grid is a fully filled valid Sudoku."""
    for r in range(9):
        if sorted(grid[r]) != list(range(1, 10)):
            return False
    for c in range(9):
        if sorted(grid[r][c] for r in range(9)) != list(range(1, 10)):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = [grid[br + r][bc + c] for r in range(3) for c in range(3)]
            if sorted(block) != list(range(1, 10)):
                return False
    return True


# A single canonical solved Sudoku (used by tests / fixtures so we can construct
# puzzles deterministically without Kaggle data).
_CANONICAL_SOLVED = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]
assert _is_valid_sudoku(_CANONICAL_SOLVED), "canonical Sudoku is invalid"


def synthetic_easy_puzzle(num_clues: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Construct a synthetic puzzle by removing cells from the canonical solved grid.

    Returns (puzzle, solution). The puzzle may not be 7-strategy-solvable for low
    clue counts, but it is always logically consistent (it has at least one solution).
    """
    rng = np.random.default_rng(seed)
    solution = grid_to_tokens(_CANONICAL_SOLVED)
    keep = rng.choice(81, size=num_clues, replace=False)
    puzzle = np.zeros(81, dtype=np.int64)
    puzzle[keep] = solution[keep]
    return puzzle, solution
