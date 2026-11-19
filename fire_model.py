"""Forest fire model: a stochastic cellular automaton with wind and a Monte Carlo forecast.

This module has no plotting code, so it can be imported and tested on its own.
The Matplotlib interface lives in ``fire_spread_simulation.py``.

Each cell of the grid is EMPTY (bare ground or burned out), TREE or FIRE. Every step:

* a tree with at least one burning cell among its 8 neighbours catches fire with
  probability ``fire_prob``;
* with wind on, a tree whose upwind neighbour is burning catches with
  ``fire_prob + wind_prob_boost`` instead;
* with wind on, embers can jump one empty cell: a tree that did not catch fire, with an
  empty cell upwind and a burning cell two cells upwind, catches with ``wind_skip_prob``;
* burning cells burn out and become EMPTY.

The whole grid is updated at once with NumPy array operations.
"""

from dataclasses import dataclass

import numpy as np

EMPTY, TREE, FIRE = 0, 1, 2

# Offset (dy, dx) from a tree to its upwind cell, i.e. the cell whose fire the wind
# carries onto the tree. Row 0 is the top of the grid as drawn by imshow, so a positive
# dy means "the row below". These match the original UI labels and the on-screen arrow.
WIND = {
    "North": (1, 0),
    "South": (-1, 0),
    "West": (0, 1),
    "East": (0, -1),
}


@dataclass(frozen=True)
class FireParams:
    """Spread probabilities. The defaults are the ones the simulator uses."""

    fire_prob: float = 0.5
    wind_prob_boost: float = 0.4
    wind_skip_prob: float = 0.5


DEFAULT_PARAMS = FireParams()
DEFAULT_FIREBREAK_ROWS = range(30, 35)


def make_rng(seed=None):
    """A NumPy random generator; pass a whole number to make runs repeatable."""
    return np.random.default_rng(seed)


def create_forest(size=50, density=0.6, rng=None, firebreak_rows=None):
    """A ``size`` x ``size`` grid where each cell is a tree with probability ``density``.

    ``firebreak_rows`` (an iterable of row indices) are cleared to EMPTY across the full width.
    """
    rng = make_rng() if rng is None else rng
    forest = np.where(rng.random((size, size)) < density, TREE, EMPTY).astype(np.int8)
    if firebreak_rows is not None:
        forest[list(firebreak_rows), :] = EMPTY
    return forest


def _shifted(padded, pad, dy, dx, shape):
    """View of the padded grid such that result[y, x] == grid[y + dy, x + dx].

    Cells outside the grid read as the padding value (EMPTY).
    """
    h, w = shape
    return padded[pad + dy : pad + dy + h, pad + dx : pad + dx + w]


def step(forest, rng=None, wind=None, params=DEFAULT_PARAMS):
    """Advance the fire by one step and return the new grid (the input is not changed).

    ``wind`` is ``None`` for no wind, or a (dy, dx) offset from ``WIND``.
    """
    rng = make_rng() if rng is None else rng
    shape = forest.shape
    pad = 2
    padded = np.pad(forest, pad, constant_values=EMPTY)
    fire = padded == FIRE

    # Any of the 8 neighbours burning.
    near_fire = np.zeros(shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                near_fire |= _shifted(fire, pad, dy, dx, shape)

    prob = np.where(near_fire, params.fire_prob, 0.0)
    if wind is not None:
        wy, wx = wind
        upwind_fire = _shifted(fire, pad, wy, wx, shape)
        prob = np.where(upwind_fire, min(params.fire_prob + params.wind_prob_boost, 1.0), prob)

    trees = forest == TREE
    catches = trees & (rng.random(shape) < prob)

    if wind is not None:
        upwind_empty = _shifted(padded, pad, wy, wx, shape) == EMPTY
        two_up_fire = _shifted(fire, pad, 2 * wy, 2 * wx, shape)
        # Outside the grid reads as EMPTY, never FIRE, so embers cannot come from off the map.
        jump = trees & ~catches & upwind_empty & two_up_fire
        catches |= jump & (rng.random(shape) < params.wind_skip_prob)

    new = forest.copy()
    new[forest == FIRE] = EMPTY
    new[catches] = FIRE
    return new


def run(forest, rng=None, wind=None, params=DEFAULT_PARAMS, max_steps=1000):
    """Run until the fire is out (or ``max_steps``). Returns (final grid, burned mask).

    The burned mask marks every cell that was on fire at any point, including at the start.
    """
    rng = make_rng() if rng is None else rng
    burned = forest == FIRE
    for _ in range(max_steps):
        if not (forest == FIRE).any():
            break
        forest = step(forest, rng, wind, params)
        burned |= forest == FIRE
    return forest, burned


def forecast(forest, trials=50, steps=30, rng=None, wind=None, params=DEFAULT_PARAMS):
    """Monte Carlo forecast: run the fire forward ``trials`` times for up to ``steps`` steps.

    Returns, for each cell, the share of runs in which that cell caught fire at some
    point (0 = never, 1 = in every run).
    """
    rng = make_rng() if rng is None else rng
    burned_count = np.zeros(forest.shape, dtype=float)
    for _ in range(trials):
        _, burned = run(forest, rng, wind, params, max_steps=steps)
        burned_count += burned
    return burned_count / trials
