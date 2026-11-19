import numpy as np
import pytest

import fire_model as fm
from fire_model import EMPTY, FIRE, TREE


def full_forest(size=50):
    return np.full((size, size), TREE, dtype=np.int8)


def test_create_forest_is_repeatable_with_a_seed():
    a = fm.create_forest(rng=fm.make_rng(42))
    b = fm.create_forest(rng=fm.make_rng(42))
    assert np.array_equal(a, b)
    assert 0.5 < np.mean(a == TREE) < 0.7  # about 60 % trees


def test_firebreak_rows_are_cleared():
    forest = fm.create_forest(rng=fm.make_rng(1), firebreak_rows=range(30, 35))
    assert np.all(forest[30:35] == EMPTY)


def test_burning_cells_burn_out():
    forest = np.full((5, 5), EMPTY, dtype=np.int8)
    forest[2, 2] = FIRE
    assert np.all(fm.step(forest, fm.make_rng(0)) == EMPTY)


def test_step_does_not_change_its_input():
    forest = full_forest(10)
    forest[5, 5] = FIRE
    before = forest.copy()
    fm.step(forest, fm.make_rng(0), wind=fm.WIND["East"])
    assert np.array_equal(forest, before)


def test_fire_only_reaches_neighbours():
    forest = full_forest(9)
    forest[4, 4] = FIRE
    params = fm.FireParams(fire_prob=1.0)
    new = fm.step(forest, fm.make_rng(0), params=params)
    assert np.all(new[3:6, 3:6][np.arange(9).reshape(3, 3) != 4] == FIRE)
    assert new[4, 4] == EMPTY
    assert np.sum(new == FIRE) == 8


@pytest.mark.parametrize("seed", range(10))
def test_fire_cannot_cross_full_width_firebreak_without_wind(seed):
    rng = fm.make_rng(seed)
    forest = full_forest()
    forest[30:35, :] = EMPTY
    forest[0, :] = FIRE  # the whole top row is on fire
    final, burned = fm.run(forest, rng, wind=None, params=fm.FireParams(fire_prob=1.0))
    assert burned[:30].any()
    assert not burned[35:].any()
    assert np.all(final[35:] == TREE)


@pytest.mark.parametrize("name", list(fm.WIND))
def test_five_row_firebreak_also_stops_wind_driven_fire(name):
    forest = full_forest()
    forest[30:35, :] = EMPTY
    forest[0:30][fm.make_rng(3).random((30, 50)) < 0.3] = FIRE
    _, burned = fm.run(forest, fm.make_rng(3), wind=fm.WIND[name], params=fm.FireParams(1.0, 0.0, 1.0))
    assert not burned[35:].any()


def test_wind_embers_jump_a_one_cell_gap():
    # Fire in column 2, an empty column 3, trees from column 4. Only embers can cross.
    forest = full_forest(7)
    forest[:, 3] = EMPTY
    forest[:, 2] = FIRE
    params = fm.FireParams(fire_prob=0.0, wind_prob_boost=0.0, wind_skip_prob=1.0)
    no_wind = fm.step(forest, fm.make_rng(0), wind=None, params=params)
    toward_trees = fm.step(forest, fm.make_rng(0), wind=fm.WIND["East"], params=params)  # upwind = x - 1
    away_from_trees = fm.step(forest, fm.make_rng(0), wind=fm.WIND["West"], params=params)  # upwind = x + 1
    assert not np.any(no_wind[:, 4] == FIRE)
    assert np.all(toward_trees[:, 4] == FIRE)
    assert not np.any(away_from_trees[:, 4] == FIRE)


def test_wind_boost_raises_the_catch_rate():
    forest = full_forest(200)
    forest[:, 100] = FIRE
    rng = fm.make_rng(7)
    calm = fm.step(forest, rng)
    windy = fm.step(forest, rng, wind=fm.WIND["East"])  # fire is upwind of column 101
    assert abs(np.mean(calm[:, 101] == FIRE) - 0.5) < 0.1
    assert abs(np.mean(windy[:, 101] == FIRE) - 0.9) < 0.07


@pytest.mark.parametrize("wind", [None, fm.WIND["North"], fm.WIND["East"]])
def test_forecast_values_are_probabilities(wind):
    rng = fm.make_rng(11)
    forest = fm.create_forest(rng=rng)
    forest[25, 25] = FIRE
    chance = fm.forecast(forest, trials=20, steps=30, rng=rng, wind=wind)
    assert chance.shape == forest.shape
    assert np.all((chance >= 0) & (chance <= 1))
    assert chance[25, 25] == 1.0  # the starting fire is burning in every run
    assert np.all(chance[forest == EMPTY] == 0)  # bare ground never burns


def test_forecast_is_repeatable_with_a_seed():
    forest = fm.create_forest(rng=fm.make_rng(5))
    forest[10, 10] = FIRE
    a = fm.forecast(forest, trials=10, rng=fm.make_rng(9))
    b = fm.forecast(forest, trials=10, rng=fm.make_rng(9))
    assert np.array_equal(a, b)
