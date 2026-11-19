# Forest fire spread simulation

An interactive wildfire simulator in Python, built on a stochastic cellular automaton. You start a fire by clicking a tree, then watch it spread with optional wind and a firebreak. A Monte Carlo forecast runs 50 possible futures and maps where the fire is most likely to reach.

Built as an individual project for an Advanced Modelling and Simulation course.

![The fire spreading south with the wind and stopping at a firebreak](images/demo.gif)

## Run it

```
pip install -r requirements.txt
python fire_spread_simulation.py
```

1. Click a tree to place the first fire.
2. Press **Start**. **Pause** and **Reset** do what they say.
3. Tick **Wind** and pick a direction, or tick **Firebreak** to clear a strip across the forest.
4. Press **Forecast Spread** to see the chance of each cell burning.

## How the model works

- The forest is a 50 × 50 grid. Each cell is empty ground, a tree or fire, and 60% of cells start as trees.
- Each step, a tree next to a burning cell (all 8 neighbours count) catches fire with a 50% chance. Burning cells burn out and become empty.
- With wind on, a tree directly downwind of a fire has a 90% chance of catching, and embers can jump one empty cell in the wind direction (50% chance).
- The firebreak clears rows 30 to 34.
- The forecast copies the current forest, runs it forward 50 times for up to 30 steps, and shows the share of runs in which each cell caught fire.

Set `SEED` near the top of the file to a whole number to make runs repeatable.

## Limits

- The step function loops over every cell in Python, so larger grids get slow. Vectorising it with NumPy would speed it up a lot.
- Wind has four directions and a fixed strength.
- Terrain, fuel types and moisture aren't modelled.

The full write-up is in [docs/forest_fire_sim_documentation.pdf](docs/forest_fire_sim_documentation.pdf). There's also a short page on [jackvermaak.com](https://jackvermaak.com/projects/fire-simulation.html).
