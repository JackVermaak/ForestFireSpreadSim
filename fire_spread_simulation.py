import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

# Constants
EMPTY, TREE, FIRE = 0, 1, 2

# Parameters
grid_size = 50
tree_density = 0.6
fire_prob = 0.5
wind_prob_boost = 0.4
wind_skip_prob = 0.5
timesteps = 100
# Set SEED to a whole number (for example 42) to make every run repeatable.
SEED = None
if SEED is not None:
    np.random.seed(SEED)
firebreak_rows = range(30, 35)
firebreak_on = False

# Wind state
wind_enabled = False
wind_map = {
    'North': (1, 0),   # fire moves south
    'South': (-1, 0),  # fire moves north
    'West': (0, 1),    # fire moves east
    'East': (0, -1),   # fire moves west
}
wind_direction = wind_map['East']  # matches the direction selected in the UI at start

# UI state
anim_running = False
ani = None
interval_ms = 200

# Initialize forest
def create_forest():
    forest = np.random.choice([EMPTY, TREE], size=(grid_size, grid_size), p=[1 - tree_density, tree_density])
    if firebreak_on:
        for r in firebreak_rows:
            forest[r, :] = EMPTY
    return forest

forest = create_forest()
initial_fire = []
burned_base = np.sum(forest == EMPTY)

# Stats counter
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.45)
stats_text = ax.text(0.02, -0.08, '', transform=ax.transAxes, fontsize=10)

def compute_stats():
    burned = np.sum(forest == EMPTY) - burned_base if anim_running else np.sum(forest == EMPTY) - burned_base
    trees = np.sum(forest == TREE)
    fire = np.sum(forest == FIRE)
    return f"Trees: {trees}  |  Burning: {fire}  |  Burned: {burned}"

stats_text.set_text(compute_stats())

# Simulation step
def step(forest):
    new_forest = forest.copy()
    for y in range(grid_size):
        for x in range(grid_size):
            if forest[y, x] == TREE:
                y_min = max(0, y - 1)
                y_max = min(grid_size, y + 2)
                x_min = max(0, x - 1)
                x_max = min(grid_size, x + 2)
                neighbors = forest[y_min:y_max, x_min:x_max]
                catch_fire = False

                if np.any(neighbors == FIRE):
                    prob = fire_prob
                    if wind_enabled:
                        wy, wx = y + wind_direction[0], x + wind_direction[1]
                        if 0 <= wy < grid_size and 0 <= wx < grid_size:
                            if forest[wy, wx] == FIRE:
                                prob = min(prob + wind_prob_boost, 1.0)
                    if np.random.rand() < prob:
                        catch_fire = True

                # Check wind skip fire
                if not catch_fire and wind_enabled:
                    wy2, wx2 = y + 2 * wind_direction[0], x + 2 * wind_direction[1]
                    wy1, wx1 = y + wind_direction[0], x + wind_direction[1]
                    if (0 <= wy1 < grid_size and 0 <= wx1 < grid_size and
                        0 <= wy2 < grid_size and 0 <= wx2 < grid_size):
                        if forest[wy1, wx1] == EMPTY and forest[wy2, wx2] == FIRE:
                            if np.random.rand() < wind_skip_prob:
                                catch_fire = True

                if catch_fire:
                    new_forest[y, x] = FIRE

            elif forest[y, x] == FIRE:
                new_forest[y, x] = EMPTY
    return new_forest

# Color map
cmap = ListedColormap(['#8B4513', '#228B22', '#FF4500'])

# Image
im = ax.imshow(forest, cmap=cmap, vmin=0, vmax=2)
ax.set_title("Forest Fire Simulation\nClick a tree to start fire")

# Grid lines
ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Wind direction arrow
arrow = ax.annotate('', xy=(0.9, 0.9), xytext=(0.9, 0.9),
                    arrowprops=dict(facecolor='blue', width=2, headwidth=8),
                    xycoords='axes fraction')
arrow.set_visible(False)

# Mouse click to start fire
def on_click(event):
    global forest, initial_fire, anim_running
    if anim_running:
        return
    if event.inaxes != ax:
        return
    x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
    if 0 <= x < grid_size and 0 <= y < grid_size:
        if forest[y, x] == TREE:
            forest[forest == FIRE] = TREE
            forest[y, x] = FIRE
            initial_fire.clear()
            initial_fire.append((y, x))
            im.set_array(forest)
            ax.set_title(f"Initial fire at ({y}, {x}) — Press Start")
            stats_text.set_text(compute_stats())
            fig.canvas.draw()

# Animation update
def update(frame):
    global forest, anim_running, ani
    forest[:] = step(forest)
    im.set_array(forest)
    stats_text.set_text(compute_stats())
    if not np.any(forest == FIRE):
        ani.event_source.stop()
        anim_running = False
        ax.set_title("Simulation complete")
    return [im]

# Start button
def start(event):
    global ani, anim_running, burned_base
    if not initial_fire:
        ax.set_title("Click a tree to ignite before starting")
        fig.canvas.draw()
        return
    if ani is not None:
        ani.event_source.stop()
    anim_running = True
    burned_base = np.sum(forest == EMPTY)
    ax.set_title("Simulation Running...")
    stats_text.set_text(compute_stats())
    ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=interval_ms, repeat=False)
    fig.canvas.draw()

# Pause/Resume
def pause_resume(event):
    global anim_running
    if ani is None:
        return
    if anim_running:
        ani.event_source.stop()
        anim_running = False
        btn_pause.label.set_text("Resume")
        ax.set_title("Simulation Paused")
    else:
        ani.event_source.start()
        anim_running = True
        btn_pause.label.set_text("Pause")
        ax.set_title("Simulation Running...")
    fig.canvas.draw()

# Reset
def reset(event):
    global forest, initial_fire, anim_running, ani, burned_base
    if ani is not None:
        ani.event_source.stop()
        ani = None
    anim_running = False
    initial_fire.clear()
    forest[:] = create_forest()
    burned_base = np.sum(forest == EMPTY)
    im.set_array(forest)
    btn_pause.label.set_text("Pause")
    ax.set_title("Forest Fire Simulation\nClick a tree to start fire")
    stats_text.set_text(compute_stats())
    fig.canvas.draw()

# Toggle firebreak
def toggle_firebreak(label):
    global firebreak_on, forest, burned_base
    firebreak_on = not firebreak_on
    forest[:] = create_forest()
    burned_base = np.sum(forest == EMPTY)
    im.set_array(forest)
    stats_text.set_text(compute_stats())
    fig.canvas.draw()

# Toggle wind
def toggle_wind(label):
    global wind_enabled
    wind_enabled = not wind_enabled
    arrow.set_visible(wind_enabled)
    update_arrow()
    fig.canvas.draw()

# Change wind direction
def change_wind_direction(label):
    global wind_direction
    wind_direction = wind_map[label]
    update_arrow()
    fig.canvas.draw()

def update_arrow():
    dx, dy = -wind_direction[1], wind_direction[0]  # flip horizontal component as well
    arrow.xy = (0.9 + 0.05 * dx, 0.9 + 0.05 * dy)
    arrow.set_position((0.9, 0.9))
    arrow.set_visible(wind_enabled)



# Buttons
axstart = plt.axes([0.1, 0.3, 0.15, 0.05])
btn_start = Button(axstart, 'Start')
btn_start.on_clicked(start)

axpause = plt.axes([0.4, 0.3, 0.15, 0.05])
btn_pause = Button(axpause, 'Pause')
btn_pause.on_clicked(pause_resume)

axreset = plt.axes([0.7, 0.3, 0.15, 0.05])
btn_reset = Button(axreset, 'Reset')
btn_reset.on_clicked(reset)

# Checkbox
axcheck = plt.axes([0.1, 0.15, 0.2, 0.1])
check = CheckButtons(axcheck, ['Firebreak', 'Wind'], [firebreak_on, wind_enabled])
check.on_clicked(lambda label: toggle_firebreak(label) if label == 'Firebreak' else toggle_wind(label))

# Wind direction dropdown
axradio = plt.axes([0.4, 0.13, 0.2, 0.12])
radio = RadioButtons(axradio, list(wind_map.keys()), active=3)
radio.on_clicked(change_wind_direction)

# Speed slider
axspeed = plt.axes([0.7, 0.15, 0.2, 0.03])
speed_slider = Slider(axspeed, 'Speed', 50, 1000, valinit=interval_ms, valstep=50)

# Forecasting Monte Carlo simulation
def forecast_fire_spread(trials=50, steps=30):
    """Run the fire forward `trials` times from the current forest.

    Returns, for each cell, the share of runs in which that cell caught fire
    at some point (0 = never, 1 = in every run).
    """
    burned_count = np.zeros_like(forest, dtype=float)
    for _ in range(trials):
        sim_forest = forest.copy()
        burned = sim_forest == FIRE
        for _ in range(steps):
            sim_forest = step(sim_forest)
            burned |= sim_forest == FIRE
            if not np.any(sim_forest == FIRE):
                break
        burned_count += burned
    return burned_count / trials

# Plot forecast heatmap
def show_forecast(event):
    forecast = forecast_fire_spread()
    fig_forecast, axf = plt.subplots(figsize=(6, 6))
    shown = axf.imshow(forecast, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    fig_forecast.colorbar(shown, ax=axf, fraction=0.046, pad=0.04, label='Chance of burning')
    axf.set_title("Forecasted Fire Spread Probability")
    axf.set_xticks([])
    axf.set_yticks([])
    plt.show()

# Forecast button
axforecast = plt.axes([0.4, 0.02, 0.2, 0.05])
btn_forecast = Button(axforecast, 'Forecast Spread')
btn_forecast.on_clicked(show_forecast)


def update_speed(val):
    global interval_ms
    # Map slider val so bigger val means smaller interval (faster)
    max_val = 1000
    min_val = 50
    # invert: when val is max_val, interval_ms is min_val (fastest)
    # when val is min_val, interval_ms is max_val (slowest)
    interval_ms = int(max_val + min_val - val)

    if ani is not None:
        ani.event_source.interval = interval_ms


speed_slider.on_changed(update_speed)

# Mouse click binding
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
