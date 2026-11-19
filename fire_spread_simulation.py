"""Interactive Matplotlib interface for the forest fire model in ``fire_model.py``.

Run it with ``python fire_spread_simulation.py``.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

import fire_model as fm
from fire_model import EMPTY, FIRE, TREE

# Parameters
GRID_SIZE = 50
TREE_DENSITY = 0.6
TIMESTEPS = 100
FIREBREAK_ROWS = fm.DEFAULT_FIREBREAK_ROWS
# Set SEED to a whole number (for example 42) to make every run repeatable.
SEED = None
START_WIND = "East"  # matches the direction selected in the UI at start


class FireApp:
    def __init__(self, seed=SEED):
        self.rng = fm.make_rng(seed)
        self.params = fm.DEFAULT_PARAMS
        self.firebreak_on = False
        self.wind_enabled = False
        self.wind_name = START_WIND
        self.anim_running = False
        self.ani = None
        self.interval_ms = 200
        self.initial_fire = None

        self.forest = self.new_forest()
        self.burned_base = np.sum(self.forest == EMPTY)
        self.build_ui()

    # ----- model glue -----

    @property
    def wind(self):
        return fm.WIND[self.wind_name] if self.wind_enabled else None

    def new_forest(self):
        rows = FIREBREAK_ROWS if self.firebreak_on else None
        return fm.create_forest(GRID_SIZE, TREE_DENSITY, self.rng, rows)

    def compute_stats(self):
        burned = np.sum(self.forest == EMPTY) - self.burned_base
        trees = np.sum(self.forest == TREE)
        fire = np.sum(self.forest == FIRE)
        return f"Trees: {trees}  |  Burning: {fire}  |  Burned: {burned}"

    # ----- drawing -----

    def build_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        fig, ax = self.fig, self.ax
        plt.subplots_adjust(bottom=0.45)
        self.stats_text = ax.text(0.02, -0.08, "", transform=ax.transAxes, fontsize=10)

        cmap = ListedColormap(["#8B4513", "#228B22", "#FF4500"])
        self.im = ax.imshow(self.forest, cmap=cmap, vmin=0, vmax=2)
        ax.set_title("Forest Fire Simulation\nClick a tree to start fire")

        ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.3)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        self.arrow = ax.annotate(
            "", xy=(0.9, 0.9), xytext=(0.9, 0.9),
            arrowprops=dict(facecolor="blue", width=2, headwidth=8),
            xycoords="axes fraction",
        )
        self.arrow.set_visible(False)

        self.btn_start = Button(plt.axes([0.1, 0.3, 0.15, 0.05]), "Start")
        self.btn_start.on_clicked(self.start)
        self.btn_pause = Button(plt.axes([0.4, 0.3, 0.15, 0.05]), "Pause")
        self.btn_pause.on_clicked(self.pause_resume)
        self.btn_reset = Button(plt.axes([0.7, 0.3, 0.15, 0.05]), "Reset")
        self.btn_reset.on_clicked(self.reset)

        self.check = CheckButtons(plt.axes([0.1, 0.15, 0.2, 0.1]), ["Firebreak", "Wind"],
                                  [self.firebreak_on, self.wind_enabled])
        self.check.on_clicked(lambda label: self.toggle_firebreak() if label == "Firebreak" else self.toggle_wind())

        names = list(fm.WIND)
        self.radio = RadioButtons(plt.axes([0.4, 0.13, 0.2, 0.12]), names, active=names.index(START_WIND))
        self.radio.on_clicked(self.change_wind_direction)

        self.speed_slider = Slider(plt.axes([0.7, 0.15, 0.2, 0.03]), "Speed", 50, 1000,
                                   valinit=self.interval_ms, valstep=50)
        self.speed_slider.on_changed(self.update_speed)

        self.btn_forecast = Button(plt.axes([0.4, 0.02, 0.2, 0.05]), "Forecast Spread")
        self.btn_forecast.on_clicked(self.show_forecast)

        fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.refresh()

    def refresh(self, title=None):
        self.im.set_array(self.forest)
        self.stats_text.set_text(self.compute_stats())
        if title:
            self.ax.set_title(title)
        self.fig.canvas.draw_idle()

    def update_arrow(self):
        dy, dx = fm.WIND[self.wind_name]
        self.arrow.xy = (0.9 - 0.05 * dx, 0.9 + 0.05 * dy)
        self.arrow.set_position((0.9, 0.9))
        self.arrow.set_visible(self.wind_enabled)

    # ----- callbacks -----

    def on_click(self, event):
        if self.anim_running or event.inaxes != self.ax:
            return
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and self.forest[y, x] == TREE:
            self.forest[self.forest == FIRE] = TREE
            self.forest[y, x] = FIRE
            self.initial_fire = (y, x)
            self.refresh(f"Initial fire at ({y}, {x}) — Press Start")

    def update(self, frame):
        self.forest = fm.step(self.forest, self.rng, self.wind, self.params)
        self.im.set_array(self.forest)
        self.stats_text.set_text(self.compute_stats())
        if not np.any(self.forest == FIRE):
            self.ani.event_source.stop()
            self.anim_running = False
            self.ax.set_title("Simulation complete")
        return [self.im]

    def start(self, event):
        if self.initial_fire is None:
            self.refresh("Click a tree to ignite before starting")
            return
        if self.ani is not None:
            self.ani.event_source.stop()
        self.anim_running = True
        self.burned_base = np.sum(self.forest == EMPTY)
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=TIMESTEPS,
                                           interval=self.interval_ms, repeat=False)
        self.refresh("Simulation Running...")

    def pause_resume(self, event):
        if self.ani is None:
            return
        if self.anim_running:
            self.ani.event_source.stop()
            self.anim_running = False
            self.btn_pause.label.set_text("Resume")
            self.refresh("Simulation Paused")
        else:
            self.ani.event_source.start()
            self.anim_running = True
            self.btn_pause.label.set_text("Pause")
            self.refresh("Simulation Running...")

    def reset(self, event=None):
        if self.ani is not None:
            self.ani.event_source.stop()
            self.ani = None
        self.anim_running = False
        self.initial_fire = None
        self.forest = self.new_forest()
        self.burned_base = np.sum(self.forest == EMPTY)
        self.btn_pause.label.set_text("Pause")
        self.refresh("Forest Fire Simulation\nClick a tree to start fire")

    def toggle_firebreak(self):
        self.firebreak_on = not self.firebreak_on
        self.reset()

    def toggle_wind(self):
        self.wind_enabled = not self.wind_enabled
        self.update_arrow()
        self.fig.canvas.draw_idle()

    def change_wind_direction(self, label):
        self.wind_name = label
        self.update_arrow()
        self.fig.canvas.draw_idle()

    def update_speed(self, val):
        # Bigger slider value = shorter interval = faster.
        self.interval_ms = int(1000 + 50 - val)
        if self.ani is not None:
            self.ani.event_source.interval = self.interval_ms

    def show_forecast(self, event):
        chance = fm.forecast(self.forest, trials=50, steps=30, rng=self.rng, wind=self.wind, params=self.params)
        fig_f, axf = plt.subplots(figsize=(6, 6))
        shown = axf.imshow(chance, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
        fig_f.colorbar(shown, ax=axf, fraction=0.046, pad=0.04, label="Chance of burning")
        axf.set_title("Forecasted Fire Spread Probability")
        axf.set_xticks([])
        axf.set_yticks([])
        plt.show()


def main():
    app = FireApp()
    plt.show()
    return app


if __name__ == "__main__":
    main()
