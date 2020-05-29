#   pykinsim -- Python Kinematics Simulator
#   Copyright (C) 2020  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

class MatplotlibVisHandle:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.objs = {}

def _visualize_matplotlib(raw, handle, ax, aabb):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Fetch the figure and the axis object
    if handle is None:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # Initial axis setup
        ax.set_proj_type('ortho')
        ax.set_xlim(*aabb[0])
        ax.set_ylim(*aabb[1])
        ax.set_zlim(*aabb[2])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.set_xticks(np.arange(np.floor(aabb[0][0]), np.ceil(aabb[0][1]) + 0.1, 1))
        ax.set_yticks(np.arange(np.floor(aabb[1][0]), np.ceil(aabb[1][1]) + 0.1, 1))
        ax.set_zticks(np.arange(np.floor(aabb[2][0]), np.ceil(aabb[2][1]) + 0.1, 1))
        ax.grid(False)
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.view_init(elev=0.0, azim=-90.0)


        handle = MatplotlibVisHandle(fig, ax)
    else:
        fig, ax = handle.fig, handle.ax

    # Get the axis x-limit, y-limit, and z-limit
    aabb_min = np.array([ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]])
    aabb_max = np.array([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]])

    marker_size = 25 / np.max(aabb_max - aabb_min)
    axis_size = 0.125 * (aabb_max - aabb_min)

    # Iterate over the drawing commands and draw them
    idx = [0]
    def obj(cbck):
        idx[0] += 1
        if not idx[0] in handle.objs:
            handle.objs[idx[0]] = cbck()
        return handle.objs[idx[0]]

    for cmd in raw:
        if cmd["type"] == "link":
            line, = obj(lambda: ax.plot([], [], [], linestyle="-", color="k"))
            line.set_data_3d(
                [cmd["src"][0], cmd["tar"][0]],
                [cmd["src"][1], cmd["tar"][1]],
                [cmd["src"][2], cmd["tar"][2]]
            )
        elif cmd["type"] == "axis":
            colors = {"x": "r", "y": "g", "z": "b"}
            line, = obj(lambda: ax.plot([], [], [], linestyle="-", color=colors[cmd["class"]],))
            p, q = cmd["src"], cmd["src"] + axis_size * cmd["dir"]
            line.set_data_3d([p[0], q[0]], [p[1], q[1]], [p[2], q[2]])
        elif cmd["type"] == "object":
            styles = {
                "fixture": {
                    "marker": "+",
                    "color": "k",
                },
                "joint": {
                    "marker": "o",
                    "color": "b",
                },
                "mass": {
                    "marker": "s",
                    "color": "r",
                },
            }
            marker, = obj(lambda: ax.plot([], [], [], **styles[cmd["class"]]))
            marker.set_data_3d([cmd["loc"][0]], [cmd["loc"][1]], [cmd["loc"][2]])
            marker.set_markersize(marker_size)

    return handle

def animate(sim, fps=30.0, dt=1e-2, T=1000.0):
    # Import the required matplotlib packages
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Import stuff for printing and keeping track of time
    import time, sys

    # Target FPS
    interval = 1.0 / fps

    # Visualise the initial state
    state = sim.initial_state()
    vis_handle = sim.visualize(state)

    # Define the "animate" function that needs to be passed to matplotlin
    t_tot = [0.0]
    def animate(i):
        # Run the actual simulation
        t0 = time.process_time()
        sim.run(interval, state, dt=dt)
        t1 = time.process_time()

        # Print some statistics about how long it takes to compute the
        # dynamics
        t_tot[0] += (t1 - t0)
        sys.stdout.write(
            "\rFrame time: {:0.1f}ms    \t Avg. step time: {:0.1f}µs    ".
            format((t1 - t0) * 1e3, t_tot[0] / state.step_count * 1e6))

        # Update the view
        sim.visualize(state, handle=vis_handle)

    # Run the animation
    ani = animation.FuncAnimation(vis_handle.fig,
                                  animate,
                                  range(int(T / interval)),
                                  interval=1000.0 * interval)

    # Show the window
    plt.show()
    print()

