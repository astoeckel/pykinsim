#   pykinsim -- Python Kinematics Simulator
#   Copyright (C) 2020  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.


class MatplotlibVisHandle:
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.objs = {}

def _visualize_matplotlib(raw, handle, ax):
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
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.grid(False)
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.view_init(elev=0.0, azim=-90.0)


        handle = MatplotlibVisHandle(fig, ax)
    else:
        fig, ax = handle.fig, handle.ax

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
            line.set_data_3d(
                [cmd["src"][0], cmd["tar"][0]],
                [cmd["src"][1], cmd["tar"][1]],
                [cmd["src"][2], cmd["tar"][2]]
            )
        elif cmd["type"] == "object":
            styles = {
                "fixture": {
                    "marker": "+",
                    "color": "k"
                },
                "joint": {
                    "marker": "o",
                    "color": "b"
                },
                "mass": {
                    "marker": "s",
                    "color": "r"
                },
            }
            marker, = obj(lambda: ax.plot([], [], [], **styles[cmd["class"]]))
            marker.set_data_3d([cmd["loc"][0]], [cmd["loc"][1]], [cmd["loc"][2]])

    return handle

