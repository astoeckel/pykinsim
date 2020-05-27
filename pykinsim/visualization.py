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


def _visualize_matplotlib(raw, ax):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Fetch the figure and the axis object
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Iterate over the drawing commands and draw them
    for cmd in raw:
        if cmd["type"] == "link":
            ax.plot([cmd["src"][0], cmd["tar"][0]],
                    [cmd["src"][1], cmd["tar"][1]],
                    [cmd["src"][2], cmd["tar"][2]],
                    linestyle="-",
                    color="k")
        elif cmd["type"] == "axis":
            colors = {"x": "r", "y": "g", "z": "b"}
            ax.quiver(*cmd["src"],
                      *(cmd["tar"] - cmd["src"]),
                      linestyle="-",
                      color=colors[cmd["class"]],
                      length=0.25, normalize=True)
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
            ax.plot([cmd["loc"][0]], [cmd["loc"][1]], [cmd["loc"][2]],
                    **styles[cmd["class"]])

    return fig, ax

