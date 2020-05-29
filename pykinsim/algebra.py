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

import sympy as sp


def trans(x=0.0, y=0.0, z=0.0):
    return sp.Matrix([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rot_x(theta):
    s, c = sp.sin(theta), sp.cos(theta)
    return sp.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rot_y(theta):
    s, c = sp.sin(theta), sp.cos(theta)
    return sp.Matrix([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rot_z(theta):
    s, c = sp.sin(theta), sp.cos(theta)
    return sp.Matrix([
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def scalar(x):
    if isinstance(x, (int, float)):
        return float(x)
    elif isinstance(x, (sp.Symbol, sp.Function)):
        return x
    elif (isinstance(x, np.array) and (x.size == 1)):
        return float(x[(0, ) * x.ndim])
    else:
        raise ValueError("Expected scalar, but got {}".format(x))

