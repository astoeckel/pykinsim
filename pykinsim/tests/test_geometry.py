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

import pykinsim as pks
import numpy as np


def test_rot_to_euler():
    rng = np.random.RandomState(49991)
    # Iterate over all kinds of 90 degree angle
    for i in [None, -np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]:
        for j in [None, -np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]:
            for k in [None, -np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]:
                # Test some random angles for good measure
                for _ in range(5):
                    psi, theta, phi = rng.uniform(-np.pi, np.pi, 3)
                    if not i is None:
                        psi = i
                    if not j is None:
                        theta = j
                    if not k is None:
                        phi = k

                    rot = pks.geometry.euler_to_rot(psi, theta, phi)

                    psi2, theta2, phi2 = pks.geometry.rot_to_euler(rot[:3, :3])

                    rot2 = pks.geometry.euler_to_rot(psi2, theta2, phi2)

                    assert all(
                        np.abs(np.array(rot - rot2,
                                        dtype=np.float64)).flatten() < 1e-12)

                    if not ((i is None) or (j is None) or (k is None)):
                        break
