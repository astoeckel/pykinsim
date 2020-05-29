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

import numpy as np
import sympy as sp

import pytest
import pykinsim as pks

from pykinsim.simulator import State


def test_state():
    a, b, c = object(), object(), object()
    joint_idx_map = {
        a: 0,
        b: 1,
        c: 2,
    }
    state = State(joint_idx_map)

    assert len(state) == 3
    assert all(map(lambda x: x == 0.0, state))

    for i in range(3):
        state[i] = i

    assert state[a] == 0
    assert state[b] == 1
    assert state[c] == 2


def test_simulator_error_disconnected_components_1():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()

    with pytest.raises(pks.DisconnectedChainError):
        with pks.Simulator(chain, m1) as sim:
            pass

    with pytest.raises(pks.DisconnectedChainError):
        with pks.Simulator(chain, m2) as sim:
            pass


def test_simulator_error_disconnected_components_2():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()
        j = pks.Joint()
        pks.Link(m1, j)

    with pytest.raises(pks.DisconnectedChainError):
        with pks.Simulator(chain, m1) as sim:
            pass

    with pytest.raises(pks.DisconnectedChainError):
        with pks.Simulator(chain, m2) as sim:
            pass


def test_simulator_error_cyclic_1():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        l = pks.Link(m1, m1)

    with pytest.raises(pks.CyclicChainError):
        with pks.Simulator(chain, m1) as sim:
            pass


def test_simulator_error_cyclic_2():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()
        l = pks.Link(m1, m2)
        l = pks.Link(m2, m1)

    with pytest.raises(pks.CyclicChainError):
        with pks.Simulator(chain, m1) as sim:
            pass


def test_simulator_error_multiple_sources():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()
        m3 = pks.Mass()
        pks.Link(m1, m3)
        pks.Link(m2, m3)

    with pytest.raises(pks.MultipleSourcesError):
        with pks.Simulator(chain, m1) as sim:
            pass

    with pytest.raises(pks.MultipleSourcesError):
        with pks.Simulator(chain, m2) as sim:
            pass

    with pytest.raises(pks.MultipleSourcesError):
        with pks.Simulator(chain, m3) as sim:
            pass


def test_simulator_forward_kinematics_1():
    with pks.Chain() as chain:
        m1 = pks.Mass()

    with pks.Simulator(chain, m1) as sim:
        trafos = sim.kinematics(sim.initial_state())

    assert len(trafos) == 1
    assert m1 in trafos
    assert np.all(np.array(trafos[m1]) == np.eye(4))


def test_simulator_forward_kinematics_2():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()
        pks.Link(m1, m2)

    with pks.Simulator(chain, m1) as sim:
        trafos = sim.kinematics(sim.initial_state())

    assert len(trafos) == 2

    assert m1 in trafos
    assert np.all(np.array(trafos[m1]) == np.eye(4))

    assert m2 in trafos
    assert np.all(np.array(trafos[m2]) == (np.eye(4) + np.eye(4, 4, 3)))


def test_simulator_forward_kinematics_symbolic():
    l1, l2 = sp.symbols("l1 l2")

    with pks.Chain() as chain:
        f1 = pks.Fixture()
        j1 = pks.Joint()
        m1 = pks.Mass()
        m2 = pks.Mass()

        pks.Link(f1, j1, l=0.0)
        pks.Link(j1, m1, l=l1)
        pks.Link(m1, m2, l=l2)

    with pks.Simulator(chain, f1) as sim:
        trafos = sim.kinematics()

    theta = sim.symbols.theta[0]
    x, z = sim.symbols.x, sim.symbols.z

    assert sp.simplify(x + l1 * sp.cos(theta) - trafos[m1][0, 3]) == 0
    assert sp.simplify(z - l1 * sp.sin(theta) - trafos[m1][2, 3]) == 0
    assert sp.simplify(x + (l1 + l2) * sp.cos(theta) - trafos[m2][0, 3]) == 0
    assert sp.simplify(z - (l1 + l2) * sp.sin(theta) - trafos[m2][2, 3]) == 0


def test_simulator_dynamics_symbolic():
    with pks.Chain() as chain:
        f1 = pks.Fixture()
        j1 = pks.Joint()
        m1 = pks.Mass()

        pks.Link(f1, j1, l=0.0)
        pks.Link(j1, m1, l=1.0)

    with pks.Simulator(chain, f1) as sim:
        dynamics = sim.dynamics(g=(0.0, 0.0, 9.81))
        print(dynamics)

    theta = sim.symbols.var_theta[0]

    assert sim.symbols.var_ddtheta[0] in dynamics
    ddtheta =  dynamics[sim.symbols.var_ddtheta[0]]

    assert sp.simplify(9.81 * sp.cos(theta) - ddtheta) == 0

