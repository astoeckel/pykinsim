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
    loc = (0, 1, 2)
    rot = (3, 4, 5)
    t0 = 1.0
    state = State(joint_idx_map, loc, rot, 1.0)

    assert len(state) == 3
    assert len(state[0]) == 2
    assert all(state[0] == 0.0)

    for i in range(3):
        state[i, 0] = i
        state[i, 1] = i + 1

    assert state[a][0] == 0
    assert state[a][1] == 1

    assert state[b][0] == 1
    assert state[b][1] == 2

    assert state[c][0] == 2
    assert state[c][1] == 3

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
        trafos = sim.forward_kinematics()

    assert len(trafs) == 1
    assert m1 in trafos
    assert np.all(trafos[m1] == np.eye(4))


def test_simulator_forward_kinematics_2():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()

    with pks.Simulator(chain, m1) as sim:
        trafos = sim.forward_kinematics()

    assert m1 in trafos
    assert np.all(trafos[m1] == np.eye(4))

