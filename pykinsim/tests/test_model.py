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

import pykinsim as pks

def test_variable_name_deduction():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()
        j = pks.Joint("y")
        l1 = pks.Link(m1, j)
        l2 = pks.Link(j, m2)

    assert chain.label == "chain"
    assert m1.label == "m1"
    assert m2.label == "m2"
    assert j.label ==  "j"
    assert l1.label == "l1"
    assert l2.label == "l2"


def test_tree_forward():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()
        j = pks.Joint("y")
        l1 = pks.Link(m1, j)
        l2 = pks.Link(j, m2)

    tree, root = chain.tree()
    assert root is m1
    assert len(tree) == 2
    assert m1 in tree
    assert len(tree[m1]) == 1
    assert tree[m1][0] is l1
