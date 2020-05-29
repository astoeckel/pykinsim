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

    tree = chain.tree(m1)
    assert len(tree) == 2
    assert m1 in tree
    assert len(tree[m1]) == 1
    assert tree[m1][0] is l1

    assert j in tree
    assert len(tree[j]) == 1
    assert tree[j][0] is l2

    tree = chain.tree(m2)
    assert len(tree) == 2
    assert m2 in tree
    assert len(tree[m2]) == 1
    assert tree[m2][0].direction == pks.Backward
    assert tree[m2][0].src is m2
    assert tree[m2][0].tar is j

    assert j in tree
    assert len(tree[j]) == 1
    assert tree[j][0].direction == pks.Backward
    assert tree[j][0].src is j
    assert tree[j][0].tar is m1

def test_tree_backward():
    with pks.Chain() as chain:
        m1 = pks.Mass()
        m2 = pks.Mass()
        j = pks.Joint("y")
        l1 = pks.Link(m1, j)
        l2 = pks.Link(j, m2)

    tree = chain.tree(m1, direction=pks.Backward)
    assert m2 in tree
    assert len(tree[m2]) == 1
    assert tree[m2][0].direction == pks.Backward
    assert tree[m2][0].src is m2
    assert tree[m2][0].tar is j

    assert j in tree
    assert len(tree[j]) == 1
    assert tree[j][0].direction == pks.Backward
    assert tree[j][0].src is j
    assert tree[j][0].tar is m1

    tree = chain.tree(m2, direction=pks.Backward)
    assert len(tree) == 2
    assert m1 in tree
    assert len(tree[m1]) == 1
    assert tree[m1][0] is l1

    assert j in tree
    assert len(tree[j]) == 1
    assert tree[j][0] is l2
