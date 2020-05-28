#!/usr/bin/env python3

import math
import pykinsim as pks

with pks.Chain() as chain:
    f1 = pks.Fixture(axes="xyz")
    j1 = pks.Joint(axis="x")
    j2 = pks.Joint(axis="y", theta=math.pi) # Fold the arm in on itself as an initial state
    m1 = pks.Mass(m=1.0)

    pks.Link(f1, j1, l=0.0, ry=-math.pi/2)
    pks.Link(j1, j2)
    pks.Link(j2, m1)

with pks.Simulator(chain, root=f1) as sim:
    pks.visualization.animate(sim)
