#!/usr/bin/env python3

import math
import pykinsim as pks

with pks.Chain() as chain:
    m1 = pks.Mass(m=2.0)
    f1 = pks.Fixture(axes="yz") # Allow motion in the x-direction
    j1 = pks.Joint()
    m2 = pks.Mass()
    j2 = pks.Joint()
    m3 = pks.Mass()

    pks.Link(j1, f1, l=0.0)
    pks.Link(j1, m1, l=0.5, ry=math.pi)
    pks.Link(j1, m2, l=0.33)
    pks.Link(m2, j2, l=0.33)
    pks.Link(j2, m3, l=0.33)

with pks.Simulator(chain, root=f1) as sim:
    pks.visualization.animate(sim)
