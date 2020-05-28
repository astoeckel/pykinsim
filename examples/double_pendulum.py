#!/usr/bin/env python3

import pykinsim as pks

with pks.Chain() as chain:
    f1 = pks.Fixture()
    j1 = pks.Joint()
    m1 = pks.Mass()
    j2 = pks.Joint()
    m2 = pks.Mass()

    pks.Link(f1, j1, l=0.0)
    pks.Link(j1, m1, l=0.33)
    pks.Link(m1, j2, l=0.33)
    pks.Link(j2, m2, l=0.33)

with pks.Simulator(chain, root=f1) as sim:
    pks.visualization.animate(sim)
