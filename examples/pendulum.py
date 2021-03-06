#!/usr/bin/env python3

import pykinsim as pks

with pks.Chain() as chain:
    f1 = pks.Fixture()
    j1 = pks.Joint()
    m1 = pks.Mass()

    pks.Link(f1, j1, l=0.0)  # Place the joint directly at the fixture
    pks.Link(j1, m1)

with pks.Simulator(chain, root=f1) as sim:
    pks.visualization.animate(sim)
