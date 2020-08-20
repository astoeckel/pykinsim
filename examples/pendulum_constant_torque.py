#!/usr/bin/env python3

import pykinsim as pks

with pks.Chain() as chain:
    f1 = pks.Fixture()
    m1 = pks.Mass()

    # Create a joint with a constant external torque of -9.81kg m²/s². This is
    # countering the torque produced by gravity
    j1 = pks.Joint(torque=-9.81)

    pks.Link(f1, j1, l=0.0)  # Place the joint directly at the fixture
    pks.Link(j1, m1)

with pks.Simulator(chain, root=f1) as sim:
    pks.visualization.animate(sim)
