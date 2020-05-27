#!/usr/bin/env python3

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pykinsim as pks

l = sp.Symbol("\\ell")

with pks.Chain() as chain:
#    m1 = pks.Mass()
    f1 = pks.Fixture()
    j1 = pks.Joint(axis="y", theta=0.0)
    m1 = pks.Mass()
    j2 = pks.Joint(axis="y", theta=0.0)
    m2 = pks.Mass()

    pks.Link(f1, j1, l=0.0) # Place the joint directly at the fixture
    pks.Link(j1, m1)
    pks.Link(m1, j2)
    pks.Link(j2, m2)

with pks.Simulator(chain, root=f1) as sim:
    sim.dynamics(debug=True, g=[0, 0, 9.81])

#ax.set_xlim(-1, 1)
#ax.set_ylim(-1, 1)
#ax.set_zlim(-1, 1)
#ax.set_xlabel("$x$")
#ax.set_ylabel("$y$")
#ax.set_zlabel("$z$")

#plt.show()


