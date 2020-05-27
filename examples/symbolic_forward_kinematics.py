#!/usr/bin/env python3

import sympy as sp
import pykinsim as pks

theta, l1, l2 = sp.symbols("\\theta \\ell_1 \\ell_2")

with pks.Chain() as chain:
    f1 = pks.Fixture()
    j1 = pks.Joint(axis="y", theta=theta)
    m1 = pks.Mass()
    m2 = pks.Mass()

    pks.Link(f1, j1, l=0.0)
    pks.Link(j1, m1, l=l1)
    pks.Link(m1, m2, l=l2)

with pks.Simulator(chain, f1) as sim:
    trafos = sim.forward_kinematics()

print("m1.x(t) =", trafos[m1][0, 3])
print("m1.z(t) =", trafos[m1][2, 3])
print("m2.x(t) =", trafos[m2][0, 3])
print("m2.z(t) =", trafos[m2][2, 3])

