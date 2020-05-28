#!/usr/bin/env python3

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pykinsim as pks
import time, sys

import logging
logging.basicConfig(level=logging.DEBUG)

l = sp.Symbol("\\ell")

with pks.Chain() as chain:
    f1 = pks.Fixture(axes="yz")
    m1 = pks.Mass(m=2.0)
    j1 = pks.Joint(axis="y", theta=0.0)
    m2 = pks.Mass(m=1.0)
    j2 = pks.Joint(axis="y")
    m3 = pks.Mass(m=1.0)

    pks.Link(m1, j1, l=0.5)
    pks.Link(j1, f1, l=0.0)  # Place the joint directly at the fixture
    pks.Link(j1, m2, l=0.33)
    pks.Link(m2, j2, l=0.33)
    pks.Link(j2, m3, l=0.33)

with pks.Simulator(chain, root=j1) as sim:
    # Visualise the initial state
    state = sim.initial_state()

    vis_handle = sim.visualize(state)

    interval = 1.0 / 30.0
    t_tot = [0.0]

    def animate(i):
        t0 = time.process_time()
        sim.run(interval, state, dt=1e-2)
        t1 = time.process_time()
        t_tot[0] += (t1 - t0)
        sys.stdout.write(
            "\rFrame time: {:0.1f}ms    \t Avg. step time: {:0.1f}µs    ".
            format((t1 - t0) * 1e3, t_tot[0] / state.step_count * 1e6))
        sim.visualize(state, handle=vis_handle)

    ani = animation.FuncAnimation(vis_handle.fig,
                                  animate,
                                  range(int(100.0 / interval)),
                                  interval=1000.0 * interval)

    plt.show()
    print()

