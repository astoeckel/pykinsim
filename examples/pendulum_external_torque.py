#!/usr/bin/env python3

import numpy as np
import pykinsim as pks

from pendulum_external_torque_utils import Ensemble, discretize_lti, make_delay_network

class AdaptiveController:
    """
    This is a fancy Adaptive controller (essentially a PID controller with a
    non-linear I term that nonlinearly depends on the trajectory of the
    end-effector over the last second).

    Note: In case you are just looking for an API example, you can ignore all
          the math here and just look at the lines marked with "<--"
    """
    def __init__(self, dt, theta_tar=-np.pi/2, q=3, theta=1.0, n_neurons=100, eta=0.5):
        self.theta_tar = theta_tar
        self.eta = eta
        self.last_P = None

        # Build a delay network
        self.A, self.B = discretize_lti(dt, *make_delay_network(q, theta))
        self.x = np.zeros(q)

        # Build a layer/ensemble of neurons representing the delay network state
        self.ens = Ensemble(n_neurons, q)

        # Decoding weights
        self.w = np.zeros(n_neurons)

    def __call__(self, state, dt):
        # Fetch the joint angle <-- READ THE JOINT ANGLE HERE
        theta = state[j1]

        # Update the context vector "x" and compute the neural activities "A"
        self.x = self.A @ self.x + (self.B * theta)[:, 0]
        A = self.ens(self.x.reshape(-1, 1))[:, 0]

        # Compute the proportional error
        P = self.theta_tar - theta

        # Compute the differential term, i.e., how much the error changed
        D = (P - (P if self.last_P is None else self.last_P)) / dt
        self.last_P = P

        # Output of the PD controller
        E = P + 2.0 * D

        # Update the neuron decoding weights (this is essentially a fancy I term)
        self.w += dt * self.eta * E * A

        # Compute the torque
        tau = E + A @ self.w

        # Store the torque in the state <-- SET THE EXTERNAL TORQUE HERE
        state.torques[j1] = tau

#
# Actual model
#

with pks.Chain() as chain:
    f1 = pks.Fixture()
    m1 = pks.Mass()

    # Create a joint with an external torque
    j1 = pks.Joint(torque=pks.External)

    pks.Link(f1, j1, l=0.0)  # Place the joint directly at the fixture
    pks.Link(j1, m1)

with pks.Simulator(chain, root=f1) as sim:
    dt = 1e-2
    pks.visualization.animate(sim, dt=dt, callback=AdaptiveController(dt))

#    # In case you're running your own simulation loop, the code would be
#    state = sim.initial_state()
#    controller = AdaptiveController(dt)
#    while state.t < 20.0:
#        controller(state, dt)
#        state = sim.step(state, dt)
#        if np.abs(state.t % 1.0 - dt) < dt:
#            sys.stdout.write("\rt={:0.1f}, theta={:0.2f}".format(state.t, state[j1]))
#            sys.stdout.flush()
#    sys.stdout.write("\n")
