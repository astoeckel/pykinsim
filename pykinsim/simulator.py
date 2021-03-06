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

from .errors import *
from .model import *
from .utils import *
from .geometry import *

import numpy as np
import sympy as sp

import logging
logger = logging.getLogger(__name__)

# Default gravity vector
DEFAULT_GRAVITY = (0.0, 0.0, 9.81)

class Symbols:
    """
    Class collecting all the symbols used to represent the kinematics and
    dynamics of the system.
    """
    def __init__(self, n_joints, n_cstrs, n_torques):
        # Handy aliases
        n, m = n_joints, n_cstrs

        # Variable representing time
        self.t = t = sp.Symbol("t")

        # Variables representing the global location of the system relative to the
        # origin
        self.x = sp.Function("x")(t)
        self.y = sp.Function("y")(t)
        self.z = sp.Function("z")(t)
        self.origin = sp.Matrix([self.x, self.y, self.z])

        # Variables representing the global velocity of the system
        self.dx = sp.diff(self.x, t)
        self.dy = sp.diff(self.y, t)
        self.dz = sp.diff(self.z, t)
        self.dorigin = [self.dx, self.dy, self.dz]

        # Variables representing the global acceleration of the system
        self.ddx = sp.diff(self.dx, t)
        self.ddy = sp.diff(self.dy, t)
        self.ddz = sp.diff(self.dz, t)
        self.ddorigin = [self.ddx, self.ddy, self.ddz]

        # Variables representing the individual joint rotations
        self.theta = sp.zeros(n_joints, 1)
        self.dtheta = sp.zeros(n_joints, 1)
        self.ddtheta = sp.zeros(n_joints, 1)
        for i in range(n):
            name = "θ{{{}}}".format(i + 1) if n_joints > 1 else "θ"
            self.theta[i] = sp.Function(name)(t)
            self.dtheta[i] = sp.diff(self.theta[i], t)
            self.ddtheta[i] = sp.diff(self.dtheta[i], t)

        # Lagrange multipliers
        self.lambda_ = [None] * n_cstrs
        for i in range(m):
            self.lambda_[i] = sp.Symbol("λ{}".format(i + 1))

        # Torques
        self.torques = [None] * n_torques
        for i in range(n_torques):
            self.torques[i] = sp.Symbol("τ{}".format(i + 1))

        # Gravity vector
        self.gx, self.gy, self.gz = sp.symbols("gx gy gz")
        self.g = sp.Matrix([self.gx, self.gy, self.gz])

        # Free-standing symbols substituted for the functions above
        self.var_origin = list(sp.symbols("x y z"))
        self.var_dorigin = list(sp.symbols("dx dy dz"))
        self.var_ddorigin = list(sp.symbols("ddx ddy ddz"))
        self.var_theta = [sp.Symbol("θ{}".format(i)) for i in range(n)]
        self.var_dtheta = [sp.Symbol("dθ{}".format(i)) for i in range(n)]
        self.var_ddtheta = [sp.Symbol("ddθ{}".format(i)) for i in range(n)]

        # Substitution table mapping the functions over time defined above onto
        # the free symbols below
        self._subs = []
        for i in range(3):
            self._subs.append((self.ddorigin[i], self.var_ddorigin[i]))
            self._subs.append((self.dorigin[i], self.var_dorigin[i]))
            self._subs.append((self.origin[i], self.var_origin[i]))
        for i in range(n):
            self._subs.append((self.ddtheta[i], self.var_ddtheta[i]))
            self._subs.append((self.dtheta[i], self.var_dtheta[i]))
            self._subs.append((self.theta[i], self.var_theta[i]))

    def subs(self, expr):
        """
        Substitutes functions over time, such as x(t) with free-standing
        expressions.
        """
        for key, value in self._subs:
            expr = expr.subs(key, value)
        return expr


class ExternalTorques:
    """
    Class used internally by "State" to provide a map for writing external
    torques.
    """

    def __init__(self, external_torques, joint_torque_idx_map):
        self._external_torques = external_torques
        self._joint_torque_idx_map = joint_torque_idx_map

        # Determine the total number of joints with an external torque
        if len(joint_torque_idx_map) == 0:
            self._n_torques = 0
        else:
            self._n_torques = max(joint_torque_idx_map.values()) + 1

    def idx(self, key):
        return self._joint_torque_idx_map[key]

    def _check_key(self, key):
        if isinstance(key, (int, slice)) or hasattr(key, '__len__'):
            return key
        else:
            return self.idx(key)

    def __getitem__(self, key):
        return self._external_torques[self._check_key(key)]

    def __setitem__(self, key, value):
        self._external_torques[self._check_key(key)] = value

    def __len__(self):
        return self._n_torques



class State:
    """
    The State class describes the state of the model. In particular, this class
    tracks the state (i.e., angle) of each joint, as well as the dynamics (i.e.,
    angular velocity).

    Do not manually construct State instances, use the Simulator.initial_state()
    member function instead.
    """
    def __init__(self, joint_idx_map, joint_torque_idx_map, symbolic=False):
        # Copy the given arguments
        self._joint_idx_map = joint_idx_map
        self._joint_torque_idx_map = joint_torque_idx_map

        # Determine the total number of joints
        if len(joint_idx_map) == 0:
            self._n_joints = 0
        else:
            self._n_joints = max(joint_idx_map.values()) + 1

        # Determine the total number of joints with an external torque
        if len(joint_torque_idx_map) == 0:
            self._n_torques = 0
        else:
            self._n_torques = max(joint_torque_idx_map.values()) + 1

        # State variables keeping track of the current time and the current step
        self._t = 0.0
        self._step_count = 0

        # Either use numpy or sympy arrays
        if symbolic:
            zeros = lambda l: sp.zeros(l, 1)
        else:
            zeros = lambda l: np.zeros(l)

        # State variables keeping track of the location and rotation of the root
        # element
        self._loc = zeros(3)
        self._dloc = zeros(3)
        self._rot = zeros(3)

        # Create arrays holding the joint states and velocities
        self._joint_states = zeros(self._n_joints)
        self._joint_velocities = zeros(self._n_joints)

        # Create array holding the external torques applied to the joints
        self._external_torques = zeros(self._n_torques)
        self._external_torques_obj = ExternalTorques(self._external_torques, self._joint_torque_idx_map)

    @property
    def t(self):
        return self._t

    @property
    def step_count(self):
        return self._step_count

    @property
    def loc(self):
        return self._loc

    @property
    def dloc(self):
        return self._dloc

    @property
    def rot(self):
        return self._rot

    def __len__(self):
        return self._n_joints

    def idx(self, key):
        return self._joint_idx_map[key]

    @property
    def states(self):
        return self._joint_states

    def _check_key(self, key):
        if isinstance(key, (int, slice)) or hasattr(key, '__len__'):
            return key
        else:
            return self.idx(key)

    def __getitem__(self, key):
        return self._joint_states[self._check_key(key)]

    def __setitem__(self, key, value):
        self._joint_states[self._check_key(key)] = value

    def __iter__(self):
        return iter(self._joint_states)

    @property
    def velocities(self):
        return self._joint_velocities

    @property
    def torques(self):
        return self._external_torques_obj

    @property
    def vec(self):
        """
        Returns a vectorised version of the state, including both the static
        state and the velocities. This vector is passed to the dynamics
        function returned by "Simulator.dynamics()", as well as
        "Simulator.state_differential()".
        """
        return np.concatenate((self._loc, self._joint_states, self._dloc,
                               self._joint_velocities, self._external_torques))

    @property
    def static_vec(self):
        """
        Returns a vectorised version of the static state, i.e., only the
        origin and the joint states. This does *not* include the velocities or
        torques.
        The resulting vector is passed to the kinematics function returned
        by "Simulator.kinematics()".
        """
        return np.concatenate((self._loc, self._joint_states))

    @vec.setter
    def vec(self, x):
        n, m = self._n_joints, self._n_torques
        i0, i1, i2, i3, i4, i5 = 0, 3, 3 + n, 6 + n, 6 + 2 * n, 6 + 2 * n + m
        self._loc[...] = x[i0:i1]
        self._joint_states[...] = x[i1:i2]
        self._dloc[...] = x[i2:i3]
        self._joint_velocities[...] = x[i3:i4]
        self._external_torques[...] = x[i4:i5]


class Simulator:
    """
    The Simulator class is capable of simulating a kinematics chain. It uses
    sympy to algebraically solve for the dynamics of the kinematics chain and
    provides functions that facilitate the physical simulation of the
    kinematics chain.
    """
    def __init__(self,
                 chain,
                 root=None,
                 loc=None,
                 rot=None,
                 g=(0.0, 0.0, 9.81)):
        """
        Creates a new Simulator object for the given chain. The "root" object
        is the object relative to which the location of all other objects is
        computed. If no root is given, a suitable root will automatically be
        selected.

        As a first step, the Simulator object will validate all parameters set
        on the kinematics objects; i.e., it will ensure that all objects have
        the right type, that there is only one connected component, and that
        the link structure is acyclic.

        Note
        ====

        The `chain` object will be mutated by the `Simulator` constructor.
        Once the simulator object has been created, changes to the original
        `chain` object are likely to not affect the simulation. A good practice
        is to construct a single `Chain` instance for each `Simulator` instance,
        and to not change the `Chain` once the `Simulator` has been created.

        Parameters
        ==========

        chain: The Chain instance for which the simulator should be created.
        root:  The Mass or Joint object relative to which the location of all
               other objects is reported. If None, automatically determines the
               root object by chosing the root node of the kinematics tree.
        loc:   An optional three-tuple describing the location of the root
               object within the world coordinate space. If not specified, the
               root object is located at (0, 0, 0).
        rot:   An optional three-tuple describing the rotation of the root
               element in the world coordinate space.
        g:     Gravity vector to use in the simulation. The default value
               roughly corresponds to earth's gravity near earth's surface.
               Per default, the gravity vector is aligned with the z-axis.
        """

        # Make sure that "chain" has the right type
        if not isinstance(chain, Chain):
            raise ValidationError("\"chain\" must be an instance of \"Chain\"",
                                  self)

        # Make sure "root" is either None or has the right type
        if not root is None:
            if not isinstance(root, PointObject):
                raise ValidationError(
                    "\"root\" must be a \"PointObject\" instance", self)

            # Make sure the "root" object is actually part of the Chain instance
            if not root in chain:
                raise ValidationError(
                    "The given \"root\" object must be part of the chain",
                    self)

        # Make sure loc and rot are correct
        if loc is None:
            loc = np.zeros(3)
        if rot is None:
            rot = np.zeros(3)
        assert len(loc) == 3
        assert len(rot) == 3
        assert len(g) == 3

        # Copy the gravity vector
        self._g = g

        # Copy and validate the "chain" object
        self.chain = chain.coerce()
        self.root = root

        # Build the _joint_idx_map assigning an index to each joint; furthermore
        # assign an index to each joint with an external torque
        self._joint_idx_map = {}
        self._joint_torque_idx_map = {}
        i, ti = 0, 0
        for joint in self.joints:
            # Add the joint to the joint to index map
            self._joint_idx_map[joint] = i
            i = i + 1

            # If the joint has an external joint, add it to the joint torque
            # index map
            if joint.torque is External:
                self._joint_torque_idx_map[joint] = ti
                ti = ti + 1

        # Count the number of joints, constraints, and torques
        self._n_joints = sum(1 for _ in self.joints)
        self._n_cstrs = sum(len(fix.axes) for fix in self.fixtures)
        self._n_torques = ti

        # Construct the symbols used for evaluating certain expressions
        # symbolically
        self._symbols = Symbols(self._n_joints, self._n_cstrs, self._n_torques)

        # Build the spanning tree
        self._tree, self._tree_root = self.chain.tree()

        # If a root element has been specified, setup the rotation and offset
        # in such a way that the local coordinate system of the root element
        # is aligned with the global coordinate stytem
        self._loc = np.zeros(3)
        self._rot = np.zeros(3)
        if self.root is None:
            self.root = self._tree_root
        else:
            # Compute the forward kinematics for that root element
            T = self.kinematics(state=self.initial_state())[self.root]

            # Translate the coordinate system such that the root element is
            # in the center
            self._loc = np.array(-T[:3, 3]).flatten()

            # Rotate the coordinate system such that the local coordinate system
            # if the root element is aligned with the coordinate system of the
            # global coordinate frame.
            psi, theta, phi = rot_to_euler(T[:3, :3])
            self._rot = (-psi, -theta, -phi)

        # Apply the user-specified transformation
        self._loc += loc
        self._rot += rot

        try:
            # Compile the kinematics into a function
            self._kinematics = self.kinematics(return_function=True)

            # Compile the dynamics into a function (aka the dynamics descriptor)
            self._dynamics = self.dynamics(return_function=True)
        except TypeError:
            self._kinematics = None
            self._dynamics = None
            logger.warn("Could not compile system for numerical simulation")

    @property
    def loc(self):
        return self._loc

    @property
    def rot(self):
        return self._rot

    @property
    def g(self):
        return self._g

    @property
    def joints(self):
        return filter(lambda x: isinstance(x, Joint), self.chain._objs)

    @property
    def joints_with_external_torque(self):
        return filter(lambda x: isinstance(x, Joint), self.chain._objs)

    @property
    def links(self):
        return filter(lambda x: isinstance(x, Link), self.chain._objs)

    @property
    def masses(self):
        return filter(lambda x: isinstance(x, Mass), self.chain._objs)

    @property
    def fixtures(self):
        return filter(lambda x: isinstance(x, Fixture), self.chain._objs)

    @property
    def point_objects(self):
        return filter(lambda x: isinstance(x, PointObject), self.chain._objs)

    @property
    def symbols(self):
        return self._symbols

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def initial_state(self):
        """
        Returns a new State object describing the initial state of the model.
        """

        # Create the state object
        state = State(self._joint_idx_map, self._joint_torque_idx_map)

        # Copy the initial joint angles
        for joint in self.joints:
            state[joint] = joint.theta

        # Copy the initial location and rotation of the system
        for i in range(3):
            state.loc[i] = self._loc[i]
            state.rot[i] = self._rot[i]

        return state

    def symbolic_state(self):
        """
        Returns a new "symbolic" State object in which each state variable is
        replaced with a sympy symbol. This can either be used to write out the
        dynamics of the system.
        """

        # Create the state object
        state = State(self._joint_idx_map, self._joint_torque_idx_map, symbolic=True)

        # Set the initial joint angles
        for i in range(len(state)):
            state.states[i] = self.symbols.theta[i]
            state.velocities[i] = self.symbols.dtheta[i]

        # Use the symbolic x, y, z variables
        for i in range(3):
            state.loc[i] = self.symbols.origin[i]

        # Use the non-symbolic rotation
        for i in range(3):
            state.rot[i] = float_to_pi(self._rot[i])

        return state

    def run(self, T, state=None, dt=1e-2, callback=None, callback_interval=None):
        """
        Advances the simulation up to T in timesteps smaller or equal to dt.

        Parameters
        ==========

        T:      The time in seconds to run the simulation for.
        state:  The state object containing the current joint angles and
                velocities, as well as user-defined external torques. If no
                state object is given (i.e., this is set to None), creates a new
                State object. Note that the State object must not be symbolic.
        dt:     The maximum timestep used in the evaluation. This value must be
                strictly positive.
        callback: If given, is called every "callback_interval" seconds of
                simulation time. The function receives two parameters: the
                current time "t", as well as the current state. The function may
                return the boolean False to abort the simulation early.
        callback_interval: The interval in seconds within which the callback
                function is called. If "None", this is set to dt, i.e., the
                callback is called every timestep. Note that the times at which
                the callback is called are aligned with the global simulation
                time (i.e., if callback_interval is 1s, the callback will be
                called at state.t = 0.0, state.t = 1.0, state.t = 2.0, etc.).
                This value must be greater than or equal to dt.
        """
        # Use the initial state if no state is given
        if state is None:
            state = self.initial_state()

        # Make sure "dt" has a sane value to prevent infinite loops.
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")

        # Make sure "callback_interval" has a sane value. Set to infinity if
        # there is no callback.
        if callback_interval is None:
            callback_interval = dt if callback else np.inf
        if callback_interval < dt:
            raise ValueError("callback_interval must be greater than or equal to the timestep dt.")

        # Compute the time at which the next callback should be called
        if np.isfinite(callback_interval):
            next_callback = (state.t % callback_interval)
            if abs(next_callback) > 1e-12:  # Call cbck if due in first step
                next_callback = callback_interval - next_callback
        else:
            next_callback = np.inf # Do not call callback

        # Loop until the end of the simulation period is reached
        while T > 0:
            # Periodically call the callback
            if next_callback <= 0.0:
                if callback(state, dt) is False:
                    return state
                next_callback += callback_interval

            # Step exactly to T, or to the next callback interval
            cur_dt = min(T, dt, next_callback)

            # Perform a single dynamics step
            state = self.step(state, cur_dt)

            # Keep track of passing time
            T -= cur_dt
            next_callback -= cur_dt

        return state

    def step(self, state=None, dt=1e-2):
        """
        Advances the simulation by one timestep from the given state using a
        fourth-order Runge-Kutta integrator. If no state is given, a new State()
        object corresponding to the initial state is generated.

        Parameters
        ==========

        state:  State instance that should be advanced.
        dt:     Time-step in seconds.

        Return
        ======
        """

        # Fetch the initial state
        if state is None:
            state = self.initial_state()

        # Fetch the dynamics descriptor and state vector
        dyns, x = self._dynamics, state.vec

        # Perform a single Runge-Kutta 4 (RK4) step
        k1 = self.state_differential(dyns, x)
        k2 = self.state_differential(dyns, x + dt * k1 / 2.0)
        k3 = self.state_differential(dyns, x + dt * k2 / 2.0)
        k4 = self.state_differential(dyns, x + dt * k3)
        state.vec = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        state._t += dt
        state._step_count += 1

        return state

    def kinematics(self,
                   state=None,
                   return_function=False):
        """
        Computes the location and orientation of either all point objects or
        the specified point object. If no state (or a symbolic state) is given,
        computes the forward kinematics symbolically, i.e., as mathematical
        equations depending on  the joint angles.

        Setting the "return_function" parameter to True causes "kinematics" to
        return a function that maps the a static state vector (as stored in
        State.static_vec) onto an object map such as the one usually returned
        by this function. The benefit is that the returned function no longer
        directly makes use of sympy and is thus much faster.


        Parameters
        ==========

        state:  The state used to compute the forward kinematics. This may
                either by a numerical, or a symbolic state instance (as returned
                by symbolic_state() or initial_state().

        return_function:
                If True, returns a function that maps a state vector as returned
                by "State.static_vec" onto the transformation matrices for all
                objects. Using this function is much faster than calling
                "kinematics" with a numerical state instance.

                Note that thestate argument "state" must be set to None if
                "return_function" is set to True.
        """

        if (not state is None) and return_function:
            raise ValueError(
                "A \"state\" may only be given if \"return_function\" is False"
            )

        if state is None:
            state = self.symbolic_state()

        # Compute the root object transformation
        loc, rot = state.loc, state.rot
        trafo = euler_to_rot(*rot) @ trans(*loc)

        # Perform a breadth-first search on the tree to compute all
        # transformations. In case we're returning a function, keep track of
        # the parent node of each object.
        T = {self._tree_root: (None, trafo) if return_function else trafo}
        queue = deque((self._tree_root, ))
        while len(queue) > 0:
            # Fetch the current node
            cur = queue.popleft()

            # Nothing else to do if thise node is a leaf node
            if not cur in self._tree:
                continue

            # Fetch the transformation up to the current point; if we're
            # returning a function, only compute the element-wise transformation
            if return_function:
                trafo = np.eye(4)
            else:
                trafo = T[cur]

            # If the current source object is a joint, account for the joint
            # transformation
            if isinstance(cur, Joint):
                theta = state.states[state.idx(cur)]
                trafo = trafo @ cur.trafo(theta=theta)

            # Iterate over all link targets
            for link in self._tree[cur]:
                assert (cur is link.src)
                ltrafo = trafo @ link.trafo()
                T[link.tar] = (cur, ltrafo) if return_function else ltrafo
                queue.append(link.tar)

        # We're done here if we were to solve this symbolically
        if not return_function:
            return T

        # Otherwise convert each individual transformation into a function;
        # also, produce a set of leaf nodes, i.e., objects that are never
        # referenced as a parent
        S = self.symbols  # Alias
        args = S.var_origin + S.var_theta  # Arguments passed to the functions
        leafs = set(self.point_objects)
        for key, (parent, value) in T.items():
            # The parent cannot be a leaf
            if parent in leafs:
                leafs.remove(parent)

            # Substitute the functions over time with variables and simplify
            # the resulting expression
            expr = sp.simplify(S.subs(value))

            # Convert the expression into a lambda and append it
            T[key] = (parent, sp.lambdify(args, expr, modules=["numpy"]))

        # Define the actual function we're returning here
        def evaluate_kinematics(*args):

            # Recursion base case -- map "None" onto the identity transformation
            res = {}

            # Recursive evaluation of the object location
            def eval(node):
                # Fetch the parent of the given leaf, as well as the
                # transformation function
                parent, f_trafo = T[node]

                # Abort if this is the root node
                if parent is None:
                    res[node] = f_trafo(*args)
                    return

                # Recurse upwards
                if not parent in res:
                    eval(parent)

                # Store the transformation of the specific child in the result
                res[node] = res[parent] @ f_trafo(*args)

            for leaf in leafs:
                eval(leaf)

            return res

        return evaluate_kinematics

    def lagrangian(self, g=None):
        """
        Symbolically computes the Lagrangian L = KE - PE for the given gravity
        vector (or a generic gravity vector if "g" is set to None).

        Parameters
        ==========

        g:      Gravity vector for which the Lagrangian should be computed. If
                set to None, uses the symbolic vector [g_x, g_y, g_z].
        """

        # Use the symbolic gravity vector if None is given
        if g is None:
            g = self.symbols.g

        # Symbolically compute the forward kinematics
        trafos = self.kinematics()

        # Fetch the symbol representing time
        t = self.symbols.t

        # Compute the kinetic and potential energy terms
        KE, PE = 0, 0
        for mass in self.masses:
            # Fetch the location of the mass in cartesian coordinates
            x, y, z = trafos[mass][:3, 3]

            # Compute the relative velocities
            dx, dy, dz = map(lambda e: sp.diff(e, t), trafos[mass][:3, 3])

            # The kinetic energy is just 1/2 m v^2
            KE += 0.5 * mass.m * (dx**2 + dy**2 + dz**2)

            # The potential energy is F=m<g, x>
            PE += mass.m * (x * g[0] + y * g[1] + z * g[2])

        # Return the Lagrangian, L = KE - PE
        return KE - PE

    def constraints(self):
        """
        Converts the fixture objects into a set of constraint equations.
        """

        # Symbolically compute the forward kinematics
        trafos = self.kinematics()

        # Compute the forward kinematics for the initial state
        trafos_init = self.kinematics(state=self.initial_state())

        # For each fixture, add a set of constraints regarding its location
        Cs = []
        idx = 0
        for fixture in self.fixtures:
            for i, axis in enumerate(["x", "y", "z"]):
                if not axis in fixture.axes:
                    continue
                Cs.append(trafos[fixture][i, 3] - trafos_init[fixture][i, 3])
                idx += 1

        return Cs

    def dynamics(self, g=None, return_function=False):
        """
        Returns a map of equations describing the dynamics of the system.

        Parameters
        ==========

        g:        The gravity vector. If None is given, uses the value for "g"
                  passed to the constructor of the simulator object.
        return_function:
                  If True, returns the symbolic equations describing the
                  dynamics of each state variable. Otherwise returns a tuple of
                  functions that can be used to build a linear system of
                  equations implicitly describing the dynamics. These functions
                  can be passed to the "Simulator.state_differential" function.
        """

        # Load the default values for some values
        if g is None:
            g = self._g

        # Shorthand for df/dt, d²/d²t and df/dx
        Dt = lambda f: sp.diff(f, self.symbols.t)
        Dt2 = lambda f: sp.diff(f, (self.symbols.t, 2))
        Dx = lambda f, x: sp.diff(f, x)

        S = self.symbols

        # Compute the Lagrangian for an arbitrary gravity vector
        L = self.lagrangian(g=g)

        # Compute the constraints
        Cs = self.constraints()
        C = sum([S.lambda_[i] * C for i, C in enumerate(Cs)])

        # Collect all state variables; these are the variables we're
        # differentiating over when computing the Lagrange equations
        vars_ = [*S.origin, *S.theta]

        # Collect all torques
        torques = [0, 0, 0]
        for joint in self.joints:
            if joint.torque is External:
                torques.append(S.torques[self._joint_torque_idx_map[joint]])
            else:
                torques.append(joint.torque)

        # Collect all target variables; these are the variables we're trying
        # to solve for
        tars = [S.subs(x) for x in ([Dt2(var) for var in vars_] + S.lambda_)]
        tars_idcs = {x: i for i, x in enumerate(tars)}
        tars_keys = set(tars)

        # Compute the set of Lagrange equations. Add the constraints, as well as
        # the second order derivative of the constraint. Each entry "x" in the
        # "eqns" list corresponds to an equation of the form "x = 0".
        eqns = [
            sp.expand(S.subs(x)) for x in
            ([Dx(L, var) - Dt(Dx(L, Dt(var))) + Dx(C, var) + torques[i]
              for i, var in enumerate(vars_)] + Cs + [Dt2(C) for C in Cs])
        ]

        # The resulting system is linear in the accelerations and Lagrange
        # multipliers (but not in the state variables). Hence, we arrange the
        # Lagrange equations in a linear equation of the form Ax = b,
        # where A is a (m x n) matrix (m is the number of equations, n is the
        # number of target variables).
        m, n = len(eqns), len(tars)

        # Fill the A and b matrices.
        A, b = sp.zeros(m, n), sp.zeros(m, 1)
        for i, eqn in enumerate(eqns):
            # Split each equation into the underlying additive terms
            terms = eqn.args if isinstance(eqn, sp.Add) else [eqn]
            for j, term in enumerate(terms):
                # Discard terms containing incredibly small factors
                if (isinstance(term, sp.Mul)
                        and isinstance(term.args[0], sp.Number)
                        and (abs(term.args[0]) < 1e-12)):
                    logger.debug("Discarding term \"%s\"", term)
                    continue
                # Find the target variables contained within this term
                tar = term.free_symbols & tars_keys
                if len(tar) == 0:
                    # No target variable in the term, subtract the term from
                    # the right-hand side of the linear system
                    b[i] -= term
                elif len(tar) == 1:
                    # There is exactly one target variable in the term. Add this
                    # term to the correct spot in A and substitute it with one.
                    tar = next(iter(tar))
                    A[i, tars_idcs[tar]] += term.subs(tar, 1)
                else:
                    # Ugh. We have a problem. Multiple target variables are in
                    # this (multiplicative) term. This means that the system is
                    # not solveable as a linear system of equations.
                    # TODO: Can this happen at all?
                    raise ValidationError(
                        "Resulting system is not linear in the accelerations and Lagrange multipliers!",
                        self)

        # Only use equations with at least one non-zero column in the A matrix
        keep_row_idcs, subs = [], [None] * n
        for i in range(m):
            # Compute the non-zero column indices
            nz_col_idcs = list(filter(lambda j: A[i, j] != 0.0, range(n)))
            if len(nz_col_idcs) > 1:
                # If there is more than one non-zero column index, there's
                # nothing we can do for now to simplify the system any further
                keep_row_idcs.append(i)
                continue
            elif len(nz_col_idcs) == 0:
                # If there are no non-zero entries in this row, just eliminate
                # the entire row
                logger.debug(
                    "Eliminating equation \"%s = 0\" from the system.", b[i])
                continue

            # There is exactly one non-zero column j in this row. Thus, we know
            # that the value of the corresponding variable is b[i] / A[i, j].
            # We can thus elminate this column from the system (in a separate
            # step below).
            j = nz_col_idcs[0]
            subs[j] = b[i] / A[i, j]
            logger.debug("Substituting \"%s\" with \"%s\"", tars[j], subs[j])

        # Eliminate rows from A and b
        A = A[keep_row_idcs, :]
        b = b[keep_row_idcs, :]

        # Eliminate columns from A, compute a linear mapping T that re-maps the
        # non-eliminated state variables onto their right index.
        keep_col_idcs = []
        for j in range(n):
            # If we're not doing a substitution for this variable, keep the
            # corresponding column in A
            if subs[j] is None:
                keep_col_idcs.append(j)
                continue

            # Since we know that value of the variable corresponding to column
            # j, we can move the contents of column j to the rhs
            for i in range(len(keep_row_idcs)):
                b[i] -= A[i, j] * subs[j]
        A = A[:, keep_col_idcs]
        T = np.eye(n)[:, keep_col_idcs]

        # Simplify all cells
        m, n = A.shape
        for i in range(m):
            b[i] = sp.simplify(b[i], rational=None)
            for j in range(n):
                A[i, j] = sp.simplify(A[i, j], rational=None)

        # If the user did not want the symbolic solution, just turn the matrices
        # into callable functions
        if return_function:
            args = S.var_origin + S.var_theta + S.var_dorigin + S.var_dtheta + S.torques
            f_A = sp.lambdify(args, A, modules=["numpy"])
            f_b = sp.lambdify(args, b, modules=["numpy"])
            f_subs = sp.lambdify(args,
                                 [0.0 if (x is None) else x for x in subs],
                                 modules=["numpy"])
            return f_A, f_b, f_subs, T, self._n_joints, self._n_torques

        # Solve the linear system symbolically
        symbols = [tars[i] for i in keep_col_idcs]
        solns = sp.linsolve((A, b), symbols)
        if len(solns) == 0:
            raise ValidationError("Could not solve for the dynamics", None)
        else:
            # Otherwise fetch the first set of solutions
            solns = next(iter(solns))

        # Assemble the result, mapping from the symbol names onto the
        # corresponding equation
        res = {}
        for i, soln in enumerate(solns):
            res[symbols[i]] = soln
        for i, x in enumerate(subs):
            if not x is None:
                res[tars[i]] = x
        return res

    @staticmethod
    def state_differential(dyns, x):
        """
        This function maps a state vector "x" onto a state differential given
        a dynamics descriptor as returned by the "dynamics" function for
        "return_function = True".

        Parameters
        ==========

        x:      State vector as stored in the "State.vec" property.
        dyns:   Dynamics descriptor as returned by calling Simulator.dynamics()
                with return_function=True.

        Return Value
        ============

        Returns the time-differential of x.
        """

        # Unpack the dynamics descriptor
        f_A, f_b, f_subs, T, n_joints, n_torques = dyns

        # Compute some handy indices used below
        assert len(x) == 2 * (3 + n_joints) + n_torques
        n, m = n_joints, n_torques

        # i0:i1 --> initial coordinates
        # i1:i2 --> velocities in x
        # j0:j1 --> accls. in the linear system
        i0, i1, i2 = 0, 3 + n, 6 + 2 * n
        j0, j1 = 0, 3 + n

        # Compute the linear system for the given state
        A = f_A(*x)
        b = f_b(*x)

        # Compute the variable subtitutions. These are Lagrange multipliers or
        # accelerations that were determined to have a fixed value
        subs = np.array(f_subs(*x), dtype=np.float64).reshape(-1, 1)

        # Solve for the accelerations and Lagrange multipliers
        if A.size > 0:
            soln = T @ np.linalg.lstsq(A, b, rcond=1e-6)[0]
        else:
            soln = np.zeros((len(subs), 1))

        # The resulting accelerations are the sum of the substitutions and the
        # the solution to the linear system computed above.
        ddx = (soln + subs)[:, 0]

        # Return the state update vector
        return np.concatenate((x[i1:i2], ddx[j0:j1], np.zeros(n_torques)))

    ###########################################################################
    # Visualization                                                           #
    ###########################################################################

    def _visualize_estimate_bounding_box(self, seed=58382, radius=True):
        """
        Used internally to estimate the axis aligned bounding box of the system.
        This is used to automatically scale the display area to fit the entire
        model.
        """

        # Randomly generate states and compute the axis-aligned bounding-box
        aabb = np.array(
            ((np.inf, -np.inf), (np.inf, -np.inf), (np.inf, -np.inf)))

        state = self.initial_state()
        rng = np.random.RandomState(seed)
        for _ in range(100):
            for i in range(len(state)):
                state[i] = rng.uniform(-np.pi, np.pi)
            trafos = self._kinematics(*state.static_vec)
            for T in trafos.values():
                aabb[:, 0] = np.minimum(T[:3, 3], aabb[:, 0])
                aabb[:, 1] = np.maximum(T[:3, 3], aabb[:, 1])

        # Round the result for more stability
        ext = np.maximum(1.0, aabb[:, 1] - aabb[:, 0])
        aabb[:, 0] = np.floor((aabb[:, 0] - 0.1 * ext) * 10.0) / 10.0
        aabb[:, 1] = np.ceil((aabb[:, 1] + 0.1 * ext) * 10.0) / 10.0

        # If "radius" is True, use the maximum extent in one dimension to
        # set the extent of the others.
        if radius:
            max_ext, center = np.max(ext), 0.5 * (aabb[:, 0] + aabb[:, 1])
            aabb[:, 0] = center - 0.5 * max_ext
            aabb[:, 1] = center + 0.5 * max_ext

        return aabb

    def _visualize_raw(self, state=None):
        """
        Uses internally to create the raw drawing commands.
        """

        # Compute the forward kinematics for the given state. Pass the
        # non-symbolic "initial_state" instance.
        if state is None:
            state = self.initial_state()

        T = self._kinematics(*state.static_vec)

        raw = []

        # Draw the links
        for link in self.links:
            loc_src = T[link.src][:3, 3]
            loc_tar = T[link.tar][:3, 3]
            raw.append({
                "type": "link",
                "src": loc_src,
                "tar": loc_tar,
            })

        # Draw the point object locations
        for pobj, trafo in T.items():
            loc = np.array(trafo[:3, 3]).reshape(-1)
            raw.append({
                "type": "object",
                "class": pobj.__class__.__name__.lower(),
                "loc": loc,
                "rot": None,
            })

        # Draw the coordinate system at each point
        for pobj, trafo in T.items():
            for i, ax in enumerate(np.eye(3, 3)):
                l = 0.1
                src = trafo[:3, 3]
                trans = res = np.eye(4)
                trans[:3, 3] = ax
                tar = (trafo @ trans)[:3, 3]
                raw.append({
                    "type": "axis",
                    "class": "xyz"[i],
                    "src": src,
                    "dir": tar - src,
                })

        return raw

    def visualize(self,
                  state=None,
                  kind="matplotlib",
                  handle=None,
                  ax=None,
                  aabb=None):
        """
        Creates a visualisation of the kinematic chain in the given state.

        Parameters
        ==========

        kind: The kind of visualisation to perform. Possible values are
                "raw": Returns a list of "drawing commands"
                "matplotlib": Uses matplotlib to plot a 3D visualization.
        handle: An object returned by a previous call to "visualize". If set,
              this will update the previously drawn diagram.
        ax:   When using matplotlib, specifies the axis into which the
              visualization should be drawn.
        aabb:   The axis-aligned bounding box that should be used to set the
                display area. A triple of (min, max) tuples (one for each
                spatial dimension).
        """

        # Make sure the parameters are correct
        assert (kind in {"raw", "matplotlib"})
        assert (ax is None) or (kind == "matplotlib")
        assert (aabb is None) or (kind != "raw")

        # We always need the raw visualisation data
        raw = self._visualize_raw(state)
        if kind == "raw":
            return raw

        # Estimate the bounding box if none is given
        if (aabb is None) and (handle is None):
            aabb = self._visualize_estimate_bounding_box()

        if kind == "matplotlib":
            from .visualization import _visualize_matplotlib
            return _visualize_matplotlib(raw, handle, ax, aabb)
