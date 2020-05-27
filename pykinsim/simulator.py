#   pykinsim -- Python Kinematics Simulator
#   Copyright (C) 2020  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import sympy as sp

from .errors import *
from .model import *
from .utils import *
from .algebra import *


class Symbols:
    """
    Class collecting all the symbols used to represent the kinematics and
    dynamics of the system.
    """
    def __init__(self, n_joints, n_cstrs):
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
        for i in range(n_joints):
            name = "\\theta_{{{}}}".format(i) if n_joints > 1 else "\\theta"
            self.theta[i] = sp.Function(name)(t)
            self.dtheta[i] = sp.diff(self.theta[i], t)
            self.ddtheta[i] = sp.diff(self.dtheta[i], t)

        # Lagrange multipliers
        self.lambda_ = [None] * n_cstrs
        for i in range(n_cstrs):
            self.lambda_[i] = sp.Symbol("\\lambda_{{{}}}".format(i))

        # Gravity vector
        self.gx, self.gy, self.gz = sp.symbols("g_x g_y g_z")
        self.g = sp.Matrix([self.gx, self.gy, self.gz])


class State:
    """
    The State class describes the state of the model. In particular, this class
    tracks the state (i.e., angle) of each joint, as well as the dynamics (i.e.,
    angular velocity).

    Do not manually construct State instances, use the Simulator.initial_state()
    member function instead.
    """
    def __init__(self, joint_idx_map, symbolic=False):
        # Copy the given arguments
        self._joint_idx_map = joint_idx_map
        if len(joint_idx_map) == 0:
            self._n_joints = 0
        else:
            self._n_joints = max(joint_idx_map.values()) + 1

        # State variable keeping track of the current time
        self._t = 0.0

        # Either use numpy or sympy arrays
        if symbolic:
            zeros = lambda l: sp.zeros(l, 1)
        else:
            zeros = lambda l: np.zeros(l)

        # State variables keeping track of the location and rotation of the root
        # element
        self._loc = zeros(3)
        self._rot = zeros(3)

        # Create arrays holding the joint states and velocities
        self._joint_states = zeros(self._n_joints)
        self._joint_velocities = zeros(self._n_joints)

    @property
    def t(self):
        return self._t

    @property
    def loc(self):
        return self._loc

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


class Simulator:
    def __init__(self, chain, root, loc=None, rot=None):
        """
        Creates a new Simulator object for the given chain. The "root" object
        is the object relative to which the location of all other objects is
        computed.

        As a first step, the Simulator object will validate all parameters set
        on the kinematics objects; i.e., it will ensure that all objects have
        the right type, that there is only one connected component, and that
        the link structure is acyclic.

        Note
        ====

        Once the simulator object has been created, changes to the original
        Chain object do not affect the simulation.

        Parameters
        ==========

        chain: The Chain instance for which the simulator should be created.
        root:  The Mass or Joint object relative to which the location of all
               other objects is reported.
        loc:   An optional three-tuple describing the location of the root
               object within the world coordinate space. If not specified, the
               root object is located at (0, 0, 0).
        rot:   An optional three-tuple describing the rotation of the root
               element in the world coordinate space.
        """

        # Make sure that "chain" and "root" have the right type
        if not isinstance(chain, Chain):
            raise ValidationError("\"chain\" must be an instance of \"Chain\"",
                                  self)
        if not isinstance(root, PointObject):
            raise ValidationError(
                "\"root\" must be a \"PointObject\" instance", self)

        # Make sure the "root" object is actually part of the Chain instance
        if not root in chain:
            raise ValidationError(
                "The given \"root\" object must be part of the chain", self)

        # Make sure loc and rot are correct
        if loc is None:
            loc = np.zeros(3)
        if rot is None:
            rot = np.zeros(3)
        assert len(loc) == 3
        assert len(rot) == 3

        self._loc = loc
        self._rot = rot

        # Copy and validate the "chain" object
        self.object_map = {}  # External to internal object map
        self.chain = chain.copy(self.object_map).coerce()

        # Translate the given root object into the corresponding root object in
        # the copy
        self.root = self.object_map[root].clone

        # Build the spanning tree
        self._tree = self.chain.tree(self.root)

        # Count the number of joints and constraints
        self._n_joints = sum(1 for _ in self.joints)
        self._n_cstrs = sum(len(fix.axes) for fix in self.fixtures)

        # Build the _joint_idx_map assigning an index to each joint
        self._joint_idx_map = {}
        for i, joint in enumerate(self.joints):
            self._joint_idx_map[joint] = i
            self._joint_idx_map[self.object_map[joint].original] = i

        # Construct the symbols used for evaluating certain expressions
        # symbolically
        self._symbols = Symbols(self._n_joints, self._n_cstrs)

    @property
    def joints(self):
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
        state = State(self._joint_idx_map)

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
        state = State(self._joint_idx_map, symbolic=True)

        # Set the initial joint angles
        for i in range(len(state)):
            state.states[i] = self.symbols.theta[i]
            state.velocities[i] = self.symbols.dtheta[i]

        # Use the symbolic x, y, z, rx, ry, rz variables
        for i in range(3):
            state.loc[i] = self.symbols.origin[i]
            #state.rot[i] = self.symbols.rot[i] # TODO

        return state

    def step(self, state=None, g=None, dt=1e-3):
        """
        Advances the simulation by one timestep from the given state. If no
        state is given, a new State() object corresponding to the initial state
        is generated.
        """
        pass

    def forward_kinematics(self, state=None, use_external_objects=True):
        """
        Computes the location and orientation of either all point objects or the
        specified point object. If no state is given, computes the forward
        kinematics symbolically, i.e., as mathematical equations depending on
        the joint angles.

        If an object is returned, returns a transformation matrix for each point
        object in the chain encoding the location and rotation of that point.

        Parameters
        ==========

        state: The state object determines the configuration of all joints. If
               None, computes the forward kinematics symbolically, where the
               joint angles are called "theta_1" to "theta_n"
        obj:   The object for which thre transformation should be computed. If
               "None", the transformation will be computed for all point objects
               in the chain. TODO
        use_external_objects: If true, the returned object to transformation
               map will use the original user-provided object references.
               Setting this to False will return internal object references.
               This is mainly meant for internal use.
        """

        if state is None:
            state = self.symbolic_state()

        # Compute the root object transformation
        loc, rot = state.loc, state.rot
        trafo = trans(*loc) @ rot_z(rot[2]) @ rot_y(rot[1]) @ rot_x(rot[0])

        # Store the root object in a transformation dictionary
        T = {self.root: trafo}

        # Perform a breadth-first search on the tree to compute all
        # transformations
        queue = deque((self.root, ))
        while len(queue) > 0:
            # Fetch the current node
            cur = queue.popleft()

            # Nothing else to do if thise node is a leaf node
            if not cur in self._tree:
                continue

            # Fetch the transformation up to the current point
            trafo = T[cur]

            # If the current source object is a joint, account for the joint
            # transformation
            if isinstance(cur, Joint):
                theta = state.states[state.idx(cur)]
                trafo = trafo @ cur.trafo(theta=theta)

            # Iterate over all link targets
            for link in self._tree[cur]:
                assert (cur is link.src)
                T[link.tar] = trafo @ link.trafo()
                queue.append(link.tar)

        # Simplify the sympy expressions; if the expression is purely numerical,
        # convert it to a numpy array
        for key, value in T.items():
            value = sp.trigsimp(sp.simplify(value, rational=None)).evalf()
            if all(map(lambda x: isinstance(x, sp.core.numbers.Number),
                       value)):
                value = np.array(value, dtype=np.float64)
            T[key] = value

        # Translate the internal clone object references to external references
        if use_external_objects:
            res = {}
            for key, value in T.items():
                res[self.object_map[key].original] = value
            return res
        else:
            return T

    def lagrangian(self, g=None):
        """
        Symbolically computes the Lagrangian L = KE - PE for the given gravity
        vector (or a generic gravity vector if "g" is set to None).

        Parameters
        ==========

        g: Gravity vector for which the Lagrangian should be computed. If set to
           None, uses the symbolic vector [g_x, g_y, g_z].
        """

        # Use the symbolic gravity vector if None is given
        if g is None:
            g = self.symbols.g

        # Symbolically compute the forward kinematics
        trafos = self.forward_kinematics(use_external_objects=False)

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
        return sp.simplify(sp.trigsimp(KE - PE), rational=None)

    def constraints(self):
        """
        Converts the fixture objects into a set of constraint equations.
        """

        # Symbolically compute the forward kinematics
        trafos = self.forward_kinematics(use_external_objects=False)

        # Compute the forward kinematics for the initial state
        trafos_init = self.forward_kinematics(state=self.initial_state(),
                                              use_external_objects=False)

        # For each fixture, add a set of constraints regarding its location
        Cs = []
        idx = 0
        for fixture in self.fixtures:
            for i, axis in enumerate({"x", "y", "z"}):
                if not axis in fixture.axes:
                    continue
                Cs.append(trafos[fixture][i, 3] - trafos_init[fixture][i, 3])
                idx += 1

        return Cs

    def dynamics(self, debug=False, g=None):
        """
        Returns a list of equations describing the dynamics of the system.
        """

        # Shorthand for df/dt, d²/d²t and df/dx
        Dt = lambda f: sp.diff(f, self.symbols.t)
        Dt2 = lambda f: sp.diff(f, (self.symbols.t, 2))
        Dx = lambda f, x: sp.diff(f, x)

        # Compute the Lagrangian for an arbitrary gravity vector
        L = self.lagrangian(g=g)

        # Compute the constraints
        Cs = self.constraints()
        C = sum([self.symbols.lambda_[i] * C for i, C in enumerate(Cs)])

        # Collect all state variables
        vars_ = [*self.symbols.theta, *self.symbols.origin]

        # Compute the energy preservation equation for each variable, then add
        # the constraints, as well as the second order derivative of the
        # constraints the set of equations
        eqns = (
            [Dx(L, var) - Dt(Dx(L, Dt(var))) + Dx(C, var) for var in vars_] +
            Cs + [Dt2(C) for C in Cs])

        # Solve for the second-order derivatives, as well as all Lagrange
        # multpliers
        tar_vars = [Dt2(var) for var in vars_] + self.symbols.lambda_

        # Sometimes we have additional degrees of freedom in our solution. If
        # this is the case, we clamp a corresponding number of variables to zero
        # in the solutions
        zeros = []
        while True:
            if debug:
                print("SOLVING FOR DYNAMICS")
                print("====================")
                print("")
                print("Solving system")
                for i, eqn in enumerate(eqns):
                    print("\t({})\t{}".format(i, eqn))
                print("For variables")
                for i, var in enumerate(tar_vars):
                    print("\t({})\t{}".format(i, var))
                if len(zeros) > 0:
                    print("With superfluous DoFs eliminated by setting")
                    for i, var in enumerate(zeros):
                        print("\t({})\t{} = 0".format(i, var))

            # Actually perform the solving process
            solns = sp.solve(eqns, tar_vars, rational=None, dict=True)

            # Abort if we're not able to find a solution for this system
            if (solns is None) or (len(solns) == 0):
                raise ValidationError("Cannot solve for the system dynamics", self)

            # If "solns" is an array, just use the first solution. Note: I'm not
            # aware of this ever happening, but technically sympy can return a
            # list of solutions
            if isinstance(solns, list):
                if len(solns) > 1:
                    if debug: # TODO: Warning?
                        print("Multiple possible solutions; using first one")
                solns = solns[0]

            if debug:
                print("Solution")
                for i, (key, value) in enumerate(solns.items()):
                    print("\t({})\t{}\t=\t{}".format(i, key, value))
                print()

            # The returned solutions should now be a dictionary
            assert isinstance(solns, dict)

            # If the length of the solutions dictionary is smaller than the
            # total number of variables, this means that we have additional
            # degrees of freedom (this for example happens if a part of the
            # kinematic chain is weightless)
            if len(solns) < len(tar_vars):
                dof = len(tar_vars) - len(solns)
                for i in range(dof):
                    # Extract the first target variable; we'll just assume that
                    # this variable is zero. Note that this will set the joint
                    # angles to zero first, in contrast to sympy, which
                    # arbitrarily selects variables with additional degrees of
                    # freedom.
                    zero_var = tar_vars[0]
                    for i, eqn in enumerate(eqns):
                        eqns[i] = eqn.subs(zero_var, 0)
                    tar_vars = tar_vars[1:]
                    zeros.append(zero_var)

                # Perform another iteration
                continue

            # We're done, exit the loop!
            break

        # Add the variables in the "zeros" list to the result
        for var in zeros:
            solns[var] = 0.0

        # Return the solutions
        return solns

    def _visualize_raw(self, state=None):
        # Compute the forward kinematics for the given state. Pass the
        # non-symbolic "initial_state" instance.
        if state is None:
            state = self.initial_state()
        T = self.forward_kinematics(state, use_external_objects=False)

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
            loc = trafo[:3, 3]
            raw.append({
                "type": "object",
                "class": pobj.__class__.__name__.lower(),
                "loc": loc,
                "rot": None,
            })

        # Draw the coordinate system at each point
        for pobj, trafo in T.items():
            for i, ax in enumerate(np.eye(3, 3)):
                src = trafo[:3, 3]
                tar = np.array((trafo @ trans(*ax)), dtype=np.float64)[:3, 3]
                raw.append({
                    "type": "axis",
                    "class": "xyz"[i],
                    "src": src,
                    "tar": tar,
                })

        return raw

    def visualize(self, state=None, kind="matplotlib", ax=None):
        """
        Creates a visualisation of the kinematic chain in the given state.

        kind: The kind of visualisation to perform. Possible values are
              "raw": Returns a list of "drawing commands"
        """

        # Make sure the parameters are correct
        assert kind in {"raw", "matplotlib"}
        assert (ax is None) or (kind == "matplotlib")

        raw = self._visualize_raw(state)
        if kind == "raw":
            return raw
        elif kind == "matplotlib":
            from .visualization import _visualize_matplotlib
            return _visualize_matplotlib(raw, ax)

