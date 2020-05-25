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

from .errors import *
from .model import *
from .utils import *

class State:
    """
    The State class describes the state of the model. In particular, this class
    tracks the state (i.e., angle) of each joint, as well as the dynamics (i.e.,
    angular velocity).

    Do not manually construct State instances, use the Simulator.initial_state()
    member function instead.
    """
    def __init__(self, joint_idx_map, loc, rot, t0=0.0):
        assert len(loc) == len(rot) == 3

        # Copy the given arguments
        self._joint_idx_map = joint_idx_map
        if len(joint_idx_map) == 0:
            self._n_joints = 0
        else:
            self._n_joints = max(joint_idx_map.values()) + 1

        # State variable keeping track of the current time
        self._t = t0

        # State variables keeping track of the location and rotation of the root
        # element
        self._loc = np.copy(loc)
        self._rot = np.copy(rot)

        # Create arrays holding the joint states and velocities
        self._state = np.zeros((self._n_joints, 2), dtype=np.float64)

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

    def _check_key(self, key):
        # If the given key is an int, slice, or an array, use the key directly
        if isinstance(key, (int, slice)) or hasattr(key, '__len__'):
            return key
        elif key in self._joint_idx_map:
            return self._joint_idx_map[key]
        else:
            raise KeyError("Key {} not found".format(key))

    def __getitem__(self, key):
        return self._state[self._check_key(key)]

    def __setitem__(self, key, value):
        self._state[self._check_key(key)] = value

    def __iter__(self):
        return self._state.__iter__()


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
        self._loc = np.copy(loc)
        self._rot = np.copy(rot)

        # Copy and validate the "chain" object
        self.object_map = {}  # External to internal object map
        self.chain = chain.copy(self.object_map).coerce()

        # Translate the given root object into the corresponding root object in
        # the copy
        self.root = self.object_map[root].clone

        # Build the spanning tree
        self._tree = self.chain.tree(self.root)

        # Build the _joint_idx_map assigning an index to each joint
        self._joint_idx_map = {}
        for i, joint in enumerate(self.joints):
            self._joint_idx_map[joint] = i
            self._joint_idx_map[self.object_map[joint].original] = i

    @property
    def joints(self):
        return filter(lambda x: isinstance(x, Joint), self.chain._objs)

    @property
    def links(self):
        return filter(lambda x: isinstance(x, Link), self.chain._objs)

    @property
    def point_objects(self):
        return filter(lambda x: isinstance(x, PointObject), self.chain._objs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def initial_state(self):
        """
        Returns a new State object describing the initial state of the model.
        """

        # Create the state object
        state = State(self._joint_idx_map, self._loc, self._rot, 0.0)

        # Set the initial joint angle
        for joint in self.joints:
            state[joint][0] = joint.theta

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
        specified point object.

        If an object is returned, returns a transformation matrix for each point
        object in the chain encoding the location and rotation of that point.

        Parameters
        ==========

        state: The state object determines the configuration of all joints. If
               None, computes the forward kinematics of the chain in its initial
               configuration.
        obj:   The object for which thre transformation should be computed. If
               "None", the transformation will be computed for all point objects
               in the chain. TODO
        use_external_objects: If true, the returned object to transformation
               map will use the original user-provided object references.
               Setting this to False will return internal object references.
               This is mainly meant for internal use.
        """

        if state is None:
            state = self.initial_state()

        # Compute the root object transformation
        loc, rot = state.loc, state.rot
        trafo = trans(*loc) @ rot_z(rot[2]) @ rot_y(rot[1]) @ rot_x(rot[0])

        # Store the root object in a transformation dictionary
        T = {self.root: trafo}

        # Perform a breadth-first search on the tree to compute all
        # transformations
        queue = deque((self.root,))
        while len(queue) > 0:
            # Fetch the current node
            cur = queue.popleft()

            # Nothing else to do if thise node is a leaf node
            if not cur in self._tree:
                continue

            # Iterate over all links and compute the transformation of the
            # object
            for link in self._tree[cur]:
                trafo = T[cur]
                if isinstance(link.src, Joint):
                    print(state[link.src])
                    trafo = link.src.trafo(theta=state[link.src][0]) @ trafo
                trafo = link.trafo() @ trafo
                T[link.tar] = trafo
                queue.append(link.tar)

        # Translate the internal clone object references to external references
        if use_external_objects:
            res = {}
            for key, value in T.items():
                res[self.object_map[key].original] = value
            return res
        else:
            return T

    def _visualize_raw(self, state=None):
        # Compute the forward kinematics for the given state
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

        return raw

    def _visualize_matplotlib(self, raw, ax):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Fetch the figure and the axis object
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        # Iterate over the drawing commands and draw them
        for cmd in raw:
            if cmd["type"] == "link":
                ax.plot(
                    [cmd["src"][0], cmd["tar"][0]],
                    [cmd["src"][1], cmd["tar"][1]],
                    [cmd["src"][2], cmd["tar"][2]],
                    linestyle="-",
                    color="k"
                )
            elif cmd["type"] == "object":
                styles = {
                    "fixture": {
                        "marker": "+",
                        "color": "k"
                    },
                    "joint": {
                        "marker": "o",
                        "color": "b"
                    },
                    "mass": {
                        "marker": "s",
                        "color": "r"
                    },
                }
                ax.plot(
                    [cmd["loc"][0]],
                    [cmd["loc"][1]],
                    [cmd["loc"][2]],
                    **styles[cmd["class"]]
                )

        return fig, ax

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
            return self._visualize_matplotlib(raw, ax)

