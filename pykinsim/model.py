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
"""
This file contains all the data objects holding information about the model
built by the user. These objects are merely data containers and do not contain
any simulation logic. The "Simulator" class must be used to compile a model
into an actual simulation.
"""

from .errors import (ValidationError, CyclicChainError, DisconnectedChainError,
                     MultipleSourcesError)
from .utils import variables
from .geometry import scalar, trans, rot_x, rot_y, rot_z

import sympy as sp
import re
from collections import deque, OrderedDict

# Initialize thread-local storage -- this is needed to properly implement the
# "with" magic of the Chain object
import threading
_thread_local = threading.local()
"""
When used as a value for "torque" in the Joint constructor, signifies that the
torque will be supplied by the user in the run/step function.
"""
External = object()


class Labeled:
    """
    A "Labeled" object is simply an object with a "label" property. This label
    will be printed as the representation of the object. PyKinSim automagically™
    deduces object labels from variable names whenever the scope of a
    "with Chain..." statement is left.
    """
    def __init__(self, label=None):
        self.label = label

    def __repr__(self):
        fmt = {
            "label": str(self.label),
            "class": self.__class__.__name__,
            "addr": id(self),
        }
        if self.label is None:
            return "<{class} @ 0x{addr:02x}>".format(**fmt)
        else:
            return "<{label} ({class} @ 0x{addr:02x})>".format(**fmt)


class Chain(Labeled):
    """
    The Chain object is simply a container of Objectinstances. This class, and
    all the objects contained therein are purely data-containers, i.e., they do
    not perform any computation or validation.

    A Simulator object must be used to validate a Chain, to simulate its rigid
    body dynamics of the Chain, and to compute forward-kinematics.

    The Chain object is intended to be used in conjunction with a "with"
    statement. Object instances created inside this block will automatically be
    added to the Chain (see the example below).

    Example
    =======

        import pykinsim as pks

        # Build the Chain
        with pks.Chain() as chain:
            m1 = pks.Mass(1.0)
            m2 = pks.Mass(0.5)
            l = pks.Link(m1, m2)

        # Simulate the Chain with "m1" as the origin
        with pks.Simulator(chain, root=m1) as sim:
            # Run the simulation for 1s at 1e-3 timesteps
            dt, T = 1e-3
            for _ in range(0, T, dt):
                # Perform a single step (this may perform multiple adaptive
                # stepsize simulation steps internally)
                sim.step(dt=dt, g=(0, 0, -9.81))

                # Compute the locations and orientations of all KinematicObject
                # instances in the kinematics chain (see result map)
                sim.forward_kinematics()
    """
    def __init__(self, label=None):
        """
        Constructor of the Chain object.

        Parameters
        ==========

        label: An (optional) name describing this object.
        """
        # Create the super-constructor
        super().__init__(label)

        # Copy the given arguments
        self.label = label

        # Create the object container; explicitly use an OrderedDict since
        # we want to maintain the object insertion order
        self._objs = OrderedDict()

    def __enter__(self):
        if not hasattr(_thread_local, 'active_chain'):
            _thread_local.active_chain = []
        _thread_local.active_chain.append(self)

        return self

    def __exit__(self, type, value, traceback):
        _thread_local.active_chain.pop()

        # Try to assign labels to objects that do not have labels yet
        names = variables(skip=1, flt=lambda obj: isinstance(obj, Labeled))
        if (self in names) and (self.label is None):
            self.label = names[self]
        for obj in self._objs:
            if (obj in names) and (obj.label is None):
                obj.label = names[obj]

    def __contains__(self, key):
        # TODO: Support nested chains?
        return key in self._objs

    def coerce(self):
        """
        Creates a copy of the Chain object in which all child-objects have been
        validated and normalised.
        """

        # Make sure the label is a string
        if not self.label is None:
            self.label = str(self.label)

        # Coerce all child objects. The "coerce_obj" function will add the
        # object to the result "Chain" object.
        for obj in self._objs:
            # Make sure the object is of the right type
            if not isinstance(obj, (Link, Mass, Joint, Fixture)):
                raise ValueError(
                    "Object is not a Link, Mass, Joint or Fixture")

            # Coerce the object
            obj.coerce()

        return self

    def tree(self):
        """
        Builds a spanning tree. Automatically choses a suitable root node from
        which all other nodes can be reached. This function may raise various
        ValidationError types if the chain cannot be converted into a tree
        (i.e., because there are cycles in the chain, multiple source joints
        for a link or multiple disconnected components in the graph).
        """

        # Make sure the chain is not empty
        pobjs = set(filter(lambda x: isinstance(x, PointObject), self._objs))
        if len(pobjs) == 0:
            raise ValidationError(
                "Kinematic chain must contain at least one point object", self)

        # Collect all possible links as a map of source objects onto links
        links = {}
        roots = set(pobjs)
        for link in filter(lambda x: isinstance(x, Link), self._objs):
            # Remove the target node from the set of potential roots
            if link.tar in roots:
                roots.remove(link.tar)
            else:
                raise MultipleSourcesError(
                    "Node {} already targeted by another link".format(
                        link.tar), link)

            # Store forward and backward links as tuples
            if not link.src in links:
                links[link.src] = []
            if not link.tar in links:
                links[link.tar] = []
            links[link.src].append(link)

        # There must be exactly one root node; still to give a better error
        # message below, just use either the first point object or the first
        # possible root.
        if len(roots) == 0:
            raise CyclicChainError(
                "No possible root node found; kinematic chain is cyclic")

        # Fetch the first possible node as a root node
        root = next(iter(roots))

        # Perform a breadth-first search on the graph
        tree = {}
        visited, links_visited, queue = {root}, set(), deque((root, ))
        while len(queue) > 0:
            # Get the current item from the queue
            cur = queue.popleft()

            # Make sure this node has outgoing links
            if not cur in links:
                continue

            # Iterate over all links originating from this node
            for link in links[cur]:
                # Continue if this link has already been used
                if link in links_visited:
                    continue
                links_visited.add(link)  # Remember that we've used this link

                # Fetch the target object of this link
                tar = link.tar

                # Make sure the target object has not already been visited
                if tar in visited:
                    raise CyclicChainError(
                        "Kinematic chain is cyclic; {} can be reached through "
                        "multiple paths from root {} through this link {}".
                        format(tar, root, link), link)
                visited.add(tar)  # Remember that we've visited this node

                # Add the node to the queue
                queue.append(tar)

                # Add this link to the tree
                k, v = cur, link
                if not k in tree:
                    tree[k] = []
                tree[k].append(v)

        # Make sure all point objects were visited
        for pobj in filter(lambda x: isinstance(x, PointObject), self._objs):
            if not pobj in visited:
                raise DisconnectedChainError(
                    "The point object {} is not connected to the root node {}".
                    format(pobj, root), pobj)

        # If there were multiple possible roots, one of the exceptions above
        # should have triggered. Still, assert this, just to be on the safe
        # side.
        assert len(roots) == 1, "Multiple roots. This should not happen."

        return tree, root


class Object(Labeled):
    def __init__(self, label=None, parent=None):
        # Call the super-constructor
        super().__init__(label)

        # If no parent object has been given, make sure the constructor has been
        # called while a Chain object is being constructed. Otherwise just use
        # the given parent.
        if parent is None:
            if len(getattr(_thread_local, 'active_chain', [])) == 0:
                raise RuntimeError(
                    "{} instances must be constructed within an active chain".
                    format(self.__class__.__name__))

            # Fetch the parent kinematic chain object
            self.parent = _thread_local.active_chain[-1]
        else:
            # Use the given parent as a parent
            self.parent = parent

        # Add this object the parent object set
        self.parent._objs[self] = None

    def coerce(self):
        """
        The coerce method checks and normalises all parameters; returns a
        reference at this, or a copied instance.
        """

        # Make sure the parent object is correct
        if (self.parent is None) or (not self in self.parent):
            raise ValidationError("Invalid parent", self)

        # Make sure the label is either "None" or a string
        if not self.label is None:
            self.label = str(self.label)


class Link(Object):
    """
    Describes a rigid connection between two PointObject instances (i.e., a
    Joint, Mass, or Fixture). A link does not have any mass, or any dynamical
    properties, it just describes a fixed relationship between two PointObject
    instances.

    The Simulator object will ensure that there are no conflicting links between
    objects once a Chain is instantiated. I.e., there cannot be two links
    between two PointObject instances with conflicting specifications.

    Note that links are directed, i.e., a link from an object "o1" to "o2" is
    different from a link from "o2" to "o1". See the constructor documentation
    for more information.
    """
    def __init__(self,
                 src,
                 tar,
                 l=1.0,
                 ry=0.0,
                 rz=0.0,
                 label=None,
                 parent=None):
        """
        Creates a link from the object "src" to the object "tar". Per default,
        the object "tar" is located l units away along the x-axis of src's local
        coordinate system. This translation axis can be rotated along the local
        y- and z-axis by specifying the ry and rz parameters (in rad).

        Note that the local coordinate system of the target object is aligned
        with the link axis; this means that---right now---there can not be
        multiple links targeting the same object (but multiple links can
        originate from the same object). Correspondingly, the resulting graph
        structure must be a directed, acyclic graph (with one connected
        component).

        Parameters
        ==========

        src: Is the source object, i.e., the PointObject instance from which the
             link originates.
        tar: Is the target object, i.e., the PointObject instance that is at the
             end of the link.
        l: Is the length of the link in meters.
        ry: Is the rotation in rad of the link around the y-axis of the
            local coordinate system of the source object.
        rz: Is the rotation in rad of the link around the z-axis of the
            local coordinate system of the source object.
        """

        # Call the inherited super-constructor
        super().__init__(label, parent)

        # Copy the provided parameters
        self.src = src
        self.tar = tar
        self.l = l
        self.ry = ry
        self.rz = rz

    def trafo(self):
        return rot_y(self.ry) @ rot_z(self.rz) @ trans(x=self.l)

    def coerce(self):
        # Call the inherited implementation of "coerce"
        super().coerce()

        # Make sure src and tar are PointObject instances
        if ((not isinstance(self.src, PointObject))
                or (not self.src in self.parent)):
            raise ValidationError("Invalid link source", self)
        if ((not isinstance(self.tar, PointObject))
                or (not self.tar in self.parent)):
            raise ValidationError("Invalid link target", self)

        # Make sure l, ry, rz are floats
        self.l = scalar(self.l)
        self.ry = scalar(self.ry)
        self.rz = scalar(self.rz)


class PointObject(Object):
    """
    The PointObject class is a base class for all objects in the kinematic chain
    that can be described as a being located at a single point. This includes
    Joint, Mass, and Fixture objects. PointObject instances do not describe
    the relationship in the location/orientation.
    """
    pass


class Joint(PointObject):
    """
    Describes a rotational joint along the given axis. As with all PointObject
    classes, a joint itself does not specify which objects are connected to it,
    it merely acts as a connection site for a Link.
    """
    def __init__(self,
                 axis="y",
                 theta=0.0,
                 torque=0.0,
                 label=None,
                 parent=None):
        """
        Creates a new Joint object.

        axis:   Describe the axis of rotation, relative to the current local
                coordinate system. Must be one of "x", "y", "z".
        theta:  Initial rotation angle. An angle of zero means that a link
                originating from this joint will be continued relative to the
                local coordinate system of the joint without any change.
        torque: Describes an (external) torque that is being applied to this
                joint. When set to `pykinsim.External`, the torque for this
                joint may be specified in the run/step functions. If a constant
                number is given, a constant torque is applied to the joint.
                Setting "torque" to None is equivalent to a constant torque of
                zero.
        label:  The label assigned to the joint object. If None, pykinsim tries
                to automagically™ deduce a label from the variable names used in
                the code.
        parent: The parent Chain object this object belongs to. Usually you
                should not set this parameter and instead 
        """

        # Call the inherited super-constructor
        super().__init__(label, parent)

        # Copy the provided parameters
        self.axis = axis
        self.theta = theta
        self.torque = torque

    def trafo(self, theta):
        if self.axis == "x":
            return rot_x(theta)
        elif self.axis == "y":
            return rot_y(theta)
        elif self.axis == "z":
            return rot_z(theta)

    def coerce(self):
        # Call the inherited coerce function
        super().coerce()

        # Make sure "axis" is a string and convert it to lowercase
        self.axis = str(self.axis).lower()

        # Make sure the axis specifier is valid
        if not self.axis in ["x", "y", "z"]:
            raise ValidationError(
                "Invalid axis specifier \"{}\"; must be one of "
                "{{\"x\", \"y\", \"z\"}}".format(self.axis, self))

        # Make sure theta is valid
        self.theta = scalar(self.theta)

        # Make sure torque is either a special value or a number
        if self.torque is External:
            pass  # Nothing to do
        else:
            self.torque = scalar(self.torque)


class Mass(PointObject):
    """
    Describes a mass point object. Masses are objects posessing an (angular)
    momentum and are affected by gravity.
    """
    def __init__(self, m=1.0, label=None, parent=None):
        # Call the inherited super-constructor
        super().__init__(label, parent)

        # Copy the provided parameters
        self.m = m

    def coerce(self):
        # Call the inherited coerce function
        super().coerce()

        # Make sure the given mass is a float
        self.m = scalar(self.m)

        # Make sure masses are not negative. Physics gets all weird with negative
        # masses.
        if self.m < 0.0:
            raise ValidationError("Masses must not be negative", self)


class Fixture(PointObject):
    """
    A fixture is a point object with constrained translational motion. Per
    default a fixture cannot move at all, i.e., it is fixed in space the moment
    the simulation instance is created. This constraint can be relaxed by
    specifying fewer axes along which the motion is constrained.

    Fixtures serve as anchors for the kinematic chain, i.e., as a base for a
    robotic arm.
    """
    def __init__(self, axes="xyz", label=None, parent=None):
        # Call the inherited super-constructor
        super().__init__(label, parent)

        # Copy the provided parameters
        self.axes = axes

    def coerce(self):
        # Call the inherited coerce function
        super().coerce()

        # Make sure "axes" is a string and convert it to lowercase
        self.axes = str(self.axes).lower()

        # Remove all punctuation and space characters
        self.axes = re.sub("[{}(),\\s]", "", self.axes)

        # Make sure the axes specifier is valid
        self.axes = set(self.axes)
        for a in self.axes:
            if not a in {"x", "y", "z"}:
                raise ValidationError(
                    "Invalid axis specifier \"{}\"; must be one of {{\"x\", \"y\", \"z\"}}"
                    .format(a), self)
