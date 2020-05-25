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


class ValidationError(ValueError):
    """
    Exception raised if the specified model is invalid. The model is checked for
    validity when the simulator object is created.
    """
    def __init__(self, msg, obj=None):
        if not obj is None:
            msg = "Error while validating object {}: {}".format(repr(obj), msg)
        super().__init__(msg)


class CyclicChainError(ValidationError):
    """
    Exception raised if the connection grpah is cyclic, i.e., a PointObject is
    (indirectly) supported by itself.
    """
    pass


class DisconnectedChainError(ValidationError):
    """
    Exception raised if the kinematic chain is not a single connected component,
    i.e., not all PointObject instances can be reached from the specified root
    object.
    """
    pass


class MultipleSourcesError(ValidationError):
    """
    ValidationError raised if the specified model is cyclic (i.e., multiple
    source PointObject instances are linked to the same target object).
    """
    pass

