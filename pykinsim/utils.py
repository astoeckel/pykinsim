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

# Try to import the traceback module. This is used for automagic component
# label extraction
try:
    import traceback
except ModuleNotFoundError:
    traceback = None


def variables(skip=0, flt=None):
    """
    Returns a map from object id to the corresponding variable names over all
    stack frames. The innermost name of an object is returned.

    skip: Number of stack frames to skip.
    flt:  Optional filter function that is applied to each object before its
          name is recorded.
    """

    # If we have the "traceback" module, dump the stack and list all variables
    # matching the filter
    names = {}
    if not traceback is None:
        for i, (frame, _) in enumerate(traceback.walk_stack(None)):
            if i < skip:
                continue
            for key, value in frame.f_locals.items():
                if (flt is None) or (flt(value)):
                    if (not (value in names)): # innermost name
                        names[value] = key

    return names

