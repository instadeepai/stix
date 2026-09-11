# Copyright 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class VarType(Enum):
    """Enum of prediction target types for the stochastic interpolant.

    The types of variables we accept are:

    - ``RAW_SOURCE``
    - ``RAW_TARGET``
    - ``EMBEDDED_SOURCE``
    - ``EMBEDDED_TARGET``
    - ``NOISE``
    - ``VELOCITY_FIELD``
    - ``VELOCITY``
    - ``SCORE``

    For more information see the paper
    `here <https://www.jmlr.org/papers/v26/23-1605.html>`_.

    Note:
        This is a *vocabulary* enum: it is the canonical name for each
        common prediction target, referenced across the tests, docs, and
        tutorials.
        This being said, the user can easily target something with their network
        beyond these quantities.
    """

    RAW_SOURCE = "raw_source"
    RAW_TARGET = "raw_target"
    EMBEDDED_SOURCE = "embedded_source"
    EMBEDDED_TARGET = "embedded_target"
    NOISE = "noise"
    VELOCITY_FIELD = "velocity_field"
    VELOCITY = "velocity"
    SCORE = "score"
