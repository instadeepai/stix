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

"""Core type aliases shared across the library."""

from collections.abc import Sequence
from typing import Annotated

import jax
from jaxtyping import Array

__all__ = [
    "Mask",
    "PRNGKeyArray",
    "Scalar",
    "Shape",
    "Time",
    "Var",
]

#: A single modality's variable array (data or network state).
type Var = Array
#: An array shape as a sequence of dimension sizes.
type Shape = Sequence[int]
#: Scalar interpolation time in ``[0, 1]``.
type Time = Annotated[Array, "scalar time in [0, 1]"]
#: A scalar array.
type Scalar = Annotated[Array, "scalar"]
#: A boolean mask array.
type Mask = Annotated[Array, "boolean mask"]

#: A JAX PRNG key. Semantically distinct from a generic ``jax.Array``.
PRNGKeyArray = jax.Array
