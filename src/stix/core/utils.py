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

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from stix.typing import Scalar, Time


def check_gamma_is_not_zero(gamma_fn: Callable[[Time], Scalar]) -> bool:
    """Check that ``gamma_fn`` is non-zero on the interior of ``[0, 1]``.

    Evaluated on a 1000-point grid excluding the endpoints, so a ``True`` result is
    necessary but not sufficient.
    """
    tol = 1e-6
    n_grid_points = 1000
    times = jnp.linspace(0 + tol, 1 - tol, n_grid_points)
    gamma_ts = gamma_fn(times)
    return bool(jnp.all(gamma_ts != 0))


def is_one_hot_vector(x: jnp.ndarray) -> jnp.ndarray:
    """Checks if all vectors along the last axis of ``x`` are valid one-hot vectors.

    One-hot vectors are detected by checking that:
    1. They hold a non-integer dtype,
    2. They hold binary values (through ``jnp.logical_or(x == 0, x == 1)``),
    3. They sum to 1.

    Returns a single scalar JAX boolean.

    NOTE: One-hot encodings are produced by :func:`jax.nn.one_hot`, which yields a float dtype.

    Args:
        x: The input array to check.

    Returns:
        A scalar JAX boolean indicating whether each vector along the last axis
        is a valid one-hot encoding.
    """
    if jnp.issubdtype(x.dtype, jnp.integer):
        return jnp.bool_(False)
    is_zero_or_one = jnp.logical_or(x == 0, x == 1)
    all_binary = jnp.all(is_zero_or_one, axis=-1)

    sum_is_one = jnp.sum(x, axis=-1) == 1

    return jnp.all(all_binary & sum_is_one)


def infer_num_samples_from_raw_source(raw_source: PyTree) -> int | None:
    """Leading batch size shared by all non-``None`` ``raw_source`` leaves.

    Args:
        raw_source: Pytree of raw source arrays or ``None``.

    Returns:
        The common leading axis, or ``None`` if every leaf is ``None``.

    Raises:
        ValueError: If non-``None`` leaves disagree on the leading axis.
    """
    sizes = [
        int(leaf.shape[0]) for leaf in jax.tree.leaves(raw_source) if leaf is not None
    ]
    if not sizes:
        return None
    if len(set(sizes)) > 1:
        raise ValueError(f"raw_source leaves have inconsistent batch sizes: {sizes}")
    return sizes[0]
