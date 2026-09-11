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

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from stix.typing import Scalar


def reduce_modality_losses(
    losses: PyTree[Scalar],
    weights: PyTree[Scalar | float] | None = None,
) -> Scalar:
    r"""Reduce a per-modality pytree of scalar losses to a single scalar.

    A small helper for writing ``GenerativeModel.get_loss``: compute one scalar
    loss per modality (typically via ``modality_registry.map``), then collapse
    them here. Without ``weights`` this is a plain mean over the modality leaves;
    with ``weights`` it is the weight-normalised mean

    .. math::
        \mathcal{L} = \frac{\sum_i w_i\,\mathcal{L}_i}{\sum_j w_j}.

    Args:
        losses: A pytree whose leaves are per-modality scalar losses (e.g. the
            output of ``modality_registry.map(...)``).
        weights: Optional pytree of per-modality weights, structurally compatible
            with ``losses``; each leaf is a scalar (a Python ``float`` or a JAX
            scalar). This is how a custom ``get_loss`` weights different modalities.
            ``None`` weights every modality equally.

    Returns:
        The reduced scalar loss.
    """
    if weights is None:
        return jnp.mean(jnp.stack(jax.tree.leaves(losses)))

    weighted = jax.tree.map(lambda loss, weight: loss * weight, losses, weights)
    total = jnp.sum(jnp.stack(jax.tree.leaves(weighted)))
    normaliser = jnp.sum(jnp.stack(jax.tree.leaves(weights)))
    return total / normaliser
