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

"""Fuse a per-modality pytree into one array and split it back again.

A fusion backbone mixes the modalities, so it cannot work leaf by leaf: it
concatenates them into one stream, operates on that, then splits the result
back. :func:`fuse` and :func:`unfuse` are the two halves of that round trip,
and passing the ``treedef`` and ``sizes`` from one to the other rebuilds the
structure that went in: the same container type, not merely the same
modalities.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from stix.typing import Mask, Var


def fuse(modalities: PyTree[Var], axis: int = 0):
    """Concatenate the leaves of ``modalities`` along ``axis``.

    Args:
        modalities: Per-modality arrays.
        axis: Axis to concatenate along.

    Returns:
        The fused array, plus the ``treedef`` and per-modality sizes that
        :func:`unfuse` needs to rebuild the input structure.
    """
    leaves, treedef = jax.tree.flatten(modalities)
    sizes = [leaf.shape[axis] for leaf in leaves]
    return jnp.concatenate(leaves, axis=axis), treedef, sizes


def unfuse(fused: Var, treedef, sizes: list[int], axis: int = 0) -> PyTree[Var]:
    """Split ``fused`` back into the structure :func:`fuse` was given.

    Args:
        fused: The fused array.
        treedef: The structure returned by :func:`fuse`.
        sizes: The per-modality sizes returned by :func:`fuse`.
        axis: Axis to split along.

    Returns:
        The per-modality arrays, in the structure that was fused.
    """
    parts = jnp.split(fused, np.cumsum(sizes[:-1]), axis=axis)
    return jax.tree.unflatten(treedef, parts)


def fuse_mask(
    mask: PyTree[Mask | None] | None, treedef, sizes: list[int]
) -> Mask | None:
    """Concatenate a per-modality mask to line up with a fused stream.

    A whole ``None`` means no masking. A ``None`` in place of one modality's
    mask leaves that modality fully visible, so a caller only has to build the
    masks it actually needs.

    Args:
        mask: Per-modality boolean masks, shaped like the fused modalities.
        treedef: The structure returned by :func:`fuse`.
        sizes: The per-modality sizes returned by :func:`fuse`.

    Returns:
        One boolean per position of the fused stream, or ``None`` if ``mask``
        is ``None``.

    Raises:
        ValueError: If a mask is not one boolean per position of its own
            modality. Such a mask can still concatenate to the right total
            length while gating the wrong positions.
    """
    if mask is None:
        return None
    parts = []
    for part, size in zip(treedef.flatten_up_to(mask), sizes):
        if part is None:
            parts.append(jnp.ones((size,), dtype=jnp.bool_))
            continue
        if part.shape != (size,):
            msg = f"expected a mask of shape {(size,)} per modality; got {part.shape}"
            raise ValueError(msg)
        parts.append(part.astype(jnp.bool_))
    return jnp.concatenate(parts)
