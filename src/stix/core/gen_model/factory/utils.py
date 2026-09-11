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

import jax.numpy as jnp

from stix.typing import Mask, Var


def collapse_mask(mask_k: Mask, z_t_k: Var) -> Mask:
    """Collapse a mask of shape (*dims, mask_dim) to a per-position 0/1 mask of shape (*dims,).

    ``max`` over the mask_dim axis yields one bit per position.

    Args:
        mask_k: The mask of shape ``(*dims, mask_dim)``.
        z_t_k: The embedded state of shape ``(*dims, state_dim)``.

    Returns:
        A per-position mask of shape ``(*dims,)``.
    """
    if mask_k.ndim == z_t_k.ndim:
        return jnp.max(mask_k, axis=-1)
    return mask_k


def mask_out_revealed_states(active: Mask, z_t_k: Var, mask_index: int) -> Mask:
    """Zero positions that are no longer in a "masked" state for MaskDiscreteInterpolant.

    Args:
        active: Per-position 0/1 mask of shape ``(*dims,)``.
        z_t_k: The embedded state of shape ``(*dims, num_states)``.
        mask_index: Index of the mask symbol along the state axis.

    Returns:
        ``active`` with revealed (non-mask) positions set to 0.
    """
    still_mask = z_t_k[..., mask_index].astype(active.dtype)
    return active * still_mask


def drop_mask_symbol_from_one_hot(one_hot_target: Var, mask_index: int) -> Var:
    """Drop the mask-symbol channel from a one-hot vector.

    Args:
        one_hot_target: One-hot of shape ``(*dims, num_states)``.
        mask_index: Index of the mask symbol along the last axis.

    Returns:
        One-hot of shape ``(*dims, num_states - 1)`` over the data categories.
    """
    return jnp.concatenate(
        [one_hot_target[..., :mask_index], one_hot_target[..., mask_index + 1 :]],
        axis=-1,
    )
