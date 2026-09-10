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

"""EMA utilities for the training loop."""

import jax
from flax import nnx


def _ema_update(
    ema_params: nnx.State,
    new_params: nnx.State,
    decay: float,
) -> nnx.State:
    """Update EMA parameters: ema <- decay * ema + (1 - decay) * new."""
    return jax.tree.map(
        lambda ema, new: decay * ema + (1.0 - decay) * new,
        ema_params,
        new_params,
    )


def get_debiased_ema(
    ema_params: nnx.State,
    decay: float,
    num_updates: int | jax.Array,
) -> nnx.State:
    """Bias-corrected EMA (Adam convention).

    Assumes ``ema_params`` was zero-initialised and has been updated
    ``num_updates`` times. The correction ``ema / (1 - decay^num_updates)``
    makes the result unbiased under stationary-parameter expectations.

    Must not be called with ``num_updates == 0`` (division by zero).

    Args:
        ema_params: The (biased) EMA parameter tree, zero-initialised at step 0.
        decay: The EMA decay rate used during accumulation.
        num_updates: Number of EMA updates applied so far.

    Returns:
        The bias-corrected EMA parameters, same structure as ``ema_params``.
    """
    correction = 1.0 - decay**num_updates
    return jax.tree.map(lambda e: e / correction, ema_params)
