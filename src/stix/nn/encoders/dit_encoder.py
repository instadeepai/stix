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

"""Per-modality DiT encoder.

Follows the DiT architecture (Peebles & Xie, 2023): a linear projection into
embedding space plus a learned positional encoding, with optional input
standardisation driven by the interpolant schedules.
"""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from stix.nn.network_dims import NetworkDimsConfig
from stix.typing import Scalar, Time, Var


class DiTEncoder(nnx.Module):
    """Per-modality encoder: linear projection + learned positional encoding.

    Optionally applies input standardisation when ``beta_fn`` and ``gamma_fn`` are
    provided: ``z_t = z_t / clip(sqrt(beta_t**2 + gamma_t**2), min=1e-4)``.
    Standardisation must happen before the linear projection (the layer has bias,
    so order matters).
    """

    def __init__(
        self,
        network_dims: NetworkDimsConfig,
        *,
        input_dim: int,
        num_tokens: int = 1,
        gamma_fn: Callable[[Time], Scalar] | None = None,
        beta_fn: Callable[[Time], Scalar] | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise linear projection and learned positional encoding.

        Args:
            network_dims: Shared network dimensions (uses ``embedding_dim`` and ``dtype``).
            input_dim: Raw input channel dimension for this modality.
            num_tokens: Number of tokens this modality occupies in the sequence.
            gamma_fn: Noise schedule; together with ``beta_fn`` enables input
                standardisation.
            beta_fn: Data schedule; together with ``gamma_fn`` enables input
                standardisation.
            rngs: Random-number generators for parameter initialisation.
        """
        self.gamma_fn = gamma_fn
        self.beta_fn = beta_fn
        self.linear = nnx.Linear(
            input_dim, network_dims.embedding_dim, dtype=network_dims.dtype, rngs=rngs
        )
        pos_init = nnx.initializers.normal(stddev=0.02, dtype=network_dims.dtype)
        self.positional_encoding = nnx.Param(
            pos_init(rngs.params(), (num_tokens, network_dims.embedding_dim))
        )

    def __call__(self, x: Var, t: Time | None = None) -> Var:
        """Encode input with optional input standardisation.

        Args:
            x: Input array, shape (L, input_dim) or (input_dim,).
            t: Time value, required when beta_fn and gamma_fn are both set for
                standardisation.
        """
        # Optional input standardisation (before encoding — order matters due to bias)
        if self.beta_fn is not None and self.gamma_fn is not None and t is not None:
            beta_t = self.beta_fn(t)
            gamma_t = self.gamma_fn(t)
            scale = jnp.sqrt(beta_t**2 + gamma_t**2)
            scale = jnp.clip(scale, min=1e-4)
            x = x / scale

        if x.ndim == 1:
            x = x[None, :]  # (d,) → (1, d)
        return self.linear(x) + self.positional_encoding[...]
