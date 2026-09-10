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

"""Fourier-feature time/noise context encoder."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from stix.nn.layers.fourier_features import SinusoidalFourierFeatures
from stix.nn.network_dims import NetworkDimsConfig
from stix.typing import Scalar, Time


class TimeNoiseContextEncoder(nnx.Module):
    """Encodes time t and noise level into a context vector via Fourier features.

    Pipeline: [t, 0.25·log(clamp(gamma_fn(t)))] → separate Fourier features
    → concatenate → Linear → SiLU → Linear → context vector.

    Encoding both t directly and log(gamma_t) ensures the model can distinguish
    symmetric time points (e.g. t=0.2 vs t=0.8 with gamma_fn=sqrt(2t(1-t))),
    which is critical for two-sided stochastic interpolants.

    NOTE: Assumes gamma_fn(t) > 0 for all t in (0, 1). Values below 1e-6
    are clamped before log, producing a floor at 0.25 * log(1e-6) ≈ −3.45.
    """

    def __init__(
        self,
        network_dims: NetworkDimsConfig,
        *,
        num_fourier_features: int = 16,
        gamma_fn: Callable[[Time], Scalar],
        rngs: nnx.Rngs,
    ):
        """Initialise time/noise context encoder with Fourier features.

        Args:
            network_dims: Shared network dimensions (uses ``context_dim`` and ``dtype``).
            num_fourier_features: Number of Fourier features per encoded scalar.
            gamma_fn: Noise schedule used for the log-gamma feature.
            rngs: Random-number generators for parameter initialisation.
        """
        self.gamma_fn = gamma_fn
        self.dtype = network_dims.dtype
        self.time_fourier = SinusoidalFourierFeatures(
            num_fourier_features, dtype=network_dims.dtype, rngs=rngs
        )
        self.noise_fourier = SinusoidalFourierFeatures(
            num_fourier_features, dtype=network_dims.dtype, rngs=rngs
        )
        # MLP input is the concatenation of both Fourier feature sets
        self.linear_1 = nnx.Linear(
            2 * num_fourier_features,
            network_dims.context_dim,
            dtype=network_dims.dtype,
            rngs=rngs,
        )
        self.linear_2 = nnx.Linear(
            network_dims.context_dim,
            network_dims.context_dim,
            dtype=network_dims.dtype,
            rngs=rngs,
        )

    def __call__(self, t: Time) -> Array:
        """Encode t into context vector of shape (context_dim,)."""
        # Direct time features (breaks symmetry for two-sided SI)
        time_features = self.time_fourier(jnp.asarray(t, dtype=jnp.float32))

        # Noise-level features (EDM-style log-gamma encoding)
        gamma_t = self.gamma_fn(t)
        log_gamma = 0.25 * jnp.log(
            jnp.clip(jnp.asarray(gamma_t, dtype=jnp.float32), min=1e-6)
        )
        noise_features = self.noise_fourier(log_gamma)

        fourier_features = jnp.concatenate([time_features, noise_features], axis=-1)
        h = jax.nn.silu(self.linear_1(fourier_features))
        return self.linear_2(h)
