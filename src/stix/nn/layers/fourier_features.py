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

"""Learned sinusoidal Fourier features."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from stix.typing import Scalar


class SinusoidalFourierFeatures(nnx.Module):
    """Learned Fourier features: cos(2π(x·w + b)).

    Encodes a scalar input into a vector of Fourier features with learned
    frequencies and phases, initialised from N(0, 1).
    """

    def __init__(
        self, num_features: int, *, dtype: jnp.dtype = jnp.float32, rngs: nnx.Rngs
    ):
        """Initialise learned frequencies and phases.

        Args:
            num_features: Number of Fourier features (and of learned ``w``, ``b``).
            dtype: Output dtype of the encoded features.
            rngs: Random-number generators for parameter initialisation.
        """
        self.dtype = dtype
        normal_init = nnx.initializers.normal(stddev=1.0, dtype=dtype)
        self.w = nnx.Param(normal_init(rngs.params(), (num_features,)))
        self.b = nnx.Param(normal_init(rngs.params(), (num_features,)))

    def __call__(self, x: Scalar) -> Array:
        """Encode a scalar into learned Fourier features.

        Args:
            x: Scalar input.

        Returns:
            A feature vector of shape ``(num_features,)``.
        """
        # Fourier features require FP32 precision for cos accuracy
        x = jnp.asarray(x, dtype=jnp.float32)
        w = self.w[...].astype(jnp.float32)
        b = self.b[...].astype(jnp.float32)
        return jnp.cos(2.0 * jnp.pi * (x * w + b)).astype(self.dtype)
