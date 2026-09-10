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

"""EDM-style variance-exploding diffusion interpolant."""

import jax.numpy as jnp

from stix.core.interpolant.linear_interpolant import (
    OneSidedLinearStochasticInterpolant,
)


class VarianceExplodingDiffusionOneSidedInterpolant(
    OneSidedLinearStochasticInterpolant
):
    r"""One-sided variance exploding diffusion interpolant.

    :math:`\beta_t = 1`; :math:`\gamma_t` interpolates linearly between
    ``sigma_max`` at :math:`t = 0` and ``sigma_min`` at :math:`t = 1`.
    """

    def __init__(self, sigma_min: float = 2e-3, sigma_max: float = 80.0) -> None:
        r"""Initialise the diffusion interpolant with the noise-scale endpoints.

        Args:
            sigma_min: Noise scale at :math:`t = 1`.
            sigma_max: Noise scale at :math:`t = 0`.
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(
            beta_fn=lambda t: jnp.ones_like(t),
            gamma_fn=lambda t: sigma_max * (1 - t) + t * sigma_min,
        )
