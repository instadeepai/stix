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

"""Bayesian Flow Network interpolant."""

import jax.numpy as jnp

from stix.core.interpolant.linear_interpolant import (
    OneSidedLinearStochasticInterpolant,
)


class ContinuousBFNOneSidedInterpolant(OneSidedLinearStochasticInterpolant):
    r"""One-sided Bayesian Flow Network schedule (continuous data).

    With final noise level ``sigma_1``:

    .. math::
        \beta_t = 1 - \sigma_1^{2t}, \qquad
        \gamma_t = \sqrt{\beta_t (1 - \beta_t)}.

    :math:`\beta_t` is clipped to ``gamma_min`` so that :math:`\gamma_t > 0` at
    :math:`t=0`, matching the reference BFN implementation.
    """

    def __init__(self, sigma_1: float = 1e-2, gamma_min: float = 1e-6) -> None:
        r"""Initialise the BFN interpolant with final noise level ``sigma_1``.

        Args:
            sigma_1: Final noise level :math:`\sigma_1`.
            gamma_min: Lower clip on :math:`\beta_t` for numerical stability at
                :math:`t=0`.
        """
        self.sigma_1 = sigma_1
        self.gamma_min = gamma_min

        def beta_bfn(t):
            r"""BFN data schedule :math:`\beta_t` clipped to ``gamma_min``.

            Args:
                t: Interpolation time.
            """
            return jnp.clip(1 - sigma_1 ** (2 * t), min=gamma_min)

        super().__init__(
            beta_fn=beta_bfn,
            gamma_fn=lambda t: jnp.sqrt(beta_bfn(t) * (1 - beta_bfn(t))),
        )


class DiscreteBFNOneSidedInterpolant(OneSidedLinearStochasticInterpolant):
    r"""One-sided Bayesian Flow Network schedule (discrete data).

    Derived from the BFN discrete noise kernel
    :math:`y \sim \mathcal{N}(\beta(t) z_x, \beta(t)\, K\, I)` (Graves et al.
    2023), with ``num_classes`` classes (:math:`K`) and final accuracy
    ``beta_1``:

    .. math::
        \beta_t = t^2 \beta_1, \qquad
        \gamma_t = \sqrt{K\,\beta_t}.

    :math:`\beta_t` is clipped to ``beta_min`` so that :math:`\gamma_t > 0` at
    :math:`t=0`, matching the reference BFN implementation.
    """

    def __init__(
        self,
        num_classes: int,
        beta_1: float = 0.1,
        beta_min: float = 1e-6,
    ) -> None:
        r"""Initialise the discrete BFN interpolant.

        Args:
            num_classes: Number of discrete classes :math:`K`.
            beta_1: Final accuracy parameter :math:`\beta(1)`.
            beta_min: Lower clip on :math:`\beta_t` for numerical stability at
                :math:`t=0`.
        """
        self.num_classes = num_classes
        self.beta_1 = beta_1
        self.beta_min = beta_min

        def beta_bfn(t):
            r"""BFN data schedule :math:`\beta_t` clipped to ``beta_min``.

            Args:
                t: Interpolation time.
            """
            return jnp.clip(beta_1 * t**2, min=beta_min)

        super().__init__(
            beta_fn=beta_bfn,
            gamma_fn=lambda t: jnp.sqrt(num_classes * beta_bfn(t)),
        )
