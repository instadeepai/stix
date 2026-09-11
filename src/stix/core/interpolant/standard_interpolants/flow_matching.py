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

"""Flow-matching stochastic interpolants (one-sided and two-sided)."""

import jax.numpy as jnp

from stix.core.interpolant.linear_interpolant import (
    LinearDeterministicInterpolant,
    LinearStochasticInterpolant,
    OneSidedLinearStochasticInterpolant,
)


class FlowMatchingOneSidedInterpolant(OneSidedLinearStochasticInterpolant):
    r"""One-sided flow-matching interpolant: :math:`\beta_t = t`, :math:`\gamma_t = 1 - t`.

    Source is Gaussian noise; target is data.
    """

    def __init__(self) -> None:
        r"""Initialise the one-sided flow-matching interpolant."""
        super().__init__(
            beta_fn=lambda t: t,
            gamma_fn=lambda t: 1 - t,
        )


class FlowMatchingTwoSidedInterpolant(LinearDeterministicInterpolant):
    r"""Two-sided deterministic flow-matching interpolant.

    :math:`\alpha_t = 1 - t`, :math:`\beta_t = t`. Source and target are both
    data; no noise term.
    """

    def __init__(self) -> None:
        r"""Initialise the two-sided deterministic flow-matching interpolant."""
        super().__init__(
            alpha_fn=lambda t: 1 - t,
            beta_fn=lambda t: t,
        )


class StochasticFlowMatchingTwoSidedInterpolant(LinearStochasticInterpolant):
    r"""Two-sided stochastic flow-matching interpolant.

    :math:`\alpha_t = 1 - t`, :math:`\beta_t = t`,
    :math:`\gamma_t = \sqrt{2t(1-t)}`. Source and target are both data.
    """

    def __init__(self) -> None:
        r"""Initialise the two-sided stochastic flow-matching interpolant."""
        super().__init__(
            alpha_fn=lambda t: 1 - t,
            beta_fn=lambda t: t,
            gamma_fn=lambda t: jnp.sqrt(2 * t * (1 - t)),
        )
