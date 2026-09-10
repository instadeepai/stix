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

from flax import struct

from stix.typing import Var


class Generator(struct.PyTreeNode):
    r"""Base dataclass describing the infinitesimal generator of the Markov process simulated at sampling time for a single modality.

    It carries no behaviour. What the solver does with it (an Euler ODE step, an
    Euler-Maruyama SDE step, or a continuous-time-Markov-chain jump) is decided
    entirely by the solver, keyed on the concrete :class:`Generator` subclass. See
    :mod:`stix.sampling.steps` for the step/drift/diffusion dispatch tables.

    Each interpolant declares its associated generator variant via its
    ``generator_type`` class attribute; modalities expose that as
    :attr:`~stix.core.modality.Modality.generator_type`. The generators returned by
    :meth:`~stix.core.gen_model.GenerativeModel.get_generator` must match
    those declared for each modality.

    This is a frozen ``flax.struct`` pytree node, so a per-modality ``PyTree[Generator]``
    flows through ``jax.tree.map`` / :meth:`stix.core.modality.ModalityRegistry.map`
    and diffrax ``args`` unchanged, with its array leaves traced.

    Subclasses only add array fields; the solver interprets those fields.
    """


class Velocity(Generator):
    r"""A deterministic drift :math:`b_t` (an ODE generator).

    Attributes:
        velocity: The per-modality velocity field :math:`b_t` at the current
            state and time.
    """

    velocity: Var


class VelocityAndScore(Generator):
    r"""A drift :math:`b_t` together with a score :math:`s_t`.

    Supports both ODE sampling and SDE sampling (the
    score enters the drift correction and sets the diffusion coefficient).

    Attributes:
        velocity: The per-modality velocity field :math:`b_t`.
        score: The per-modality score :math:`s_t = \nabla_{z_t} \log p_t(z_t)`.
    """

    velocity: Var
    score: Var


class TransitionRates(Generator):
    r"""Forward and backward transition rates of a CTMC generator.

    ``forward_rates`` holds the generative probability velocity :math:`\hat u_t`
    out of the current discrete state :math:`z_t`. ``backward_rates`` holds the
    time-reversed velocity :math:`\check u_t`, which walks the same path of
    marginals with *decreasing* :math:`t` and is what
    :meth:`~stix.core.interpolant.DiscreteInterpolant.rates_from_mixture_distributions`
    returns for ``backward=True``. Both are valid transition rates, and both
    feed the corrector mix
    :math:`\bar u_t = (1 + \lambda_t)\,\hat u_t + \lambda_t\,\check u_t`.
    Each trailing axis has length
    equal to the number of states. The diagonal entry (the rate to stay,
    :math:`y = z`) is conventionally the negative sum of the off-diagonal rates.

    Supports continuous-time Markov chains (CTMCs) sampling.

    Attributes:
        forward_rates: Generative rates :math:`\hat u_t` out of the current
            state, shape ``(..., num_states)``.
        backward_rates: Time-reversed rates :math:`\check u_t`, same shape.
    """

    forward_rates: Var
    backward_rates: Var
