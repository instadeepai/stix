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


from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from stix.core.generator import Generator, TransitionRates
from stix.core.interpolant.interpolant import Interpolant
from stix.typing import (
    EmbeddedSourceTargetPair,
    EmbeddedVar,
    NoiseVar,
    PRNGKeyArray,
    Scalar,
    Shape,
    Time,
    Var,
)

_KAPPA_EPS = 1e-8


class DiscreteInterpolant(Interpolant):
    r"""A base class for discrete interpolants, following the Discrete Flow Matching framework of [Gat et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html5).

    These interpolants yield the mixture marginals

    .. math::
        p_t(z \mid z_{\mathrm{src}}, z_{\mathrm{tgt}})
        = \sum_{j=1}^{M} \kappa_t^j w_j(z \mid z_{\mathrm{src}}, z_{\mathrm{tgt}}),

    where :math:`\kappa_t = (\kappa_t^j)_{j=1}^{M} \in \Delta^{M-1}` are the mixture weights schedule and
    :math:`w_j(z \mid z_{\mathrm{src}}, z_{\mathrm{tgt}})` are the mixture
    conditionals.

    The corresponding interpolant map is given by

    .. math::
        I_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}, \epsilon)
        = s_J,
        \qquad
        s_J \overset{\epsilon_s}{\sim} w_J(\,\cdot\mid z_{\mathrm{src}}, z_{\mathrm{tgt}}),
        \qquad
        J \sim \mathrm{Cat}(\kappa_t)\ \text{via}\ \epsilon_m,

    with :math:`\epsilon = (\epsilon_m, \epsilon_s)` and
    :math:`\kappa_t = (\kappa_t^j)_{j=1}^{M}`. Here :math:`\epsilon_m` is used to draw the *mixture index* :math:`J` and :math:`\epsilon_s` is used to sample the *conditional state* :math:`s_J` from the *mixture conditional* :math:`w_J`.

    Concrete subclasses implement :meth:`mixture_conditionals` (the laws
    :math:`w_j`) and are initialised with a schedule :math:`\kappa_t`.
    """

    generator_type: type[Generator] = TransitionRates

    def __init__(
        self,
        num_categories: int,
        num_states: int,
        num_components: int,
        kappa_fn: Callable[[Time], Var],
    ) -> None:
        r"""Initialise with category count, CTMC width, and vector schedule.

        Args:
            num_categories: Number of data categories ``K`` (indices ``0..K-1``).
            num_states: Width of the one-hot / rate axis (typically ``K`` or ``K + 1``).
            num_components: Number of mixture distributions :math:`M`.
            kappa_fn: The schedule :math:`\kappa_t \in \Delta^{M-1}` of mixture weights.
        """
        self.num_categories = num_categories
        self.num_states = num_states
        self.num_components = num_components
        self.kappa_fn = kappa_fn

    def sample_noise(self, key: PRNGKeyArray, shape: Shape) -> NoiseVar:
        r"""Sample packed :math:`(\epsilon_m, \epsilon_s)`, both i.i.d. :math:`\mathrm{Unif}[0, 1)`.

        Args:
            key: PRNG key used to draw :math:`\epsilon = (\epsilon_m, \epsilon_s)`.
            shape: Embedded state shape ``(..., num_states)``.

        Returns:
            Noise of shape ``(*shape[:-1], 2)``.
        """
        return jr.uniform(key, (*shape[:-1], 2))

    def draw_mixing_index(self, epsilon_m: Var, t: Time) -> Var:
        r"""Inverse-CDF draw :math:`J \in \{0,\dots,M-1\}` with masses :math:`\kappa_t` from uniform noise :math:`\epsilon_m\sim \mathrm{Unif}[0, 1)`.

        Args:
            epsilon_m: Per-position uniforms :math:`\epsilon_m\sim \mathrm{Unif}[0, 1)`.
            t: Interpolation time.

        Returns:
            Integer indices of shape ``epsilon_m.shape``.
        """
        kappa = jnp.asarray(self.kappa_fn(t))
        cdf = jnp.cumsum(kappa)
        index = jnp.sum(epsilon_m[..., None] >= cdf, axis=-1)
        return jnp.clip(index, 0, self.num_components - 1).astype(jnp.int32)

    @abstractmethod
    def mixture_conditionals(
        self,
        embedded_pairs: EmbeddedSourceTargetPair,
    ) -> Var:
        r"""Mixture conditionals :math:`w_j(\cdot \mid z_{\mathrm{src}}, z_{\mathrm{tgt}})`.

        Args:
            embedded_pairs: Embedded pair
                :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.

        Returns:
            Stacked categorical laws of shape ``(..., M, num_states)``.
        """
        ...

    def interpolate(
        self,
        embedded_pairs: EmbeddedSourceTargetPair,
        t: Time,
        epsilon: NoiseVar,
    ) -> EmbeddedVar:
        r"""The interpolant :math:`I_t = s_J`.

        Draws :math:`J` from :math:`\kappa_t` with :math:`\epsilon_m`, then
        samples the selected conditional :math:`w_J` with :math:`\epsilon_s`.

        Args:
            embedded_pairs: Embedded pair
                :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            t: Interpolation time.
            epsilon: Packed :math:`(\epsilon_m, \epsilon_s)`.

        Returns:
            The interpolated one-hot state :math:`z_t`.
        """
        conditionals = self.mixture_conditionals(
            embedded_pairs
        )  # (*dims, M, num_states)
        epsilon_m = epsilon[..., 0]
        index = self.draw_mixing_index(epsilon_m, t)  # (*dims,)
        index = index[..., None, None]  # (*dims, 1, 1)
        # gather the selected mixture conditionals
        selected_conditionals = jnp.take_along_axis(
            conditionals, index, axis=-2
        )  # (*dims, 1, num_states)
        selected_conditionals = selected_conditionals.squeeze(
            axis=-2
        )  # (*dims, num_states)
        # sample states from the selected mixture conditional, using the inverse CDF method
        epsilon_s = epsilon[..., 1]
        cdf = jnp.cumsum(selected_conditionals, axis=-1)  # (*dims, num_states)
        state_index = jnp.sum(epsilon_s[..., None] >= cdf, axis=-1)  # (*dims,)
        state_index = jnp.clip(state_index, 0, self.num_states - 1).astype(jnp.int32)
        return jax.nn.one_hot(
            state_index, self.num_states, dtype=selected_conditionals.dtype
        )  # (*dims, num_states)

    @staticmethod
    def _rate_coefficients(kappa: Var, kappa_dot: Var) -> tuple[Var, Scalar]:
        r"""CTMC rate coefficients :math:`(a_t, b_t)` for a given :math:`\dot\kappa`.

        .. math::
            a^j_t = \dot\kappa^j_t - \kappa^j_t\,\dot\kappa^\ell_t / \kappa^\ell_t,
            \qquad
            b_t = \dot\kappa^\ell_t / \kappa^\ell_t,
            \qquad
            \ell = \arg\min_j \dot\kappa^j_t / \kappa^j_t.

        Args:
            kappa: The schedule :math:`\kappa_t`.
            kappa_dot: The derivative of the schedule :math:`\kappa_t`.

        Returns:
            The vector :math:`a_t` of shape ``(M,)`` and the scalar :math:`b_t`.
        """
        ratio = kappa_dot / jnp.clip(kappa, min=_KAPPA_EPS)
        ell = jnp.argmin(ratio)
        b = kappa_dot[ell] / jnp.clip(kappa[ell], min=_KAPPA_EPS)
        a = kappa_dot - kappa * b
        return a, b

    def rates_from_mixture_distributions(
        self,
        distributions: Var,
        z_t: EmbeddedVar,
        t: Time,
        *,
        backward: bool = False,
    ) -> Var:
        r"""Computes the CTMC rates from stacked mixture distributions.

        .. math::
            u_t = \sum_{j=1}^{M} a^j_t\, w^j + b_t\,\delta_{z_t},

        with :math:`(a_t, b_t)` from ``_rate_coefficients``. The same
        formula is the *conditional* rates when ``distributions`` are the
        mixture conditionals :math:`w_j(\cdot\mid z_{\mathrm{src}},
        z_{\mathrm{tgt}})`, and the *unconditional* rates when they are the
        posteriors :math:`\hat w^j_t`.

        ``backward=False`` (default) is the generative forward rates (t:0→1).
        ``backward=True`` builds the time-reversed CTMC that runs the same
        mixture path backwards in :math:`t` (t:1→0): substitute
        :math:`\dot\kappa \to -\dot\kappa` in :math:`(a_t, b_t)`. Either branch
        gives valid transition rates, non-negative away from :math:`z_t` and
        summing to zero.

        Args:
            distributions: Stacked mixture laws of shape ``(..., M, num_states)``.
            z_t: Embedded current state. Shape: ``(..., num_states)``.
            t: Interpolation time.
            backward: If ``True``, return the time-reversed rates. Defaults to ``False``.

        Returns:
            Transition rates of shape ``(..., num_states)``.
        """
        kappa = jnp.asarray(self.kappa_fn(t))
        kappa_dot = jnp.asarray(jax.jacfwd(self.kappa_fn)(t))
        signed_dot = -kappa_dot if backward else kappa_dot
        a, b = self._rate_coefficients(kappa, signed_dot)
        weighted = jnp.einsum("...ms,m->...s", distributions, a)
        return weighted + b * z_t.astype(distributions.dtype)

    def get_conditional_rates(
        self,
        embedded_pairs: EmbeddedSourceTargetPair,
        z_t: EmbeddedVar,
        t: Time,
        *,
        backward: bool = False,
    ) -> Var:
        r"""Conditional CTMC rates given the source-target pair.

        Applies :meth:`rates_from_mixture_distributions` to
        :meth:`mixture_conditionals`.

        Args:
            embedded_pairs: Embedded pair
                :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            z_t: Embedded current state. Shape: ``(..., num_states)``.
            t: Interpolation time.
            backward: If ``True``, return the time-reversed rates. Defaults to ``False``.

        Returns:
            Conditional rates of shape ``(..., num_states)``.
        """
        distributions = self.mixture_conditionals(embedded_pairs)
        return self.rates_from_mixture_distributions(
            distributions, z_t, t, backward=backward
        )
