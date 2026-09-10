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

from typing import Callable

import jax
import jax.numpy as jnp

from stix.core.interpolant.discrete_interpolant import DiscreteInterpolant
from stix.core.interpolant.interpolant import OneSidedInterpolant
from stix.typing import (
    EmbeddedSourceTargetPair,
    EmbeddedVar,
    NoiseVar,
    Scalar,
    Time,
    Var,
)


def _two_point_kappa(
    scalar_fn: Callable[[Time], Scalar],
) -> Callable[[Time], Var]:
    r"""Lift a scalar target weight :math:`\kappa_t` to :math:`(\kappa_t, 1-\kappa_t)`."""

    def vector_fn(t: Time) -> Var:
        kappa = jnp.asarray(scalar_fn(t))
        return jnp.stack([kappa, 1.0 - kappa])

    return vector_fn


class MaskDiscreteInterpolant(DiscreteInterpolant, OneSidedInterpolant):
    r"""Discrete interpolant for masked diffusion.

    The interpolant path is given by

    .. math::
        p_t(z \mid z_{\mathrm{tgt}}) = \kappa_t \delta_{z_{\mathrm{tgt}}} + (1-\kappa_t) \delta_{m},

    where :math:`\kappa_t` is the schedule and :math:`m` is the mask symbol.

    The state space has ``K+1`` states (``K`` data categories + 1 mask state), and the mask index is ``K``.

    Initial state :math:`z_0` is the mask state.
    """

    def __init__(
        self,
        num_categories: int,
        kappa_fn: Callable[[Time], Scalar],
    ) -> None:
        r"""Initialise masked diffusion with ``K + 1`` CTMC states.

        Args:
            num_categories: Number of data categories ``K``.
            kappa_fn: Scalar target weight :math:`\kappa_t`, expanded to
                :math:`(\kappa_t, 1-\kappa_t)`.
        """
        super().__init__(
            num_categories=num_categories,
            num_states=num_categories + 1,
            num_components=2,
            kappa_fn=_two_point_kappa(kappa_fn),
        )

    @property
    def mask_index(self) -> int:
        """The dedicated mask symbol's index, ``num_categories``.

        Returns:
            The mask symbol index ``num_categories``.
        """
        return self.num_categories

    def mixture_conditionals(
        self,
        embedded_pairs: EmbeddedSourceTargetPair,
    ) -> Var:
        r"""Computes the mixture conditional distributions from the embedded pair.

        The mixture conditional distributions are given by

        .. math::
            w_1(\cdot \mid z_{\mathrm{tgt}}) = \delta_{z_{\mathrm{tgt}}}
            \quad \text{and} \quad
            w_2(\cdot \mid z_{\mathrm{tgt}}) = \delta_{m}.

        Args:
            embedded_pairs: Embedded pair; only
                :math:`z_{\mathrm{tgt}}` is used.

        Returns:
            Stacked laws of shape ``(..., 2, num_states)``.
        """
        z_tgt = embedded_pairs.target
        indices = jnp.full(z_tgt.shape[:-1], self.mask_index, dtype=jnp.int32)
        mask = jax.nn.one_hot(indices, z_tgt.shape[-1], dtype=z_tgt.dtype)
        return jnp.stack([z_tgt, mask], axis=-2)

    def mixture_distributions_from_target_posterior(
        self, target_posterior: Var, z_t: EmbeddedVar
    ) -> Var:
        r"""Computes the mixture distributions from the target posterior :math:`\hat{p}_{\mathrm{tgt}\mid t}(z_{\mathrm{tgt}} \mid z_{t})`.

        The mixture distributions are given by

        .. math::
            \hat{w}_1 = \begin{cases}
                \hat{p}_{\mathrm{tgt}\mid t}(z_{\mathrm{tgt}} \mid z_{t}) & \text{if } z_{t} = m, \\
                \delta_{m} & \text{otherwise}.
            \end{cases}
            \quad \text{and} \quad
            \hat{w}_2 = \delta_{m},

        where the posterior probability :math:`\hat{p}_{\mathrm{tgt}\mid t}(z_{\mathrm{tgt}} \mid z_{t})` is extended
        with a zero probability for the mask state.

        Args:
            target_posterior: Target denoising posterior :math:`\hat{p}_{\mathrm{tgt}\mid t}(z_{\mathrm{tgt}} \mid z_{t})` over
                the ``K`` data categories, at :math:`(t, z_{t})`. Shape: ``(..., num_states - 1)``.
            z_t: Embedded current state. Shape: ``(..., num_states)``.

        Returns:
            Stacked posteriors of shape ``(..., 2, num_states)``.
        """
        zeros = jnp.zeros(
            (*target_posterior.shape[:-1], 1), dtype=target_posterior.dtype
        )
        p_learned = jnp.concatenate([target_posterior, zeros], axis=-1)
        is_masked = z_t[..., self.mask_index : self.mask_index + 1]
        p_tgt = is_masked * p_learned + (1.0 - is_masked) * z_t.astype(p_learned.dtype)
        indices = jnp.full(
            target_posterior.shape[:-1], self.mask_index, dtype=jnp.int32
        )
        p_src = jax.nn.one_hot(indices, self.num_states, dtype=target_posterior.dtype)
        return jnp.stack([p_tgt, p_src], axis=-2)

    def rates_from_target_posterior(
        self,
        target_posterior: Var,
        z_t: EmbeddedVar,
        t: Time,
        *,
        backward: bool = False,
    ) -> Var:
        r"""Computes the unconditional CTMC rates from a learned target posterior.

        Applies
        :meth:`~stix.core.interpolant.DiscreteInterpolant.rates_from_mixture_distributions`
        to
        :meth:`mixture_distributions_from_target_posterior`. For the two-point
        schedule :math:`\kappa_t = (\kappa_t, 1-\kappa_t)` the forward rates are
        :math:`\hat u_t = \dot\kappa_t/(1-\kappa_t)\,(p_{\mathrm{tgt}\mid t} - \delta_z)`.

        Args:
            target_posterior: Target denoising posterior :math:`p_{\mathrm{tgt}\mid t}` over
                the ``K`` data categories. Shape: ``(..., num_states - 1)``.
            z_t: Embedded current state. Shape: ``(..., num_states)``.
            t: Interpolation time.
            backward: If ``True``, return the time-reversed rates. Defaults to
                ``False``.

        Returns:
            Transition rates of shape ``(..., num_states)``.
        """
        distributions = self.mixture_distributions_from_target_posterior(
            target_posterior, z_t
        )
        return self.rates_from_mixture_distributions(
            distributions, z_t, t, backward=backward
        )

    def sample_initial_state(self, z_src: EmbeddedVar | None, epsilon: NoiseVar) -> Var:
        r"""Draw :math:`z_0`: every position starts on the mask symbol.

        Args:
            z_src: Unused (one-sided).
            epsilon: Packed :math:`(\epsilon_m, \epsilon_s)`; supplies shape
                and dtype.

        Returns:
            One-hot initial state with every position at the mask index.
        """
        del z_src
        indices = jnp.full(epsilon.shape[:-1], self.mask_index, dtype=jnp.int32)
        return jax.nn.one_hot(indices, self.num_states, dtype=epsilon.dtype)


class UniformDiscreteInterpolant(DiscreteInterpolant, OneSidedInterpolant):
    r"""Discrete interpolant for uniform diffusion.

    The interpolant path is given by

    .. math::
        p_t(z \mid z_{\mathrm{tgt}}) = \kappa_t \delta_{z_{\mathrm{tgt}}} + (1-\kappa_t)\,\mathrm{Unif}(\{0,\dots,K-1\}),

    where :math:`\kappa_t` is the schedule.

    The state space has exactly ``K`` states (one state per data category).

    Initial state :math:`z_0` is drawn uniformly from the data categories.
    """

    def __init__(
        self,
        num_categories: int,
        kappa_fn: Callable[[Time], Scalar],
    ) -> None:
        r"""Initialise uniform diffusion with ``K`` CTMC states.

        Args:
            num_categories: Number of data categories ``K``.
            kappa_fn: Scalar target weight :math:`\kappa_t`, expanded to
                :math:`(\kappa_t, 1-\kappa_t)`.
        """
        super().__init__(
            num_categories=num_categories,
            num_states=num_categories,
            num_components=2,
            kappa_fn=_two_point_kappa(kappa_fn),
        )

    def mixture_conditionals(
        self,
        embedded_pairs: EmbeddedSourceTargetPair,
    ) -> Var:
        r"""Computes the mixture conditional distributions from the embedded pair.

        The mixture conditional distributions are given by

        .. math::
            w_1(\cdot \mid z_{\mathrm{tgt}}) = \delta_{z_{\mathrm{tgt}}}
            \quad \text{and} \quad
            w_2(\cdot \mid z_{\mathrm{tgt}}) = \mathrm{Unif}(\{0,\dots,K-1\}).

        Args:
            embedded_pairs: Embedded pair; only
                :math:`z_{\mathrm{tgt}}` is used.

        Returns:
            Stacked laws of shape ``(..., 2, num_states)``.
        """
        z_tgt = embedded_pairs.target
        p_src = jnp.full_like(z_tgt, 1.0 / self.num_categories)
        return jnp.stack([z_tgt, p_src], axis=-2)

    def mixture_distributions_from_target_posterior(
        self, target_posterior: Var, z_t: EmbeddedVar
    ) -> Var:
        r"""Computes the mixture distributions from the target posterior :math:`\hat{p}_{\mathrm{tgt}\mid t}(z_{\mathrm{tgt}} \mid z_{t})`.

        The mixture distributions are given by

        .. math::
            \hat{w}_1 = \hat{p}_{\mathrm{tgt}\mid t}(z_{\mathrm{tgt}} \mid z_{t})
            \quad \text{and} \quad
            \hat{w}_2 = \mathrm{Unif}(\{0,\dots,K-1\}),

        where the current state :math:`z_t` does not enter the closed-form source
        atom.

        Args:
            target_posterior: Target denoising posterior :math:`\hat{p}_{\mathrm{tgt}\mid t}(z_{\mathrm{tgt}} \mid z_{t})` over
                the ``K`` data categories, at :math:`(t, z_{t})`. Shape: ``(..., num_states)``.
            z_t: Embedded current state. Shape: ``(..., num_states)``.

        Returns:
            Stacked posteriors of shape ``(..., 2, num_states)``.
        """
        del z_t
        p_src = jnp.full_like(target_posterior, 1.0 / self.num_categories)
        return jnp.stack([target_posterior, p_src], axis=-2)

    def rates_from_target_posterior(
        self,
        target_posterior: Var,
        z_t: EmbeddedVar,
        t: Time,
        *,
        backward: bool = False,
    ) -> Var:
        r"""Computes the unconditional CTMC rates from a learned target posterior.

        Applies
        :meth:`~stix.core.interpolant.DiscreteInterpolant.rates_from_mixture_distributions`
        to
        :meth:`mixture_distributions_from_target_posterior`. For the two-point
        schedule :math:`\kappa_t = (\kappa_t, 1-\kappa_t)` the forward rates are
        :math:`\hat u_t = \dot\kappa_t/(1-\kappa_t)\,(p_{\mathrm{tgt}\mid t} - \delta_z)`.

        Args:
            target_posterior: Target denoising posterior :math:`p_{\mathrm{tgt}\mid t}` over
                the ``K`` data categories. Shape: ``(..., num_states)``.
            z_t: Embedded current state. Shape: ``(..., num_states)``.
            t: Interpolation time.
            backward: If ``True``, return the time-reversed rates. Defaults to
                ``False``.

        Returns:
            Transition rates of shape ``(..., num_states)``.
        """
        distributions = self.mixture_distributions_from_target_posterior(
            target_posterior, z_t
        )
        return self.rates_from_mixture_distributions(
            distributions, z_t, t, backward=backward
        )

    def sample_initial_state(self, z_src: EmbeddedVar | None, epsilon: NoiseVar) -> Var:
        r"""Draw :math:`z_0` by sampling the uniform source atom.

        At :math:`t=0` the schedule selects the uniform component; categories are
        drawn with :math:`\epsilon_s`.

        Args:
            z_src: Unused (one-sided).
            epsilon: Packed :math:`(\epsilon_m, \epsilon_s)`.

        Returns:
            One-hot initial state over ``K`` categories.
        """
        del z_src
        epsilon_s = epsilon[..., 1]
        p_uniform = 1.0 / self.num_categories
        selected = jnp.full(
            (*epsilon.shape[:-1], self.num_states), p_uniform, dtype=epsilon.dtype
        )
        cdf = jnp.cumsum(selected, axis=-1)
        state_index = jnp.sum(epsilon_s[..., None] >= cdf, axis=-1)
        state_index = jnp.clip(state_index, 0, self.num_states - 1).astype(jnp.int32)
        return jax.nn.one_hot(state_index, self.num_states, dtype=epsilon.dtype)
