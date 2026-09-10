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

from stix.core.generator import Generator, Velocity, VelocityAndScore
from stix.core.interpolant.interpolant import Interpolant, OneSidedInterpolant
from stix.core.utils import check_gamma_is_not_zero
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


class ContinuousInterpolant(Interpolant):
    r"""A base class for continuous interpolants, following the Stochastic Interpolants framework of [Albergo et al. 2024](https://www.jmlr.org/papers/v26/23-1605.html).

    .. math::
        z_t = J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}) + \gamma_t\,\epsilon,
        \qquad \epsilon \sim \mathcal{N}(0, I).

    Concrete subclasses implement :meth:`interpolant_fn`, which returns the deterministic part of the interpolant :math:`J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}})`,
    and are initialized with a noise schedule function :math:`\gamma_t`.
    """

    gamma_fn: Callable[[Time], Scalar] | None

    def __init__(
        self,
        gamma_fn: Callable[[Time], Scalar] | None = None,
    ):
        r"""Initialise the continuous interpolant.

        Args:
            gamma_fn: The noise schedule function :math:`\gamma_t`. If ``None``, the interpolant is
                deterministic (:math:`\gamma_t = 0`).
        """
        self.gamma_fn = gamma_fn

    @abstractmethod
    def interpolant_fn(
        self, embedded_pairs: EmbeddedSourceTargetPair, t: Time
    ) -> EmbeddedVar:
        r"""Compute the deterministic part of the interpolant.

        .. math::
            J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}).

        Args:
            embedded_pairs: Embedded source-target pair :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            t: Interpolation time.

        Returns:
            The deterministic part of the interpolant at time :math:`t`.
        """
        ...

    def sample_noise(self, key: PRNGKeyArray, shape: Shape) -> NoiseVar:
        r"""Sample standard Gaussian noise :math:`\epsilon \sim \mathcal{N}(0, I)`.

        Args:
            key: PRNG key used to draw the noise.
            shape: Shape of the noise variable to draw.

        Returns:
            A Gaussian noise sample of shape ``shape``.
        """
        return jr.normal(key, shape)

    def interpolate(
        self, embedded_pairs: EmbeddedSourceTargetPair, t: Time, epsilon: NoiseVar
    ) -> EmbeddedVar:
        r"""Compute the full interpolation including noise.

        .. math::
            z_t = J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}) + \gamma_t\,\epsilon.

        Args:
            embedded_pairs: Embedded source-target pair :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            t: Interpolation time.
            epsilon: The interpolant's noise variable :math:`\epsilon`.

        Returns:
            The interpolated state :math:`z_t`.
        """
        gamma_val = self.gamma_fn(t) if self.gamma_fn is not None else 0.0
        z_t = self.interpolant_fn(embedded_pairs, t) + gamma_val * epsilon
        return z_t

    def get_conditional_velocity_field(
        self,
        embedded_pairs: EmbeddedSourceTargetPair,
        t: Time,
        epsilon: NoiseVar,
    ) -> Var:
        r"""Compute the conditional velocity of the deterministic interpolation.

        .. math::
            v(t, z_t \mid z_{\mathrm{src}}, z_{\mathrm{tgt}}) = \partial_t J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}).

        Args:
            embedded_pairs: Embedded source-target pair :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            t: Interpolation time.
            epsilon: The interpolant's noise variable. Unused; accepted so the
                signature matches :meth:`get_conditional_velocity`.

        Returns:
            The time derivative of the deterministic interpolation at :math:`t`.
        """
        # argnums=1 targets the 2nd index of the arguments of interpolant_fn, i.e. t.
        partial_t_interpolant_fn = jax.jacfwd(self.interpolant_fn, argnums=1)(
            embedded_pairs, t
        )
        return partial_t_interpolant_fn

    def get_conditional_velocity(
        self,
        embedded_pairs: EmbeddedSourceTargetPair,
        t: Time,
        epsilon: NoiseVar,
    ) -> Var:
        r"""Compute the conditional velocity of the full (noisy) interpolation.

        .. math::
            b(t, z_t \mid z_{\mathrm{src}}, z_{\mathrm{tgt}})
            = \partial_t J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}) + \dot{\gamma}(t)\,\epsilon.

        Args:
            embedded_pairs: Embedded source-target pair :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            t: Interpolation time.
            epsilon: The interpolant's noise variable.

        Returns:
            The conditional velocity at time ``t``.
        """
        partial_t_interpolate = jax.jacfwd(self.interpolate, argnums=1)(
            embedded_pairs, t, epsilon
        )
        return partial_t_interpolate


class ContinuousStochasticInterpolant(ContinuousInterpolant, Interpolant):
    r"""Continuous-state stochastic interpolant.

    .. math::
        z_t = J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}) + \gamma_t\,\epsilon,
        \qquad \epsilon \sim \mathcal{N}(0, I),

    with :math:`\gamma_t` non-zero for :math:`t` in :math:`(0, 1)`.
    """

    gamma_fn: Callable[[Time], Scalar]
    generator_type: type[Generator] = VelocityAndScore

    def __init__(
        self,
        gamma_fn: Callable[[Time], Scalar],
    ):
        r"""Initialise the continuous stochastic interpolant.

        Args:
            gamma_fn: The noise schedule function :math:`\gamma_t`. Must be non-zero for :math:`t` in :math:`(0,1)`.
        """
        ContinuousInterpolant.__init__(self, gamma_fn=gamma_fn)
        if gamma_fn is None:
            raise ValueError(
                "ContinuousStochasticInterpolant requires a gamma function."
            )
        if not check_gamma_is_not_zero(gamma_fn):
            raise ValueError(
                "ContinuousStochasticInterpolant requires non-zero gamma_t for t in "
                "(0,1). Use a ContinuousDeterministicInterpolant if you want "
                "gamma_t = 0."
            )

    def get_conditional_score(
        self,
        embedded_pairs: EmbeddedSourceTargetPair,
        t: Time,
        epsilon: NoiseVar,
    ) -> Var:
        r"""Compute the conditional score.

        .. math::
            s(t, z_t \mid z_{\mathrm{src}}, z_{\mathrm{tgt}}) = -\epsilon / \gamma_t.

        Args:
            embedded_pairs: Embedded source-target pair :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`. Unused;
                accepted for signature parity with the velocity helpers.
            t: Interpolation time.
            epsilon: The interpolant's noise variable :math:`\epsilon`.

        Returns:
            The conditional score at time :math:`t`.
        """
        return -epsilon / self.gamma_fn(t)


class ContinuousDeterministicInterpolant(ContinuousInterpolant):
    r"""A base class for the generic deterministic interpolant.

    .. math::
        z_t = J_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}).
    """

    generator_type: type[Generator] = Velocity

    def __init__(self):
        r"""Initialise the deterministic interpolant."""
        ContinuousInterpolant.__init__(self, gamma_fn=None)


class ContinuousOneSidedStochasticInterpolant(
    ContinuousStochasticInterpolant, OneSidedInterpolant
):
    r"""Continuous one-sided stochastic interpolant.

    .. math::
        z_t = J_t(z_{\mathrm{tgt}}) + \gamma_t\,\epsilon.

    with :math:`\gamma_t` non-zero for :math:`t` in :math:`[0, 1)`.
    Note that :math:`\gamma_{t=0}` must be non-zero to ensure that
    the initial state :math:`z_0` is not deterministic.
    """

    def __init__(
        self,
        gamma_fn: Callable[[Time], Scalar],
    ) -> None:
        r"""Initialise the continuous one-sided stochastic interpolant.

        Args:
            gamma_fn: The noise schedule :math:`\gamma_t`. Must be non-zero for
                :math:`t` in :math:`[0, 1)`.
        """
        if float(gamma_fn(jnp.array(0.0))) == 0.0:
            raise ValueError(
                "ContinuousOneSidedStochasticInterpolant requires non-zero "
                "gamma_t at t=0."
            )
        ContinuousStochasticInterpolant.__init__(self, gamma_fn=gamma_fn)

    def interpolant_fn(
        self, embedded_pairs: EmbeddedSourceTargetPair, t: Time
    ) -> EmbeddedVar:
        r"""Compute the deterministic interpolant using only the embedded target.

        Args:
            embedded_pairs: Embedded source-target pair :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`. Only
                ``target`` is used.
            t: Interpolation time.

        Returns:
            The deterministic interpolation at time :math:`t`.
        """
        embedded_target = embedded_pairs.target
        return self._interpolant_fn(embedded_target, t)

    @abstractmethod
    def _interpolant_fn(self, embedded_target: EmbeddedVar, t: Time) -> EmbeddedVar:
        r"""Compute the deterministic one-sided interpolant in terms of the target only.

        Args:
            embedded_target: The embedded target :math:`z_{\mathrm{tgt}}`.
            t: Interpolation time.

        Returns:
            The deterministic interpolation at time ``t``.
        """
        ...
