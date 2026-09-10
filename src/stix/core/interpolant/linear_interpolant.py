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

from stix.core.interpolant.continuous_interpolant import (
    ContinuousDeterministicInterpolant,
    ContinuousInterpolant,
    ContinuousOneSidedStochasticInterpolant,
    ContinuousStochasticInterpolant,
)
from stix.typing import (
    EmbeddedSourceTargetPair,
    EmbeddedVar,
    NoiseVar,
    Scalar,
    Time,
    Var,
)


class LinearInterpolant(ContinuousInterpolant):
    r"""A base class for the generic linear interpolant.

    .. math::
        z_t = \alpha_t z_{\mathrm{src}} + \beta_t z_{\mathrm{tgt}} + \gamma_t \epsilon.
    """

    gamma_fn: Callable[[Time], Scalar] | None

    def __init__(
        self,
        gamma_fn: Callable[[Time], Scalar] | None,
        alpha_fn: Callable[[Time], Scalar] | None,
        beta_fn: Callable[[Time], Scalar],
    ):
        r"""Initialise the linear interpolant.

        Args:
            gamma_fn: Noise schedule :math:`\gamma_t`, or ``None`` for a
                deterministic interpolant.
            alpha_fn: Source schedule :math:`\alpha_t`. ``None`` for a
                one-sided interpolant with no explicit source.
            beta_fn: Target schedule :math:`\beta_t`.
        """
        ContinuousInterpolant.__init__(self, gamma_fn=gamma_fn)
        self.alpha_fn = alpha_fn
        self.beta_fn = beta_fn

    def interpolant_fn(
        self, embedded_pairs: EmbeddedSourceTargetPair, t: Time
    ) -> EmbeddedVar:
        r"""Compute the deterministic part of the interpolant.

        Args:
            embedded_pairs: Embedded source-target pair :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            t: Interpolation time.

        Returns:
            The deterministic interpolation at time ``t``.

        Raises:
            ValueError: If ``alpha_fn`` is set but the pair has no source.
        """
        if self.alpha_fn is None:
            return self.beta_fn(t) * embedded_pairs.target
        if embedded_pairs.source is None:
            raise ValueError(
                "Source is None for linear interpolant while alpha_fn is not None."
            )
        return (
            self.alpha_fn(t) * embedded_pairs.source
            + self.beta_fn(t) * embedded_pairs.target
        )

    def sample_initial_state(self, z_src: EmbeddedVar | None, epsilon: NoiseVar) -> Var:
        r"""Return :math:`\alpha(0)\,z_{\mathrm{src}} + \gamma(0)\,\epsilon`.

        Drops any leftover :math:`\beta(0)\,z_{\mathrm{tgt}}` term. One-sided
        interpolants (``alpha_fn is None``) ignore ``z_src`` and return
        :math:`\gamma(0)\,\epsilon`.

        Args:
            z_src: Embedded source :math:`z_{\mathrm{src}}`, or ``None`` when
                the interpolant is one-sided.
            epsilon: The interpolant's noise variable :math:`\epsilon`.

        Returns:
            The initial embedded state :math:`z_0`.

        Raises:
            ValueError: If ``alpha_fn`` is set but ``z_src`` is ``None``.
        """
        t0 = jnp.asarray(0.0)
        gamma_0 = self.gamma_fn(t0) if self.gamma_fn is not None else 0.0
        if self.alpha_fn is None:
            return gamma_0 * epsilon
        if z_src is None:
            raise ValueError(
                "z_src is None for linear interpolant while alpha_fn is not None."
            )
        return self.alpha_fn(t0) * z_src + gamma_0 * epsilon


class LinearDeterministicInterpolant(
    LinearInterpolant, ContinuousDeterministicInterpolant
):
    r"""Linear two-sided deterministic interpolant.

    .. math::
        z_t = \alpha_t z_{\mathrm{src}} + \beta_t z_{\mathrm{tgt}}.
    """

    def __init__(
        self,
        alpha_fn: Callable[[Time], Scalar] | None,
        beta_fn: Callable[[Time], Scalar],
    ) -> None:
        r"""Initialise the linear two-sided deterministic interpolant.

        Args:
            alpha_fn: Source schedule :math:`\alpha_t`.
            beta_fn: Target schedule :math:`\beta_t`.
        """
        LinearInterpolant.__init__(
            self, alpha_fn=alpha_fn, beta_fn=beta_fn, gamma_fn=None
        )
        ContinuousDeterministicInterpolant.__init__(self)

    def velocity_from_velocity_field(self, v_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the velocity from the velocity field, :math:`b_t = v_t`.

        Args:
            v_t: The velocity field at time :math:`t`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity :math:`b_t`.
        """
        return v_t

    def velocity_field_from_velocity(self, b_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the velocity field from the velocity, :math:`v_t = b_t`.

        Args:
            b_t: The velocity at time :math:`t`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity field :math:`v_t`.
        """
        return b_t


class LinearStochasticInterpolant(LinearInterpolant, ContinuousStochasticInterpolant):
    r"""Linear stochastic interpolant.

    .. math::
        z_t = \alpha_t z_{\mathrm{src}} + \beta_t z_{\mathrm{tgt}} + \gamma_t \epsilon

    with :math:`\gamma_t` non-zero for :math:`t` in :math:`(0, 1)`.
    """

    gamma_fn: Callable[[Time], Scalar]

    def __init__(
        self,
        alpha_fn: Callable[[Time], Scalar] | None,
        beta_fn: Callable[[Time], Scalar],
        gamma_fn: Callable[[Time], Scalar],
    ) -> None:
        r"""Initialise the linear stochastic interpolant.

        Args:
            alpha_fn: Source schedule :math:`\alpha_t`.
            beta_fn: Target schedule :math:`\beta_t`.
            gamma_fn: Noise schedule :math:`\gamma_t`.
        """
        ContinuousStochasticInterpolant.__init__(self, gamma_fn=gamma_fn)
        LinearInterpolant.__init__(
            self, alpha_fn=alpha_fn, beta_fn=beta_fn, gamma_fn=gamma_fn
        )

    def score_from_noise(self, epsilon: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the score from the noise.

        .. math::
            s_t = -\epsilon / \gamma_t.

        Args:
            epsilon: The noise variable :math:`\epsilon`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The score :math:`s_t`.
        """
        return -epsilon / self.gamma_fn(t)

    def noise_from_score(self, s_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the noise from the score.

        .. math::
            \epsilon = -\gamma_t s_t.

        Args:
            s_t: The score at time :math:`t`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The noise variable :math:`\epsilon`.
        """
        return -self.gamma_fn(t) * s_t

    def velocity_from_velocity_field_and_noise(
        self, v_t: Var, epsilon: Var, z_t: EmbeddedVar, t: Time
    ) -> Var:
        r"""Compute the velocity from the velocity field and noise.

        .. math::
            b_t = v_t + \dot{\gamma}_t \epsilon.

        Args:
            v_t: The velocity field at time ``t``.
            epsilon: The noise variable :math:`\epsilon`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity :math:`b_t`.
        """
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        return v_t + dot_gamma_t * epsilon

    def velocity_field_from_velocity_and_noise(
        self, b_t: Var, epsilon: Var, z_t: EmbeddedVar, t: Time
    ) -> Var:
        r"""Compute the velocity field from the velocity and noise.

        .. math::
            v_t = b_t - \dot{\gamma}_t \epsilon.

        Args:
            b_t: The velocity at time ``t``.
            epsilon: The noise variable :math:`\epsilon`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity field :math:`v_t`.
        """
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        return b_t - dot_gamma_t * epsilon

    def velocity_from_velocity_field_and_score(
        self, v_t: Var, s_t: Var, z_t: EmbeddedVar, t: Time
    ) -> Var:
        r"""Compute the velocity from the velocity field and score.

        .. math::
            b_t = v_t - \dot{\gamma}_t \gamma_t s_t.

        Args:
            v_t: The velocity field at time ``t``.
            s_t: The score at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity :math:`b_t`.
        """
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        b_t = v_t - dot_gamma_t * self.gamma_fn(t) * s_t
        return b_t

    def velocity_field_from_velocity_and_score(
        self, b_t: Var, s_t: Var, z_t: EmbeddedVar, t: Time
    ) -> Var:
        r"""Compute the velocity field from the velocity and score.

        .. math::
            v_t = b_t + \dot{\gamma}_t \gamma_t s_t.

        Args:
            b_t: The velocity at time ``t``.
            s_t: The score at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity field :math:`v_t`.
        """
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        v_t = b_t + dot_gamma_t * self.gamma_fn(t) * s_t
        return v_t


class OneSidedLinearStochasticInterpolant(
    LinearStochasticInterpolant, ContinuousOneSidedStochasticInterpolant
):
    r"""Linear one-sided interpolant.

    .. math::
        z_t = \beta_t z_{\mathrm{tgt}} + \gamma_t\,\epsilon,

    with :math:`\gamma_t` non-zero for :math:`t` in :math:`[0, 1)`.
    Note that :math:`\gamma_{t=0}` must be non-zero to ensure that
    the initial state :math:`z_0` is not deterministic.

    This interpolant class implements built-in conversion methods to convert between
    the interpolated variable :math:`z_t`, the embedded target variable :math:`z_{\mathrm{tgt}}`,
    the noise variable :math:`\epsilon`,
    the velocity field :math:`v_t`, the velocity :math:`b_t`, and the score :math:`s_t`.
    """

    gamma_fn: Callable[[Time], Scalar]

    def __init__(
        self,
        gamma_fn: Callable[[Time], Scalar],
        beta_fn: Callable[[Time], Scalar],
    ) -> None:
        r"""Initialise the linear one-sided stochastic interpolant.

        Args:
            gamma_fn: Noise schedule :math:`\gamma_t`.
            beta_fn: Target schedule :math:`\beta_t`.
        """
        ContinuousOneSidedStochasticInterpolant.__init__(self, gamma_fn=gamma_fn)
        LinearStochasticInterpolant.__init__(
            self, gamma_fn=gamma_fn, alpha_fn=None, beta_fn=beta_fn
        )

    def _interpolant_fn(self, embedded_target: EmbeddedVar, t: Time) -> EmbeddedVar:
        r"""Return :math:`\beta_t z_{\mathrm{tgt}}`.

        Args:
            embedded_target: The embedded target :math:`z_{\mathrm{tgt}}`.
            t: Interpolation time.

        Returns:
            The deterministic interpolation :math:`\beta_t z_{\mathrm{tgt}}`.
        """
        return self.beta_fn(t) * embedded_target

    # Score from variable conversions

    def score_from_target(self, embedded_target: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the score from the embedded target :math:`z_{\mathrm{tgt}}`.

        .. math::
            s_t = (\beta_t z_{\mathrm{tgt}} - z_t) / \gamma_t^2

        Args:
            embedded_target: The embedded target :math:`z_{\mathrm{tgt}}`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The score :math:`s_t`.
        """
        return (self.beta_fn(t) * embedded_target - z_t) / self.gamma_fn(t) ** 2

    def score_from_velocity_field(self, v_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the score from the velocity field.

        .. math::
            s_t = ((\beta_t / \dot{\beta}_t) * v_t - z_t) / \gamma_t^2

        Args:
            v_t: The velocity field at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The score :math:`s_t`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return ((self.beta_fn(t) / dot_beta_t) * v_t - z_t) / self.gamma_fn(t) ** 2

    def score_from_velocity(self, b_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the score from the velocity.

        .. math::
            s_t = 1 / ( (\dot{\beta}_t / \beta_t)*\gamma_t^2 - \dot{\gamma}_t*\gamma_t) * (b_t - (\dot{\beta}_t / \beta_t) * z_t)

        Args:
            b_t: The velocity at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The score :math:`s_t`.
        """
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return (
            1
            / (
                (dot_beta_t / self.beta_fn(t)) * self.gamma_fn(t) ** 2
                - dot_gamma_t * self.gamma_fn(t)
            )
            * (b_t - (dot_beta_t / self.beta_fn(t)) * z_t)
        )

    # Velocity from variable conversions

    def velocity_from_noise(self, epsilon: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the velocity from the noise.

        .. math::
            b_t = \dot{\beta}_t (z_t - \gamma_t \epsilon) / \beta_t + \dot{\gamma}_t \epsilon.

        Args:
            epsilon: The noise variable :math:`\epsilon`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity :math:`b_t`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        return (
            dot_beta_t * (z_t - self.gamma_fn(t) * epsilon) / self.beta_fn(t)
            + dot_gamma_t * epsilon
        )

    def velocity_from_score(self, s_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the velocity from the score.

        .. math::
            b_t = (\dot{\beta}_t / \beta_t) * (z_t + \gamma_t^2 * s_t) - \dot{\gamma}_t * \gamma_t * s_t

        Args:
            s_t: The score at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity :math:`b_t`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        return (dot_beta_t / self.beta_fn(t)) * (
            z_t + self.gamma_fn(t) ** 2 * s_t
        ) - dot_gamma_t * self.gamma_fn(t) * s_t

    def velocity_from_target(
        self, embedded_target: Var, z_t: EmbeddedVar, t: Time
    ) -> Var:
        r"""Compute the velocity from the embedded target :math:`z_{\mathrm{tgt}}`.

        .. math::
            b_t = \dot{\beta}_t z_{\mathrm{tgt}} + (\dot{\gamma}_t / \gamma_t) * (z_t - \beta_t z_{\mathrm{tgt}})

        Args:
            embedded_target: The embedded target :math:`z_{\mathrm{tgt}}`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity :math:`b_t`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        return dot_beta_t * embedded_target + (dot_gamma_t / self.gamma_fn(t)) * (
            z_t - self.beta_fn(t) * embedded_target
        )

    def velocity_from_velocity_field(self, v_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the velocity from the velocity field.

        .. math::
            b_t = v_t + (\dot{\gamma}_t / \gamma_t) * (z_t - (\beta_t / \dot{\beta}_t) * v_t)

        Args:
            v_t: The velocity field at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity :math:`b_t`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        return v_t + (dot_gamma_t / self.gamma_fn(t)) * (
            z_t - (self.beta_fn(t) / dot_beta_t) * v_t
        )

    # Velocity-field from variable conversions

    def velocity_field_from_noise(self, epsilon: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the velocity field from the noise.

        .. math::
            v_t = \dot{\beta}_t (z_t - \gamma_t \epsilon) / \beta_t

        Args:
            epsilon: The noise variable :math:`\epsilon`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity field :math:`v_t`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return dot_beta_t * (z_t - self.gamma_fn(t) * epsilon) / self.beta_fn(t)

    def velocity_field_from_score(self, s_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the velocity field from the score.

        .. math::
            v_t = (\dot{\beta}_t / \beta_t) * (z_t + \gamma_t^2 * s_t)

        Args:
            s_t: The score at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity field :math:`v_t`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return (dot_beta_t / self.beta_fn(t)) * (z_t + self.gamma_fn(t) ** 2 * s_t)

    def velocity_field_from_velocity(self, b_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the velocity field from the velocity.

        .. math::
            v_t =(1/(1-(\dot{\gamma}_t/\dot{\beta}_t)*(\beta_t/\gamma_t))*(b_t - (\dot{\gamma}_t / \gamma_t) * z_t)

        Args:
            b_t: The velocity at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity field :math:`v_t`.
        """
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return (
            1 / (1 - (dot_gamma_t / dot_beta_t) * (self.beta_fn(t) / self.gamma_fn(t)))
        ) * (b_t - (dot_gamma_t / self.gamma_fn(t)) * z_t)

    def velocity_field_from_target(
        self, embedded_target: Var, z_t: EmbeddedVar, t: Time
    ) -> Var:
        r"""Compute the velocity field from the embedded target :math:`z_{\mathrm{tgt}}`.

        .. math::
            v_t = \dot{\beta}_t z_{\mathrm{tgt}}

        Args:
            embedded_target: The embedded target :math:`z_{\mathrm{tgt}}`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The velocity field :math:`v_t`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return dot_beta_t * embedded_target

    # Noise from variable conversions

    def noise_from_target(self, embedded_target: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the noise from the embedded target :math:`z_{\mathrm{tgt}}`.

        .. math::
            \epsilon = (z_t - \beta_t z_{\mathrm{tgt}}) / \gamma_t

        Args:
            embedded_target: The embedded target :math:`z_{\mathrm{tgt}}`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The noise variable :math:`\epsilon`.
        """
        return (z_t - self.beta_fn(t) * embedded_target) / self.gamma_fn(t)

    def noise_from_velocity_field(self, v_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the noise from the velocity field.

        .. math::
            \epsilon = (z_t - (\beta_t / \dot{\beta}_t) * v_t) / \gamma_t

        Args:
            v_t: The velocity field at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The noise variable :math:`\epsilon`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return (z_t - (self.beta_fn(t) / dot_beta_t) * v_t) / self.gamma_fn(t)

    def noise_from_velocity(self, b_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the noise from the velocity.

        .. math::
            \epsilon = -\gamma_t / ( (\dot{\beta}_t / \beta_t)*\gamma_t^2 - \dot{\gamma}_t*\gamma_t) * (b_t - (\dot{\beta}_t / \beta_t) * z_t)

        Args:
            b_t: The velocity at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The noise variable :math:`\epsilon`.
        """
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return (
            -self.gamma_fn(t)
            / (
                (dot_beta_t / self.beta_fn(t)) * self.gamma_fn(t) ** 2
                - dot_gamma_t * self.gamma_fn(t)
            )
            * (b_t - (dot_beta_t / self.beta_fn(t)) * z_t)
        )

    # Target from variable conversions

    def target_from_noise(self, epsilon: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the embedded target :math:`z_{\mathrm{tgt}}` from the noise.

        .. math::
            z_{\mathrm{tgt}} = (z_t - \gamma_t \epsilon) / \beta_t

        Args:
            epsilon: The noise variable :math:`\epsilon`.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The embedded target :math:`z_{\mathrm{tgt}}`.
        """
        return (z_t - self.gamma_fn(t) * epsilon) / self.beta_fn(t)

    def target_from_score(self, s_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the embedded target :math:`z_{\mathrm{tgt}}` from the score.

        .. math::
            z_{\mathrm{tgt}} = (z_t + \gamma_t^2 * s_t) / \beta_t

        Args:
            s_t: The score at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The embedded target :math:`z_{\mathrm{tgt}}`.
        """
        return (z_t + self.gamma_fn(t) ** 2 * s_t) / self.beta_fn(t)

    def target_from_velocity_field(self, v_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the embedded target :math:`z_{\mathrm{tgt}}` from the velocity field.

        .. math::
            z_{\mathrm{tgt}} = v_t / \dot{\beta}_t

        Args:
            v_t: The velocity field at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The embedded target :math:`z_{\mathrm{tgt}}`.
        """
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return v_t / dot_beta_t

    def target_from_velocity(self, b_t: Var, z_t: EmbeddedVar, t: Time) -> Var:
        r"""Compute the embedded target :math:`z_{\mathrm{tgt}}` from the velocity.

        .. math::
            z_{\mathrm{tgt}} = b_t/\dot{\beta}_t + 1/((\dot{\beta}_t / \dot{\gamma}_t)*(\gamma_t / \beta_t) - 1) * (b_t / \dot{\beta}_t - z_t / \beta_t)

        Args:
            b_t: The velocity at time ``t``.
            z_t: The interpolated state.
            t: Interpolation time.

        Returns:
            The embedded target :math:`z_{\mathrm{tgt}}`.
        """
        dot_gamma_t = jax.jacfwd(self.gamma_fn)(t)
        dot_beta_t = jax.jacfwd(self.beta_fn)(t)
        return b_t / dot_beta_t + 1 / (
            (dot_beta_t / dot_gamma_t) * (self.gamma_fn(t) / self.beta_fn(t)) - 1
        ) * (b_t / dot_beta_t - z_t / self.beta_fn(t))
