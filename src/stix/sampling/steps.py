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

r"""Per-generator-type stepping logic for the samplers.

:class:`~stix.core.generator.Generator` is pure data; this module
holds the behaviour, as plain functions keyed by the concrete generator type.
Three tables cover the two solver backends:

- :data:`STEP_FNS` — one update ``z_t -> z_{t+dt}`` over a given ``dt``, as used
  by the manual solver, :class:`~stix.sampling.solver_manual.ManualSolver`.
- :data:`DRIFT_FNS` and :data:`DIFFUSION_FNS` — the drift/diffusion split the
  adaptive (diffrax) solver needs.

NOTE: every :data:`STEP_FNS` entry receives an **already-evaluated**
:class:`~stix.core.generator.Generator`, so it gets exactly one network
evaluation per step and cannot query the model at an intermediate stage. That
fixes the manual solver to first-order explicit schemes — Euler, Euler-Maruyama,
and the CTMC analogues. Multi-stage schemes (Heun, midpoint, higher-order
stochastic Runge-Kutta) need the step function to own the network evaluation.
The adaptive solver, :class:`~stix.sampling.solver.Solver`, is not subject to
this: it owns the drift/diffusion closures and diffrax calls them as many times
per step as its tableau requires.

A generator type that is a key of :data:`STEP_FNS` but *not* of
:data:`DRIFT_FNS` (e.g. :class:`~stix.core.generator.TransitionRates`) can
be simulated only by the manual solver:
:func:`~stix.sampling.utils.validate_generator_types` rejects it before the
adaptive solver traces anything. Adding a new generator type is one data class
plus one entry per table, with no solver edits.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from stix.core.generator import (
    Generator,
    TransitionRates,
    Velocity,
    VelocityAndScore,
)
from stix.sampling.utils import Direction
from stix.typing import PRNGKeyArray, Scalar, Time, Var

# A per-modality time-dependent stochasticity scale ``t -> lambda_t``, shared by
# the continuous SDE term and the CTMC corrector. ``None`` means
# :math:`\lambda \equiv 0`: no noise for an SDE, no corrector for a CTMC.
StochasticityScaleFn = Callable[[Time], Scalar]

# One update of a single modality over an interval ``dt``, from an already-evaluated
# generator (hence single-stage schemes only; see the module docstring).
StepFn = Callable[
    [
        Time,
        Var,
        Generator,
        float,
        PRNGKeyArray,
        StochasticityScaleFn | None,
        Direction,
    ],
    Var,
]
# Diffrax drift for a single modality.
DriftFn = Callable[[Time, Generator, StochasticityScaleFn | None, Direction], Var]
# Diffrax diffusion (diagonal) for a single modality.
DiffusionFn = Callable[[Time, Var, StochasticityScaleFn | None], Var]


# ── Step updates ──


def velocity_euler_step(
    t: Time,
    z_t_k: Var,
    generator_k: Generator,
    dt: float,
    key_k: PRNGKeyArray,
    stochasticity_scale_k: StochasticityScaleFn | None,
    direction: Direction,
) -> Var:
    r"""Euler ODE step for a :class:`~stix.core.generator.Velocity`.

    One first-order update :math:`z_{t+\mathrm{d}t} = z_t + b_t(z_t)\,\mathrm{d}t`,
    evaluating the velocity once, at the left endpoint.

    Args:
        t: Current time (unused).
        z_t_k: Current state for one modality.
        generator_k: Per-modality generator (must be a
            :class:`~stix.core.generator.Velocity`).
        dt: Signed time step.
        key_k: PRNG key (unused).
        stochasticity_scale_k: Stochasticity scale (unused; must be ``None`` for ODE).
        direction: Integration direction (unused; the signed ``dt`` already
            encodes it).

    Returns:
        Updated state after one Euler step.
    """
    del t, key_k, stochasticity_scale_k, direction
    assert isinstance(generator_k, Velocity)
    return z_t_k + generator_k.velocity * dt


def velocity_and_score_euler_maruyama_step(
    t: Time,
    z_t_k: Var,
    generator_k: Generator,
    dt: float,
    key_k: PRNGKeyArray,
    stochasticity_scale_k: StochasticityScaleFn | None,
    direction: Direction,
) -> Var:
    r"""Euler-Maruyama SDE step for a :class:`~stix.core.generator.VelocityAndScore`.

    .. math::
        z_{t+\mathrm{d}t} = z_t
        + \left(b_t \pm \lambda_t s_t\right)\mathrm{d}t
        + \sqrt{2\lambda_t}\,\sqrt{|\mathrm{d}t|}\,\xi,
        \qquad \xi\sim\mathcal{N}(0, I),

    with ``+`` forward and ``-`` reverse. Drift and score are evaluated once, at
    the left endpoint. When ``stochasticity_scale_k`` is ``None`` the noise term vanishes and
    this degenerates to the Euler ODE step.

    Args:
        t: Current time.
        z_t_k: Current state for one modality.
        generator_k: Per-modality generator (must be a
            :class:`~stix.core.generator.VelocityAndScore`).
        dt: Signed time step.
        key_k: PRNG key for the Brownian increment.
        stochasticity_scale_k: Time-dependent stochasticity scale, or ``None`` for ODE.
        direction: Integration direction; flips the score term on reverse.

    Returns:
        Updated state after one step.
    """
    assert isinstance(generator_k, VelocityAndScore)
    if stochasticity_scale_k is None:
        return z_t_k + generator_k.velocity * dt
    score_term = stochasticity_scale_k(t) * generator_k.score
    if direction == Direction.REVERSE:
        score_term = -score_term
    drift = generator_k.velocity + score_term
    diffusion = jnp.sqrt(2 * stochasticity_scale_k(t))
    noise = jr.normal(key_k, z_t_k.shape)
    return z_t_k + drift * dt + diffusion * jnp.sqrt(jnp.abs(dt)) * noise


def ctmc_euler_step(
    t: Time,
    z_t_k: Var,
    generator_k: Generator,
    dt: float,
    key_k: PRNGKeyArray,
    stochasticity_scale_k: StochasticityScaleFn | None,
    direction: Direction,
) -> Var:
    r"""Euler CTMC step for a :class:`~stix.core.generator.TransitionRates`.

    ``z_t_k`` is the current state as a **one-hot** vector over the CTMC states,
    shape ``(..., num_states)``. The solver mixes the generator's forward and
    backward rates into the corrector of stochasticity scale :math:`\lambda_t`:

    .. math::
        \bar u_t = (1 + \lambda_t)\,\hat u_t + \lambda_t\,\check u_t,

    the one-parameter family that walks the marginals :math:`p_t` forward, since
    :math:`\hat u_t + \check u_t` is divergence-free against :math:`p_t`.
    ``stochasticity_scale_k is None`` is :math:`\lambda\equiv 0`, the plain generative chain.
    On :attr:`~stix.sampling.utils.Direction.REVERSE` the two weights are
    swapped, namely :math:`(1 + \lambda_t)\,\hat u_t + \lambda_t\,\check u_t`
    becomes :math:`\lambda_t\,\hat u_t + (1 + \lambda_t)\,\check u_t`, so the same
    :math:`\lambda_t` time-reverses the process. Euler-forward
    on the transition probabilities, over the unsigned interval :math:`|\mathrm{d}t|`:

    .. math::
        p(\cdot \mid z) = \delta(\cdot, z) + \mathrm{d}t \, \bar u_t(\cdot \mid z),

    where the one-hot ``z_t_k`` *is* the :math:`\delta(\cdot, z)` term. The
    probabilities are clipped to be non-negative and renormalised, then a
    categorical index is drawn independently at each position and re-encoded as
    a one-hot vector so the sampling state stays one-hot.

    Args:
        t: Current time, used to evaluate ``stochasticity_scale_k``.
        z_t_k: One-hot CTMC state for one modality.
        generator_k: Per-modality generator (must be a
            :class:`~stix.core.generator.TransitionRates`).
        dt: Time step. Only its magnitude is used, the direction coming from
            ``direction`` via the swap of the two weights.
        key_k: PRNG key for the categorical draw.
        stochasticity_scale_k: Time-dependent corrector scale :math:`\lambda_t \geq 0`, or
            ``None`` for :math:`\lambda\equiv 0`. Clamped at zero.
        direction: Integration direction; reverse swaps the weights of the forward and backward rates.

    Returns:
        Updated one-hot state after one Euler step.

    Note:
        One categorical draw per position per step means **at most one jump
        per position per step**, so this is the Euler discretisation of the CTMC.
    """
    assert isinstance(generator_k, TransitionRates)
    # Clamped, not trusted: a negative lambda would put negative off-diagonal rates
    # in the mix, and the probability clip below would then absorb them silently.
    lam = (
        0.0
        if stochasticity_scale_k is None
        else jnp.clip(stochasticity_scale_k(t), min=0.0)
    )
    forward_weight, backward_weight = 1.0 + lam, lam
    if direction == Direction.REVERSE:
        forward_weight, backward_weight = backward_weight, forward_weight
    rates = (
        forward_weight * generator_k.forward_rates
        + backward_weight * generator_k.backward_rates
    )
    num_states = rates.shape[-1]
    # The weight swap above already points the chain the right way, so the
    # step takes the unsigned interval. I.e., a signed ``dt`` would reverse it twice and
    # drive every off-diagonal probability negative, clipping the state onto itself.
    probs = z_t_k + jnp.abs(dt) * rates
    probs = jnp.clip(probs, min=0.0)
    probs = probs / jnp.clip(jnp.sum(probs, axis=-1, keepdims=True), min=1e-12)
    next_index = jr.categorical(key_k, jnp.log(jnp.clip(probs, min=1e-12)))
    return jax.nn.one_hot(next_index, num_states, dtype=z_t_k.dtype)


# ── Diffrax drift / diffusion ──


def velocity_drift(
    t: Time,
    generator_k: Generator,
    stochasticity_scale_k: StochasticityScaleFn | None,
    direction: Direction,
) -> Var:
    """Diffrax drift for a :class:`~stix.core.generator.Velocity` generator.

    Returns the velocity itself.

    Args:
        t: Current time (unused).
        generator_k: Per-modality generator (must be a
            :class:`~stix.core.generator.Velocity`).
        stochasticity_scale_k: Stochasticity scale (unused).
        direction: Integration direction (unused; the signed time interval
            already encodes it).

    Returns:
        Drift term equal to the generator velocity.
    """
    del t, stochasticity_scale_k, direction
    assert isinstance(generator_k, Velocity)
    return generator_k.velocity


def velocity_and_score_drift(
    t: Time,
    generator_k: Generator,
    stochasticity_scale_k: StochasticityScaleFn | None,
    direction: Direction,
) -> Var:
    r"""Diffrax drift for a :class:`~stix.core.generator.VelocityAndScore`.

    ``b`` when ``stochasticity_scale_k`` is ``None`` (ODE), otherwise the SDE drift
    :math:`b_t \pm \lambda_t s_t` (``+`` forward, ``-`` reverse).

    Args:
        t: Current time.
        generator_k: Per-modality generator (must be a
            :class:`~stix.core.generator.VelocityAndScore`).
        stochasticity_scale_k: Time-dependent stochasticity scale, or ``None`` for ODE.
        direction: Integration direction; flips the score term on reverse.

    Returns:
        Drift term for the diffrax term.
    """
    assert isinstance(generator_k, VelocityAndScore)
    if stochasticity_scale_k is None:
        return generator_k.velocity
    score_term = stochasticity_scale_k(t) * generator_k.score
    if direction == Direction.REVERSE:
        score_term = -score_term
    return generator_k.velocity + score_term


def velocity_diffusion(
    t: Time, z_t_k: Var, stochasticity_scale_k: StochasticityScaleFn | None
) -> Var:
    """Diffrax diffusion for a :class:`~stix.core.generator.Velocity`.

    Always zero (ODE).

    Args:
        t: Current time (unused).
        z_t_k: Current state (used only for shape/dtype).
        stochasticity_scale_k: Stochasticity scale (unused).

    Returns:
        A zero array matching ``z_t_k``.
    """
    del t, stochasticity_scale_k
    return jnp.zeros_like(z_t_k)


def velocity_and_score_diffusion(
    t: Time, z_t_k: Var, stochasticity_scale_k: StochasticityScaleFn | None
) -> Var:
    r"""Diffrax diffusion for a :class:`~stix.core.generator.VelocityAndScore`.

    Zero when ``stochasticity_scale_k`` (:math:`\lambda_t`) is ``None``, otherwise
    :math:`\sqrt{2\lambda_t}` broadcast to the state shape.

    Args:
        t: Current time.
        z_t_k: Current state (used for shape/dtype).
        stochasticity_scale_k: Time-dependent stochasticity scale, or ``None`` for ODE.

    Returns:
        Diagonal diffusion coefficient matching ``z_t_k``.
    """
    if stochasticity_scale_k is None:
        return jnp.zeros_like(z_t_k)
    return jnp.ones_like(z_t_k) * jnp.sqrt(2 * stochasticity_scale_k(t))


# ── Generator-type tables ──

STEP_FNS: dict[type[Generator], StepFn] = {
    Velocity: velocity_euler_step,
    VelocityAndScore: velocity_and_score_euler_maruyama_step,
    TransitionRates: ctmc_euler_step,
}
"""Single-stage update ``z_t -> z_{t+dt}`` keyed by generator type."""

DRIFT_FNS: dict[type[Generator], DriftFn] = {
    Velocity: velocity_drift,
    VelocityAndScore: velocity_and_score_drift,
}
"""Diffrax drift term keyed by generator type (continuous modalities only)."""

DIFFUSION_FNS: dict[type[Generator], DiffusionFn] = {
    Velocity: velocity_diffusion,
    VelocityAndScore: velocity_and_score_diffusion,
}
"""Diffrax diffusion term keyed by generator type (continuous modalities only)."""
