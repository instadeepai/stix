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

"""Tests for the fixed-step ManualSolver (with the merged guidance path)."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import (
    SHAPE,
    TOL,
    toy_constant_velocity_gen_model,
    toy_guided_constant_velocity_gen_model,
)

from stix.core.generator import VelocityAndScore
from stix.sampling.guidance import (
    get_classifier_free_guidance_generator,
    get_intrinsic_guidance_generator,
)
from stix.sampling.solver_manual import ManualSolver, ManualSolverConfig
from stix.sampling.utils import Direction

# ══════════════════════════════════════════════════════════════════════
# ManualSolver — ODE: constant velocity (Euler is exact for constant drift)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "value,direction",
    [
        (1.0, Direction.FORWARD),
        (1.0, Direction.REVERSE),
        (2.5, Direction.FORWARD),
    ],
)
def test_ode_constant_velocity(value, direction):
    """Euler integration is exact for a constant drift field. With default
    config ``t_min_tolerance=t_max_tolerance=TOL``, starting from ``x=0`` and
    integrating over ``[TOL, 1-TOL]`` (or its reverse) gives ``sign * value *
    (1 - 2*TOL)``.
    """
    gen_model = toy_constant_velocity_gen_model(value=value)
    solver = ManualSolver(
        ManualSolverConfig(
            direction=direction, num_steps=10, t_min_tolerance=TOL, t_max_tolerance=TOL
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    result = solver(gen_model, z_init, key=jr.PRNGKey(0))

    sign = 1.0 if direction == Direction.FORWARD else -1.0
    expected = jnp.ones(SHAPE) * sign * value * (1.0 - 2 * TOL)
    assert jnp.allclose(result["mod_a"], expected, atol=1e-4)


# ══════════════════════════════════════════════════════════════════════
# ManualSolver — SDE
# ══════════════════════════════════════════════════════════════════════


def test_sde_produces_finite_output():
    """``ManualSolver`` in SDE mode with a small stochasticity scale produces
    finite output of the expected shape (smoke test).
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = ManualSolver(
        ManualSolverConfig(
            stochasticity_scale={"mod_a": lambda t: jnp.array(0.01)},
            direction=Direction.FORWARD,
            num_steps=20,
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    result = solver(gen_model, z_init, key=jr.PRNGKey(42))

    assert result["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(result["mod_a"]))


def test_sde_at_zero_scale_matches_ode_integration():
    """SDE drift with zero stochasticity scale collapses to ODE drift exactly.

    Concrete check that the SDE code path adds no extra terms — the score
    contribution is multiplied by zero, the Brownian noise is multiplied by
    zero. Result must match the pure ODE integration (within float precision).
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    ode_solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.FORWARD,
            num_steps=50,
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )
    sde_zero_solver = ManualSolver(
        ManualSolverConfig(
            stochasticity_scale={"mod_a": lambda t: jnp.array(0.0)},
            direction=Direction.FORWARD,
            num_steps=50,
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    ode_result = ode_solver(gen_model, z_init, key=jr.PRNGKey(0))
    sde_zero_result = sde_zero_solver(gen_model, z_init, key=jr.PRNGKey(0))

    assert jnp.allclose(ode_result["mod_a"], sde_zero_result["mod_a"], atol=1e-5)


# ══════════════════════════════════════════════════════════════════════
# ManualSolver — Trajectory mode
# ══════════════════════════════════════════════════════════════════════


def test_trajectory_mode_shapes():
    """``integrate_with_trajectory`` should return a final state with the
    modality's shape, a trajectory of shape ``(num_steps, *shape)`` per
    modality, and a ``trajectory_t`` tensor of shape ``(num_steps,)``.
    """
    num_steps = 20
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.FORWARD,
            num_steps=num_steps,
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    z_final, trajectory, trajectory_t = solver.integrate_with_trajectory(
        gen_model, z_init, key=jr.PRNGKey(0)
    )

    assert z_final["mod_a"].shape == SHAPE
    assert trajectory["mod_a"].shape == (num_steps, *SHAPE)
    assert trajectory_t.shape == (num_steps,)


def test_trajectory_monotonic_for_constant_velocity():
    """With a constant positive velocity applied to an initial state of zero,
    the per-step norm of the trajectory should increase monotonically.
    """
    num_steps = 20
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.FORWARD,
            num_steps=num_steps,
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    _, trajectory, _ = solver.integrate_with_trajectory(
        gen_model, z_init, key=jr.PRNGKey(0)
    )

    norms = jnp.linalg.norm(trajectory["mod_a"], axis=-1)
    assert jnp.all(norms[1:] > norms[:-1])


# ══════════════════════════════════════════════════════════════════════
# Guidance recipe smoke tests (merged into ManualSolver via guidance_fn)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "direction",
    [Direction.FORWARD, Direction.REVERSE],
)
def test_cfg_recipe_ode_constant_velocity(direction):
    """Analytical ODE check with the CFG recipe.

    Exercises the score->velocity round-trip: the CFG recipe rebuilds the
    velocity from the (combined) score via ``velocity_from_score``. With a
    context-blind constant net (v=1), the round-trip returns v=1, so constant
    ``v=1`` starting at ``x=0`` must produce ``sign * (1 - 2*TOL)``.
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = ManualSolver(
        ManualSolverConfig(
            direction=direction,
            num_steps=10,
            guidance_fn=get_classifier_free_guidance_generator,
            guidance_scale={"mod_a": lambda t: jnp.array(1.0)},
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    result = solver(gen_model, z_init, key=jr.PRNGKey(0))

    sign = 1.0 if direction == Direction.FORWARD else -1.0
    expected = jnp.ones(SHAPE) * sign * (1.0 - 2 * TOL)
    assert jnp.allclose(result["mod_a"], expected, atol=1e-4)


def test_manual_solver_with_intrinsic_recipe_runs():
    """``ManualSolver`` with the intrinsic-guidance recipe + real conditioning
    runs end-to-end and produces finite output.

    Wiring test only — recipe correctness is covered in ``test_guidance.py``.
    """
    gen_model = toy_guided_constant_velocity_gen_model(value=1.0)
    z_init = {"mod_a": jnp.zeros(SHAPE)}
    intrinsic_data = {"mod_a": jnp.ones(SHAPE)}
    intrinsic_mask = {"mod_a": jnp.ones(SHAPE)}

    solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.FORWARD,
            num_steps=20,
            guidance_fn=get_intrinsic_guidance_generator,
            guidance_scale={"mod_a": lambda t: jnp.array(0.5)},
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    result = solver(
        gen_model,
        z_init,
        key=jr.PRNGKey(0),
        intrinsic_data=intrinsic_data,
        intrinsic_mask=intrinsic_mask,
    )

    assert result["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(result["mod_a"]))


def test_manual_solver_with_cfg_recipe_runs():
    """``ManualSolver`` with the CFG recipe + real conditioning runs end-to-end
    and produces finite output.

    Wiring test only — recipe correctness is covered in ``test_guidance.py``.
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    z_init = {"mod_a": jnp.zeros(SHAPE)}
    context_data = {"context": jnp.ones(SHAPE)}
    context_mask = {"context": jnp.ones(SHAPE)}

    solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.FORWARD,
            num_steps=20,
            guidance_fn=get_classifier_free_guidance_generator,
            guidance_scale={"mod_a": lambda t: jnp.array(1.5)},
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    result = solver(
        gen_model,
        z_init,
        key=jr.PRNGKey(0),
        context_data=context_data,
        context_mask=context_mask,
    )

    assert result["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(result["mod_a"]))


def test_manual_solver_integrate_with_trajectory_shape_cfg():
    """``ManualSolver.integrate_with_trajectory`` (CFG recipe) returns the
    expected shapes: final state per modality, ``(num_steps, *shape)``
    trajectory, and ``(num_steps,)`` time vector.
    """
    num_steps = 20
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.FORWARD,
            num_steps=num_steps,
            guidance_fn=get_classifier_free_guidance_generator,
            guidance_scale={"mod_a": lambda t: jnp.array(1.0)},
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    z_final, trajectory, trajectory_t = solver.integrate_with_trajectory(
        gen_model, z_init, key=jr.PRNGKey(0)
    )

    assert z_final["mod_a"].shape == SHAPE
    assert trajectory["mod_a"].shape == (num_steps, *SHAPE)
    assert trajectory_t.shape == (num_steps,)


def test_manual_solver_custom_recipe_controls_velocity():
    """A custom recipe controls the generator the solver steps.

    A recipe returning a constant sentinel velocity makes the ODE Euler integral
    analytic: from ``x=0`` with constant drift ``sentinel`` over ``[TOL, 1-TOL]``
    gives ``sentinel * (1 - 2*TOL)``. Pins that the configured recipe, not the
    model's own generator (``v=1``), drives the step.
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    sentinel = 7.0

    def sentinel_recipe(
        gen_model,
        z_t,
        t,
        context_data=None,
        context_mask=None,
        intrinsic_data=None,
        intrinsic_mask=None,
        *,
        attention_mask=None,
        guidance_scale,
    ):
        del context_data, context_mask, intrinsic_data, intrinsic_mask
        del t, attention_mask, guidance_scale
        return {
            k: VelocityAndScore(
                velocity=jnp.full_like(v, sentinel), score=jnp.zeros_like(v)
            )
            for k, v in z_t.items()
        }

    solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.FORWARD,
            num_steps=10,
            guidance_fn=sentinel_recipe,
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    result = solver.integrate(gen_model, z_init, key=jr.PRNGKey(0))

    expected = jnp.full(SHAPE, sentinel * (1.0 - 2 * TOL))
    assert jnp.allclose(result["mod_a"], expected, atol=1e-4)
