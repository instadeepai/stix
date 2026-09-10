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

"""Tests for the diffrax-based Solver (with the merged guidance path)."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import (
    SHAPE,
    TOL,
    IdentityNetwork,
    toy_constant_velocity_gen_model,
    toy_guided_constant_velocity_gen_model,
    two_sided_deterministic_si,
)

from stix.core.embedder import IdentityEmbedder
from stix.core.gen_model.factory import VelocityTwoSidedGenerativeModel
from stix.core.modality import Modality, ModalityRegistry
from stix.sampling.guidance import (
    get_classifier_free_guidance_generator,
    get_intrinsic_guidance_generator,
)
from stix.sampling.solver import Solver, SolverConfig
from stix.sampling.solver_manual import ManualSolver, ManualSolverConfig
from stix.sampling.utils import Direction

# ══════════════════════════════════════════════════════════════════════
# ODE: constant velocity field (analytically solvable)
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
    """Integrating an ODE with a constant velocity field ``v=value`` between
    ``t=TOL`` and ``t=1-TOL`` starting at ``x=0`` should produce the analytic
    solution ``sign * value * (1 - 2*TOL)``, where ``sign`` is ``+1`` for
    forward integration and ``-1`` for reverse.
    """
    gen_model = toy_constant_velocity_gen_model(value=value)
    solver = Solver(
        SolverConfig(direction=direction, t_min_tolerance=TOL, t_max_tolerance=TOL)
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    result = solver(gen_model, z_init, key=jr.PRNGKey(0))

    sign = 1.0 if direction == Direction.FORWARD else -1.0
    expected = jnp.ones(SHAPE) * sign * value * (1.0 - 2 * TOL)
    assert result["mod_a"].shape == SHAPE
    assert jnp.allclose(result["mod_a"], expected, atol=1e-2)


@pytest.mark.parametrize("num_tokens", [1, 4])
def test_output_shape_matches_x_init_and_manual_solver(num_tokens):
    """The solved state is shaped exactly like ``z_init``, for any input rank.

    ``diffrax`` returns states stacked over saved timesteps, so the endpoint
    must be unstacked before it reaches the caller — otherwise every caller
    carries a spurious leading axis and has to reshape it away. Cross-checked
    against ``ManualSolver``, which integrates by ``scan`` and is naturally
    shape-preserving: the two solvers must be interchangeable.
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    z_init = {"mod_a": jnp.zeros((num_tokens, *SHAPE))}

    solver = Solver(
        SolverConfig(
            direction=Direction.FORWARD, t_min_tolerance=TOL, t_max_tolerance=TOL
        )
    )
    manual_solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.FORWARD, t_min_tolerance=TOL, t_max_tolerance=TOL
        )
    )

    result = solver(gen_model, z_init, key=jr.PRNGKey(0))
    manual_result = manual_solver(gen_model, z_init, key=jr.PRNGKey(0))

    assert result["mod_a"].shape == z_init["mod_a"].shape
    assert result["mod_a"].shape == manual_result["mod_a"].shape


# ══════════════════════════════════════════════════════════════════════
# SDE
# ══════════════════════════════════════════════════════════════════════


def test_sde_produces_finite_output():
    """Running the solver in SDE mode with a small stochasticity scale should
    produce an output of the expected shape whose entries are all finite.
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = Solver(
        SolverConfig(
            stochasticity_scale={"mod_a": lambda t: jnp.array(0.01)},
            direction=Direction.FORWARD,
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    result = solver(gen_model, z_init, key=jr.PRNGKey(42))

    assert result["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(result["mod_a"]))


def test_sde_at_zero_scale_matches_ode_integration():
    """SDE drift with zero stochasticity scale collapses to ODE drift
    (within adaptive-solver tolerance).
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    ode_solver = Solver(
        SolverConfig(
            direction=Direction.FORWARD, t_min_tolerance=TOL, t_max_tolerance=TOL
        )
    )
    sde_zero_solver = Solver(
        SolverConfig(
            stochasticity_scale={"mod_a": lambda t: jnp.array(0.0)},
            direction=Direction.FORWARD,
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    key = jr.PRNGKey(0)
    ode_result = ode_solver(gen_model, z_init, key=key)
    sde_zero_result = sde_zero_solver(gen_model, z_init, key=key)

    # Both adaptive solvers (Tsit5 for ODE, ShARK for SDE-at-zero); tolerate
    # the adaptive-controller difference but require the drifts to agree.
    assert jnp.allclose(ode_result["mod_a"], sde_zero_result["mod_a"], atol=1e-2)


# ══════════════════════════════════════════════════════════════════════
# Guidance recipe smoke tests (merged into Solver via guidance_fn)
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
    ``v=1`` between ``t=TOL`` and ``t=1-TOL`` starting at ``x=0`` must produce
    ``sign * (1 - 2*TOL)``.
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = Solver(
        SolverConfig(
            direction=direction,
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
    assert result["mod_a"].shape == SHAPE
    assert jnp.allclose(result["mod_a"], expected, atol=1e-2)


def test_solver_with_intrinsic_recipe_runs():
    """Solver with the intrinsic-guidance recipe + real conditioning runs
    end-to-end and produces finite output.

    Wiring test only — recipe correctness is covered in ``test_guidance.py``.
    """
    gen_model = toy_guided_constant_velocity_gen_model(value=1.0)
    z_init = {"mod_a": jnp.zeros(SHAPE)}
    intrinsic_data = {"mod_a": jnp.ones(SHAPE)}
    intrinsic_mask = {"mod_a": jnp.ones(SHAPE)}

    solver = Solver(
        SolverConfig(
            direction=Direction.FORWARD,
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


def test_solver_with_cfg_recipe_runs():
    """Solver with the CFG recipe + real conditioning runs end-to-end and
    produces finite output.

    Wiring test only — recipe correctness is covered in ``test_guidance.py``.
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    z_init = {"mod_a": jnp.zeros(SHAPE)}
    context_data = {"context": jnp.ones(SHAPE)}
    context_mask = {"context": jnp.ones(SHAPE)}

    solver = Solver(
        SolverConfig(
            direction=Direction.FORWARD,
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


def test_cfg_recipe_sde_produces_finite_output():
    """CFG recipe in SDE mode with a small stochasticity scale produces finite
    output of the expected shape (smoke test).
    """
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = Solver(
        SolverConfig(
            stochasticity_scale={"mod_a": lambda t: jnp.array(0.01)},
            direction=Direction.FORWARD,
            guidance_fn=get_classifier_free_guidance_generator,
            guidance_scale={"mod_a": lambda t: jnp.array(1.5)},
            t_min_tolerance=TOL,
            t_max_tolerance=TOL,
        )
    )

    z_init = {"mod_a": jnp.zeros(SHAPE)}
    result = solver(gen_model, z_init, key=jr.PRNGKey(42))

    assert result["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(result["mod_a"]))


# ══════════════════════════════════════════════════════════════════════
# validation
# ══════════════════════════════════════════════════════════════════════


def test_solver_integrate_rejects_incompatible_scale():
    """The solver validates scales upfront: ``integrate`` raises on a structural
    mismatch with the modality registry, rather than failing deep in a closure."""
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = Solver(SolverConfig(stochasticity_scale={"nonexistent": lambda t: t}))
    with pytest.raises(ValueError, match="not compatible with the modality registry"):
        solver.integrate(gen_model, {"mod_a": jnp.zeros(SHAPE)}, key=jr.PRNGKey(0))


def test_solver_rejects_stochasticity_scale_on_velocity_modality():
    """A deterministic ``Velocity`` modality carries no score, so setting a
    stochasticity scale must be rejected up front with a clear error."""
    registry = ModalityRegistry(
        {
            "mod_a": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=two_sided_deterministic_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
            )
        }
    )
    gen_model = VelocityTwoSidedGenerativeModel(
        network=IdentityNetwork(),
        modality_registry=registry,
    )
    solver = Solver(
        SolverConfig(stochasticity_scale={"mod_a": lambda t: jnp.array(0.1)})
    )
    with pytest.raises(ValueError, match="carries no score"):
        solver.integrate(gen_model, {"mod_a": jnp.zeros(SHAPE)}, key=jr.PRNGKey(0))


def test_solver_rejects_negative_stochasticity_scale():
    """A negative scale asks for an imaginary diffusion ``sqrt(2 * lambda)``, whose
    only runtime symptom is a state full of NaNs. It must be refused up front."""
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = Solver(
        SolverConfig(stochasticity_scale={"mod_a": lambda t: jnp.array(-0.1)})
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        solver.integrate(gen_model, {"mod_a": jnp.zeros(SHAPE)}, key=jr.PRNGKey(0))


def test_solver_rejects_scale_negative_on_part_of_the_interval():
    """The sign check probes across the interval, so a scale that is only
    negative early (here for ``t < 0.5``) is caught too."""
    gen_model = toy_constant_velocity_gen_model(value=1.0)
    solver = Solver(
        SolverConfig(stochasticity_scale={"mod_a": lambda t: jnp.asarray(t) - 0.5})
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        solver.integrate(gen_model, {"mod_a": jnp.zeros(SHAPE)}, key=jr.PRNGKey(0))
