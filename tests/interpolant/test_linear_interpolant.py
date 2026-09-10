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

"""Tests for linear interpolant variable conversion methods."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import (
    one_sided_si,
    random_inputs_one_sided,
    two_sided_deterministic_si,
    two_sided_si,
)

from stix.core.interpolant.continuous_interpolant import (
    ContinuousOneSidedStochasticInterpolant,
)
from stix.core.interpolant.linear_interpolant import (
    LinearDeterministicInterpolant,
    LinearInterpolant,
    LinearStochasticInterpolant,
)
from stix.core.interpolant.standard_interpolants.diffusion import (
    VarianceExplodingDiffusionOneSidedInterpolant,
)
from stix.core.interpolant.standard_interpolants.flow_matching import (
    FlowMatchingOneSidedInterpolant,
)
from stix.typing import EmbeddedSourceTargetPair, Var, VarType

# Maps each VarType to the token used in the concrete converter method names,
# which follow a strict ``{output}_from_{input}`` (and
# ``{output}_from_{input_0}_and_{input_1}`` for pairs) naming convention.
_VARTYPE_TOKEN = {
    VarType.NOISE: "noise",
    VarType.SCORE: "score",
    VarType.EMBEDDED_TARGET: "target",
    VarType.VELOCITY: "velocity",
    VarType.VELOCITY_FIELD: "velocity_field",
}


def _convert(interpolant, input_type, output_type, var, z_t, t) -> Var:
    """Dispatch to an interpolant's concrete converter by naming convention.

    A small test-only shim that lets the round-trip tests stay parametrized over
    ``VarType`` pairs. Single conversions call
    ``{output}_from_{input}(var, z_t, t)``; pair conversions call
    ``{output}_from_{input_0}_and_{input_1}(var_0, var_1, z_t, t)`` (with
    the identity/extraction short-circuit when the output type is already present
    in the input).
    """
    if isinstance(input_type, tuple):
        if output_type in input_type:
            return var[input_type.index(output_type)]
        name = (
            f"{_VARTYPE_TOKEN[output_type]}_from_"
            f"{_VARTYPE_TOKEN[input_type[0]]}_and_{_VARTYPE_TOKEN[input_type[1]]}"
        )
        return getattr(interpolant, name)(var[0], var[1], z_t, t)
    if input_type == output_type:
        return var
    name = f"{_VARTYPE_TOKEN[output_type]}_from_{_VARTYPE_TOKEN[input_type]}"
    return getattr(interpolant, name)(var, z_t, t)


# ── Helpers ──
def _one_sided_stochastic_edm() -> VarianceExplodingDiffusionOneSidedInterpolant:
    """One-sided EDM schedule: ``gamma(t) = 80(1-t) + 1e-4 t`` and ``beta(t) = 1``."""
    return VarianceExplodingDiffusionOneSidedInterpolant(sigma_min=1e-4, sigma_max=80.0)


# ══════════════════════════════════════════════════════════════════════
# Single-conversion round-trip tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "type_a,type_b",
    [(VarType.NOISE, VarType.SCORE)],
    ids=["NOISE↔SCORE"],
)
def test_two_sided_stochastic_single_round_trip(type_a: VarType, type_b: VarType):
    """NOISE↔SCORE is the only single conversion for two-sided stochastic SI
    (velocity requires a pair: velocity_field + noise/score)."""
    interpolant = two_sided_si()
    var, z_t, t = random_inputs_one_sided(shape=(1, 3))

    via_b = _convert(interpolant, type_a, type_b, var, z_t, t)
    restored_a = _convert(interpolant, type_b, type_a, via_b, z_t, t)
    assert jnp.allclose(var, restored_a, atol=1e-6)

    via_a = _convert(interpolant, type_b, type_a, var, z_t, t)
    restored_b = _convert(interpolant, type_a, type_b, via_a, z_t, t)
    assert jnp.allclose(var, restored_b, atol=1e-6)


def test_deterministic_fm_velocity_identity():
    """Velocity↔VelocityField is identity for deterministic FM (gamma=0, dot_gamma=0)."""
    interpolant = two_sided_deterministic_si()
    var, z_t, t = random_inputs_one_sided(shape=(1, 3))
    var_1 = _convert(interpolant, VarType.VELOCITY_FIELD, VarType.VELOCITY, var, z_t, t)
    var_2 = _convert(interpolant, VarType.VELOCITY, VarType.VELOCITY_FIELD, var, z_t, t)
    var_3 = _convert(
        interpolant, VarType.VELOCITY_FIELD, VarType.VELOCITY_FIELD, var, z_t, t
    )
    var_4 = _convert(interpolant, VarType.VELOCITY, VarType.VELOCITY, var, z_t, t)
    var_5 = interpolant.velocity_from_velocity_field(var, z_t, t)
    var_6 = interpolant.velocity_field_from_velocity(var, z_t, t)
    assert jnp.allclose(var_1, var)
    assert jnp.allclose(var_2, var)
    assert jnp.allclose(var_3, var)
    assert jnp.allclose(var_4, var)
    assert jnp.allclose(var_5, var)
    assert jnp.allclose(var_6, var)


@pytest.mark.parametrize(
    "type_a,type_b",
    [
        (VarType.NOISE, VarType.SCORE),
        (VarType.NOISE, VarType.EMBEDDED_TARGET),
        (VarType.NOISE, VarType.VELOCITY_FIELD),
        (VarType.NOISE, VarType.VELOCITY),
        (VarType.SCORE, VarType.EMBEDDED_TARGET),
        (VarType.SCORE, VarType.VELOCITY_FIELD),
        (VarType.SCORE, VarType.VELOCITY),
        (VarType.EMBEDDED_TARGET, VarType.VELOCITY_FIELD),
        (VarType.EMBEDDED_TARGET, VarType.VELOCITY),
        (VarType.VELOCITY_FIELD, VarType.VELOCITY),
    ],
)
def test_one_sided_fm_single_round_trip(type_a: VarType, type_b: VarType):
    """All one-sided FM variable types are pairwise interconvertible."""
    interpolant = one_sided_si()
    var, z_t, t = random_inputs_one_sided(shape=(1, 3))

    via_b = _convert(interpolant, type_a, type_b, var, z_t, t)
    restored_a = _convert(interpolant, type_b, type_a, via_b, z_t, t)
    assert jnp.allclose(var, restored_a, atol=1e-6)

    via_a = _convert(interpolant, type_b, type_a, var, z_t, t)
    restored_b = _convert(interpolant, type_a, type_b, via_a, z_t, t)
    assert jnp.allclose(var, restored_b, atol=1e-6)


@pytest.mark.parametrize(
    "type_a,type_b",
    [
        (VarType.NOISE, VarType.SCORE),
        (VarType.NOISE, VarType.EMBEDDED_TARGET),
        (VarType.SCORE, VarType.EMBEDDED_TARGET),
        (VarType.VELOCITY, VarType.NOISE),
        (VarType.VELOCITY, VarType.SCORE),
    ],
)
def test_one_sided_edm_single_round_trip(type_a: VarType, type_b: VarType):
    """EDM subset: VELOCITY_FIELD conversions are excluded because dot_beta=0
    makes the velocity field identically zero, so VF↔X is non-invertible."""
    interpolant = _one_sided_stochastic_edm()
    var, z_t, t = random_inputs_one_sided(shape=(1, 3))

    via_b = _convert(interpolant, type_a, type_b, var, z_t, t)
    restored_a = _convert(interpolant, type_b, type_a, via_b, z_t, t)
    assert jnp.allclose(var, restored_a, atol=1e-6)

    via_a = _convert(interpolant, type_b, type_a, var, z_t, t)
    restored_b = _convert(interpolant, type_a, type_b, via_a, z_t, t)
    assert jnp.allclose(var, restored_b, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
# Pair-conversion round-trip tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "forward_pair,output_type",
    [
        ((VarType.VELOCITY_FIELD, VarType.NOISE), VarType.VELOCITY),
        ((VarType.VELOCITY_FIELD, VarType.SCORE), VarType.VELOCITY),
    ],
    ids=["(VF,NOISE)→VELOCITY", "(VF,SCORE)→VELOCITY"],
)
def test_two_sided_stochastic_fm_pair_round_trip(
    forward_pair: tuple[VarType, VarType], output_type: VarType
):
    """Pair round-trip: (A, B)→C then (C, B)→A, checking A is preserved."""
    interpolant = two_sided_si()
    var_first, z_t, t = random_inputs_one_sided(shape=(1, 3))
    var_second = jr.normal(jr.PRNGKey(99), shape=(1, 3))

    output = _convert(
        interpolant,
        forward_pair,
        output_type,
        (var_first, var_second),
        z_t,
        t,
    )
    reverse_pair = (output_type, forward_pair[1])
    restored = _convert(
        interpolant,
        reverse_pair,
        forward_pair[0],
        (output, var_second),
        z_t,
        t,
    )
    assert jnp.allclose(var_first, restored, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
# interpolant_fn behaviour
# ══════════════════════════════════════════════════════════════════════


def test_linear_interpolant_fn_with_none_source_and_alpha_fn_raises():
    interpolant = LinearInterpolant(
        gamma_fn=None,
        alpha_fn=lambda t: 1 - t,
        beta_fn=lambda t: t,
    )
    pairs = EmbeddedSourceTargetPair(target=jnp.array([3.0, -1.0]), source=None)
    with pytest.raises(
        ValueError,
        match="Source is None for linear interpolant while alpha_fn is not None.",
    ):
        interpolant.interpolant_fn(pairs, jnp.array(0.4))


@pytest.mark.parametrize(
    "interpolant",
    [
        LinearStochasticInterpolant(
            gamma_fn=lambda t: (t * (1 - t)) ** 0.5,
            alpha_fn=None,
            beta_fn=lambda t: t,
        ),
        LinearInterpolant(
            gamma_fn=None,
            alpha_fn=None,
            beta_fn=lambda t: t,
        ),
        LinearDeterministicInterpolant(
            alpha_fn=None,
            beta_fn=lambda t: t,
        ),
    ],
    ids=[
        "LinearStochasticInterpolant",
        "LinearInterpolant",
        "LinearDeterministicInterpolant",
    ],
)
def test_linear_interpolant_fn_without_alpha_is_beta_times_target(
    interpolant: LinearInterpolant,
):
    """alpha_fn=None uses beta(t)*target in LinearInterpolant.interpolant_fn (not one-sided)."""
    pairs = EmbeddedSourceTargetPair(
        target=jnp.array([2.0, 3.0]),
        source=jnp.array([1.0, 0.0]),
    )
    t = jnp.array(0.4)
    out = interpolant.interpolant_fn(pairs, t)
    assert jnp.allclose(out, interpolant.beta_fn(t) * pairs.target)


@pytest.mark.parametrize(
    "interpolant",
    [
        LinearStochasticInterpolant(
            gamma_fn=lambda t: (t * (1 - t)) ** 0.5,
            alpha_fn=lambda t: 1 - t,
            beta_fn=lambda t: t,
        ),
        LinearDeterministicInterpolant(
            alpha_fn=lambda t: 1 - t,
            beta_fn=lambda t: t,
        ),
        LinearInterpolant(
            gamma_fn=None,
            alpha_fn=lambda t: 1 - t,
            beta_fn=lambda t: t,
        ),
    ],
    ids=[
        "LinearStochasticInterpolant",
        "LinearDeterministicInterpolant",
        "LinearInterpolant",
    ],
)
def test_linear_interpolant_fn_two_sided_affine(interpolant: LinearInterpolant):
    """Two-sided schedule: alpha(t)*source + beta(t)*target when source is set."""
    source = jnp.array([1.0, 2.0])
    target = jnp.array([5.0, 7.0])
    pairs = EmbeddedSourceTargetPair(target=target, source=source)
    t = jnp.array(0.3)
    out = interpolant.interpolant_fn(pairs, t)
    expected = (1 - t) * source + t * target
    assert jnp.allclose(out, expected)


def test_one_sided_interpolant_fn_ignores_source():
    """One-sided SI should ignore source entirely: z_t = beta(t)*target."""
    interpolant = FlowMatchingOneSidedInterpolant()
    pairs = EmbeddedSourceTargetPair(
        target=jnp.array([2.0, -4.0]),
        source=jnp.array([999.0, 999.0]),
    )
    t = jnp.array(0.25)
    out = ContinuousOneSidedStochasticInterpolant.interpolant_fn(interpolant, pairs, t)
    assert jnp.allclose(out, interpolant.beta_fn(t) * pairs.target)


# ══════════════════════════════════════════════════════════════════════
# Ground-truth formula verification
#
# Round-trip tests (A→B→A) can't catch symmetric formula bugs. These tests
# verify each individual converter against analytically derived values.
# ══════════════════════════════════════════════════════════════════════


# ── One-sided ground truth: schedule configs ──


def _one_sided_fm_ground_truth() -> dict:
    """Analytically computed ground-truth values for one-sided FM.

    FM one-sided: gamma(t) = 1-t, beta(t) = t.
    Given known z1, epsilon, t:
        z_t            = t * z1 + (1-t) * epsilon
        score          = -epsilon / (1-t)
        velocity_field = z1                     (dot_beta=1)
        velocity       = z1 - epsilon           (dot_gamma=-1)
        noise          = epsilon
    """
    t = jnp.array(0.3)
    z1 = jnp.array([2.0, -1.5, 0.7])
    epsilon = jnp.array([0.5, -0.8, 1.2])
    z_t = t * z1 + (1 - t) * epsilon
    return {
        "t": t,
        "target": z1,
        "epsilon": epsilon,
        "z_t": z_t,
        "score": -epsilon / (1 - t),
        "velocity_field": z1,
        "velocity": z1 - epsilon,
    }


_ONE_SIDED_FM_GT_CONVERSIONS = [
    # (method_name, input_key, expected_key)
    ("score_from_target", "target", "score"),
    ("score_from_velocity_field", "velocity_field", "score"),
    ("score_from_velocity", "velocity", "score"),
    ("velocity_from_noise", "epsilon", "velocity"),
    ("velocity_from_score", "score", "velocity"),
    ("velocity_from_target", "target", "velocity"),
    ("velocity_from_velocity_field", "velocity_field", "velocity"),
    ("velocity_field_from_noise", "epsilon", "velocity_field"),
    ("velocity_field_from_score", "score", "velocity_field"),
    ("velocity_field_from_velocity", "velocity", "velocity_field"),
    ("velocity_field_from_target", "target", "velocity_field"),
    ("noise_from_target", "target", "epsilon"),
    ("noise_from_velocity_field", "velocity_field", "epsilon"),
    ("noise_from_velocity", "velocity", "epsilon"),
    ("target_from_noise", "epsilon", "target"),
    ("target_from_score", "score", "target"),
    ("target_from_velocity_field", "velocity_field", "target"),
    ("target_from_velocity", "velocity", "target"),
]


def _one_sided_edm_ground_truth() -> dict:
    """Analytically computed ground-truth values for one-sided EDM.

    EDM one-sided: gamma(t) = 80(1-t) + 1e-4*t, beta(t) = 1.
    dot_gamma = -80 + 1e-4, dot_beta = 0.
    Given known z1, epsilon, t:
        z_t            = z1 + gamma_t * epsilon
        score          = -epsilon / gamma_t
        velocity_field = 0                       (dot_beta=0)
        velocity       = dot_gamma * epsilon
        noise          = epsilon
    """
    t = jnp.array(0.3)
    z1 = jnp.array([2.0, -1.5, 0.7])
    epsilon = jnp.array([0.5, -0.8, 1.2])
    gamma_t = 80 * (1 - t) + 1e-4 * t
    dot_gamma_t = -80 + 1e-4
    z_t = z1 + gamma_t * epsilon
    return {
        "t": t,
        "target": z1,
        "epsilon": epsilon,
        "z_t": z_t,
        "score": -epsilon / gamma_t,
        "velocity_field": jnp.zeros_like(z1),
        "velocity": dot_gamma_t * epsilon,
    }


# NOTE: EDM has dot_beta=0, so any conversion that divides by dot_beta is
# undefined (e.g. *_from_velocity_field, target_from_velocity, velocity_field_from_velocity).
# Only conversions that don't require dot_beta in a denominator are tested.
_ONE_SIDED_EDM_GT_CONVERSIONS = [
    # (method_name, input_key, expected_key)
    ("score_from_target", "target", "score"),
    ("score_from_velocity", "velocity", "score"),
    ("velocity_from_noise", "epsilon", "velocity"),
    ("velocity_from_score", "score", "velocity"),
    ("velocity_from_target", "target", "velocity"),
    ("velocity_field_from_noise", "epsilon", "velocity_field"),
    ("velocity_field_from_score", "score", "velocity_field"),
    ("velocity_field_from_target", "target", "velocity_field"),
    ("noise_from_target", "target", "epsilon"),
    ("noise_from_velocity", "velocity", "epsilon"),
    ("target_from_noise", "epsilon", "target"),
    ("target_from_score", "score", "target"),
]

_ONE_SIDED_GT_CONFIGS = {
    "fm": (
        one_sided_si,
        _one_sided_fm_ground_truth,
        _ONE_SIDED_FM_GT_CONVERSIONS,
    ),
    "edm": (
        _one_sided_stochastic_edm,
        _one_sided_edm_ground_truth,
        _ONE_SIDED_EDM_GT_CONVERSIONS,
    ),
}

_ONE_SIDED_GT_CASES = [
    (schedule, method, input_key, expected_key)
    for schedule, (_, _, conversions) in _ONE_SIDED_GT_CONFIGS.items()
    for method, input_key, expected_key in conversions
]


@pytest.mark.parametrize(
    "schedule,method_name,input_key,expected_key",
    _ONE_SIDED_GT_CASES,
    ids=[f"{schedule}-{method}" for schedule, method, _, _ in _ONE_SIDED_GT_CASES],
)
def test_one_sided_ground_truth(
    schedule: str, method_name: str, input_key: str, expected_key: str
):
    """Verify each one-sided converter against analytically derived values.

    All one-sided converters have signature (var, z_t, t) where
    z_t = z_t (the interpolated value at time t).
    """
    factory, gt_factory, _ = _ONE_SIDED_GT_CONFIGS[schedule]
    interpolant = factory()
    gt = gt_factory()
    result = getattr(interpolant, method_name)(gt[input_key], gt["z_t"], gt["t"])
    assert jnp.allclose(result, gt[expected_key], atol=1e-6)


# ── Two-sided FM ground truth ──


def _two_sided_fm_ground_truth() -> dict:
    r"""Analytically computed ground-truth values for two-sided FM.

    FM two-sided: gamma(t) = sqrt(t(1-t)), alpha(t) = 1-t, beta(t) = t.
    Given known z0, z1, epsilon, t:
        z_t            = (1-t)*z0 + t*z1 + sqrt(t(1-t))*epsilon
        velocity_field = z1 - z0
        velocity       = (z1 - z0) + dot_gamma * epsilon
        score          = -epsilon / sqrt(t(1-t))
        noise          = epsilon

    where dot_gamma = (1-2t) / (2*sqrt(t(1-t))).
    """
    t = jnp.array(0.3)
    z0 = jnp.array([1.0, -0.5, 0.3])
    z1 = jnp.array([2.0, -1.5, 0.7])
    epsilon = jnp.array([0.5, -0.8, 1.2])

    gamma_t = (t * (1 - t)) ** 0.5
    dot_gamma_t = (1 - 2 * t) / (2 * (t * (1 - t)) ** 0.5)
    z_t = (1 - t) * z0 + t * z1 + gamma_t * epsilon

    return {
        "t": t,
        "z0": z0,
        "z1": z1,
        "epsilon": epsilon,
        "z_t": z_t,
        "gamma_t": gamma_t,
        "dot_gamma_t": dot_gamma_t,
        "score": -epsilon / gamma_t,
        "velocity_field": z1 - z0,
        "velocity": (z1 - z0) + dot_gamma_t * epsilon,
    }


# Two-sided single conversions: score↔noise (signature: method(var, t), no z_t)
_TWO_SIDED_FM_GT_SINGLE_CONVERSIONS = [
    # (method_name, input_key, expected_key)
    ("score_from_noise", "epsilon", "score"),
    ("noise_from_score", "score", "epsilon"),
]


@pytest.mark.parametrize(
    "method_name,input_key,expected_key",
    _TWO_SIDED_FM_GT_SINGLE_CONVERSIONS,
    ids=[m for m, _, _ in _TWO_SIDED_FM_GT_SINGLE_CONVERSIONS],
)
def test_two_sided_fm_ground_truth_single(
    method_name: str, input_key: str, expected_key: str
):
    """Verify two-sided single converters (score↔noise) against analytical values.

    Single converters take (var, z_t, t); z_t is unused by the
    score↔noise formulas but is still part of the uniform call signature.
    """
    interpolant = two_sided_si()
    gt = _two_sided_fm_ground_truth()
    result = getattr(interpolant, method_name)(gt[input_key], gt["z_t"], gt["t"])
    assert jnp.allclose(result, gt[expected_key], atol=1e-6)


# Two-sided pair conversions: (VF, noise/score) ↔ velocity
_TWO_SIDED_FM_GT_PAIR_CONVERSIONS = [
    # (method_name, input_key_0, input_key_1, expected_key)
    ("velocity_from_velocity_field_and_noise", "velocity_field", "epsilon", "velocity"),
    ("velocity_field_from_velocity_and_noise", "velocity", "epsilon", "velocity_field"),
    ("velocity_from_velocity_field_and_score", "velocity_field", "score", "velocity"),
    ("velocity_field_from_velocity_and_score", "velocity", "score", "velocity_field"),
]


@pytest.mark.parametrize(
    "method_name,input_key_0,input_key_1,expected_key",
    _TWO_SIDED_FM_GT_PAIR_CONVERSIONS,
    ids=[m for m, _, _, _ in _TWO_SIDED_FM_GT_PAIR_CONVERSIONS],
)
def test_two_sided_fm_ground_truth_pair(
    method_name: str, input_key_0: str, input_key_1: str, expected_key: str
):
    """Verify two-sided pair converters against analytical values.

    Pair converters have signature (var_0, var_1, t).
    """
    interpolant = two_sided_si()
    gt = _two_sided_fm_ground_truth()
    result = getattr(interpolant, method_name)(
        gt[input_key_0], gt[input_key_1], gt["z_t"], gt["t"]
    )
    assert jnp.allclose(result, gt[expected_key], atol=1e-6)


def test_one_sided_sample_initial_state_is_gamma_0_times_epsilon():
    """One-sided linear path: z_0 = gamma(0) epsilon, z_src unused."""
    interpolant = one_sided_si()
    epsilon = jnp.array([1.5, -2.0])
    z0 = interpolant.sample_initial_state(None, epsilon)
    assert jnp.allclose(z0, interpolant.gamma_fn(jnp.asarray(0.0)) * epsilon)
    ignored_src = jnp.array([99.0, 99.0])
    assert jnp.allclose(interpolant.sample_initial_state(ignored_src, epsilon), z0)


def test_two_sided_sample_initial_state_drops_target_term():
    """Two-sided linear path: z_0 = alpha(0) z_src + gamma(0) epsilon."""
    interpolant = two_sided_si()
    z_src = jnp.array([3.0, -1.0])
    epsilon = jnp.array([0.5, 0.25])
    t0 = jnp.asarray(0.0)
    z0 = interpolant.sample_initial_state(z_src, epsilon)
    assert interpolant.alpha_fn is not None  # two-sided: both schedules are set
    expected = interpolant.alpha_fn(t0) * z_src + interpolant.gamma_fn(t0) * epsilon
    assert jnp.allclose(z0, expected)


def test_two_sided_sample_initial_state_requires_source():
    interpolant = two_sided_deterministic_si()
    with pytest.raises(ValueError, match="z_src is None"):
        interpolant.sample_initial_state(None, jnp.zeros(2))
