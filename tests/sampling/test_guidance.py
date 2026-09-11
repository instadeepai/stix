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

"""Tests for the guidance recipes in :mod:`stix.sampling.guidance`.

Recipes now return per-modality
:class:`~stix.core.generator.Generator` objects; these tests read the
``.score`` field off the returned ``VelocityAndScore`` generators to check the
score arithmetic (the velocity is a deterministic function of that score via the
interpolant, exercised separately).
"""

import warnings

import jax
import jax.numpy as jnp
import pytest
from conftest import (
    SHAPE,
    AddContextBackbone,
    BroadcastTimeEncoder,
    ConstantContextEncoder,
    GuidedVelocityOneSidedGenerativeModel,
    IdentityDecoder,
    IdentityEncoder,
    VelocityOneSidedGenerativeModel,
    build_one_sided_registry,
    fill_intrinsic_mask,
    one_sided_si,
    two_sided_si,
)
from flax import nnx
from jaxtyping import PyTree

from stix.core.embedder import (
    IdentityEmbedder,
    LearnedDiscreteEmbedder,
    OneHotDiscreteEmbedder,
)
from stix.core.gen_model import GenerativeModel
from stix.core.generator import Velocity, VelocityAndScore
from stix.core.loss import CrossEntropyCriterion, MSECriterion
from stix.core.modality import Modality, ModalityRegistry
from stix.nn import EncoderBackboneDecoderNetwork, Network, SumContextEncoder
from stix.sampling.guidance import (
    _with_conditional_score,
    get_classifier_free_guidance_generator,
    get_context_conditional_generator,
    get_intrinsic_guidance_generator,
)
from stix.typing import Mask, Var


def _scores(generators: PyTree) -> dict:
    """Read the per-modality ``.score`` off a pytree of ``VelocityAndScore``."""
    return {k: g.score for k, g in generators.items()}


def _scores_from_net_out(gen_model, net_out, z_t, t) -> dict:
    """Per-modality scores of the generator built from ``net_out``."""
    return _scores(gen_model.get_generator(net_out, z_t, t))


# ── Local fixtures ──


class ConditionAwareConstantNetwork(Network):
    """Toy network mirroring the context encoder's lenient None contract.

    - ``context_data is None``  →  bias = ``bias_unconditioned`` (fully unconditional).
    - ``context_data`` provided, ``context_mask["context"]`` is ``None``
      or absent  →  bias = ``bias_conditioned`` (fully conditional).
    - ``context_data`` provided, ``context_mask["context"]`` is a real
      boolean/float array  →  bias = mask · cond + (1-mask) · uncond
      (per-sample linear interpolation, mirroring ``ctx * mask`` downstream).

    Has two trainable nnx.Params so ``nnx.split``/``merge`` works correctly.
    """

    def __init__(self, value_unconditioned: float, value_conditioned: float):
        self.bias_unconditioned = nnx.Param(jnp.array(value_unconditioned))
        self.bias_conditioned = nnx.Param(jnp.array(value_conditioned))

    def __call__(
        self, z_t, t, context_data=None, context_mask=None, attention_mask=None
    ):
        del t, attention_mask  # accepted for protocol compatibility
        if context_data is None:
            bias = jnp.broadcast_to(self.bias_unconditioned[...], SHAPE)
        else:
            mask = context_mask.get("context") if context_mask is not None else None
            if mask is None:
                bias = jnp.broadcast_to(self.bias_conditioned[...], SHAPE)
            else:
                mask_f = mask.astype(self.bias_conditioned.dtype)
                bias = (
                    mask_f * self.bias_conditioned
                    + (1 - mask_f) * self.bias_unconditioned
                )
        return {k: jnp.broadcast_to(bias, v.shape) for k, v in z_t.items()}


def condition_aware_gen_model(
    value_unconditioned: float, value_conditioned: float
) -> GuidedVelocityOneSidedGenerativeModel:
    """One-sided gen-model whose net_out depends on whether context_data is None.

    Single modality ``mod_a``. Mirrors ``toy_constant_velocity_gen_model`` but
    uses :class:`ConditionAwareConstantNetwork`. VELOCITY output.
    """
    return GuidedVelocityOneSidedGenerativeModel(
        network=ConditionAwareConstantNetwork(value_unconditioned, value_conditioned),
        modality_registry=build_one_sided_registry(),
    )


@pytest.fixture
def gen_model_uncond_1_cond_2() -> GuidedVelocityOneSidedGenerativeModel:
    """Condition-aware gen-model with ``bias_unconditioned=1.0``, ``bias_conditioned=2.0``."""
    return condition_aware_gen_model(value_unconditioned=1.0, value_conditioned=2.0)


@pytest.fixture
def z_t_all_ones() -> PyTree[Var]:
    """Single-modality noisy state: ``{"mod_a": ones(SHAPE)}``."""
    return {"mod_a": jnp.ones(SHAPE)}


@pytest.fixture
def t_half() -> jax.Array:
    """Middle-of-trajectory time ``t=0.5``."""
    return jnp.array(0.5)


# ══════════════════════════════════════════════════════════════════════
# get_context_conditional_generator
# ══════════════════════════════════════════════════════════════════════


def test_get_context_conditional_generator_forwards_context_mask(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """Recipe forwards ``context_mask`` to the network: mask=0 (unconditional)
    and mask=1 (conditional) yield different scores when the network is
    mask-aware.
    """
    bias_unconditioned = 1.0  # matches gen_model_uncond_1_cond_2 fixture
    bias_conditioned = 2.0
    context_data = {"context": jnp.zeros(SHAPE)}

    score_uncond = _scores(
        get_context_conditional_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            context_data=context_data,
            context_mask={"context": jnp.zeros(SHAPE)},
            attention_mask=None,
            guidance_scale={},
        )
    )
    score_cond = _scores(
        get_context_conditional_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            context_data=context_data,
            context_mask={"context": jnp.ones(SHAPE)},
            attention_mask=None,
            guidance_scale={},
        )
    )

    # mask=0 ⇒ network bias = bias_unconditioned; mask=1 ⇒ bias_conditioned.
    # Toy network emits this bias as VELOCITY; recipe converts to SCORE via SI.
    si = gen_model_uncond_1_cond_2.modality_registry.registry["mod_a"].interpolant
    expected_uncond = si.score_from_velocity(
        jnp.full(SHAPE, bias_unconditioned),
        z_t_all_ones["mod_a"],
        t_half,
    )
    expected_cond = si.score_from_velocity(
        jnp.full(SHAPE, bias_conditioned),
        z_t_all_ones["mod_a"],
        t_half,
    )

    assert set(score_uncond.keys()) == {"mod_a"}
    assert score_uncond["mod_a"].shape == SHAPE
    assert set(score_cond.keys()) == {"mod_a"}
    assert score_cond["mod_a"].shape == SHAPE
    assert jnp.allclose(score_uncond["mod_a"], expected_uncond)
    assert jnp.allclose(score_cond["mod_a"], expected_cond)


def test_get_context_conditional_generator_jit_compiles(gen_model_uncond_1_cond_2):
    """Recipe traces and runs cleanly under ``jax.jit`` via the standard
    ``nnx.split`` / ``nnx.merge`` pattern used by the solvers.
    """
    graphdef, state = nnx.split(gen_model_uncond_1_cond_2)

    @jax.jit
    def jitted_recipe(state, z_t, t):
        merged = nnx.merge(graphdef, state)
        return get_context_conditional_generator(
            merged,
            z_t,
            t,
            attention_mask=None,
            guidance_scale={},
        )

    z_t = {"mod_a": jnp.ones(SHAPE)}
    generators = jitted_recipe(state, z_t, jnp.array(0.5))
    score = _scores(generators)

    assert score["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(score["mod_a"]))


# ══════════════════════════════════════════════════════════════════════
# get_intrinsic_guidance_generator
# ══════════════════════════════════════════════════════════════════════


def test_intrinsic_guidance_generator_shape(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """Recipe returns one entry per gen_model modality with shape matching ``z_t_all_ones``."""
    context_data = {"mod_a": jnp.zeros(SHAPE)}
    context_mask: PyTree[Mask | None] = {"mod_a": jnp.ones(SHAPE)}
    guidance_scale = {"mod_a": lambda t: jnp.array(1.0)}

    score = _scores(
        get_intrinsic_guidance_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            intrinsic_data=context_data,
            intrinsic_mask=context_mask,
            attention_mask=None,
            guidance_scale=guidance_scale,
        )
    )

    assert set(score.keys()) == {"mod_a"}
    assert score["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(score["mod_a"]))


def test_intrinsic_guidance_generator_omega_zero_recovers_network_score(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """With ``guidance_scale = 0`` the gradient term zeros out and the
    recipe collapses to the same score as a plain network forward.

    Documents the additive structure: ``s_cond = s_uncond - γ·∇L``.
    """
    intrinsic_data = {"mod_a": jnp.zeros(SHAPE)}
    intrinsic_mask: PyTree[Mask | None] = {"mod_a": jnp.ones(SHAPE)}
    zero_scale = {"mod_a": lambda t: jnp.array(0.0)}

    score_intrinsic = _scores(
        get_intrinsic_guidance_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            intrinsic_data=intrinsic_data,
            intrinsic_mask=intrinsic_mask,
            attention_mask=None,
            guidance_scale=zero_scale,
        )
    )
    score_network = _scores(
        get_context_conditional_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            attention_mask=None,
            guidance_scale={},
        )
    )

    assert jnp.allclose(score_intrinsic["mod_a"], score_network["mod_a"])


def test_intrinsic_guidance_generator_correct(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """Recipe score equals ``s_uncond - γ · ∇L`` computed independently.

    Pins down sign and scaling of the gradient term.
    """
    intrinsic_data = {"mod_a": jnp.zeros(SHAPE)}
    intrinsic_mask: PyTree[Mask | None] = {"mod_a": jnp.ones(SHAPE)}
    omega_val = 0.7

    net_out = gen_model_uncond_1_cond_2.get_network_output(
        z_t_all_ones, t_half, None, None, None
    )
    score_uncond = _scores_from_net_out(
        gen_model_uncond_1_cond_2, net_out, z_t_all_ones, t_half
    )

    def loss_of_z(z):
        no = gen_model_uncond_1_cond_2.get_network_output(z, t_half, None, None, None)
        return gen_model_uncond_1_cond_2.get_guidance_loss(
            no, z, t_half, intrinsic_data, intrinsic_mask
        )

    grad_loss = jax.grad(loss_of_z)(z_t_all_ones)
    expected = score_uncond["mod_a"] - omega_val * grad_loss["mod_a"]

    actual = _scores(
        get_intrinsic_guidance_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            intrinsic_data=intrinsic_data,
            intrinsic_mask=intrinsic_mask,
            attention_mask=None,
            guidance_scale={"mod_a": lambda t: jnp.array(omega_val)},
        )
    )

    assert jnp.allclose(actual["mod_a"], expected)


def test_intrinsic_guidance_generator_raises_on_none_intrinsic_data(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """Loud-failure contract: intrinsic recipe with ``intrinsic_data=None``
    raises rather than silently degenerating to unconditional.
    """
    guidance_scale = {"mod_a": lambda t: jnp.array(1.0)}

    with pytest.raises(ValueError):
        get_intrinsic_guidance_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            attention_mask=None,
            guidance_scale=guidance_scale,
        )


def test_intrinsic_guidance_generator_raises_without_get_guidance_loss(
    z_t_all_ones, t_half
):
    """A model that does not override ``get_guidance_loss`` cannot drive intrinsic
    guidance: the recipe differentiates that hook, and the base implementation
    raises ``NotImplementedError`` naming it.
    """
    gen_model = VelocityOneSidedGenerativeModel(
        network=ConditionAwareConstantNetwork(1.0, 2.0),
        modality_registry=build_one_sided_registry(),
    )

    with pytest.raises(NotImplementedError, match="get_guidance_loss"):
        get_intrinsic_guidance_generator(
            gen_model,
            z_t_all_ones,
            t_half,
            intrinsic_data={"mod_a": jnp.zeros(SHAPE)},
            intrinsic_mask={"mod_a": jnp.ones(SHAPE)},
            attention_mask=None,
            guidance_scale={"mod_a": lambda t: jnp.array(1.0)},
        )


def test_intrinsic_guidance_generator_jit_compiles(gen_model_uncond_1_cond_2):
    """Recipe traces under ``jax.jit`` via the standard
    ``nnx.split`` / ``nnx.merge`` pattern.
    """
    graphdef, state = nnx.split(gen_model_uncond_1_cond_2)
    guidance_scale = {"mod_a": lambda t: jnp.array(1.0)}

    @jax.jit
    def jitted_recipe(state, z_t, t, intrinsic_data, intrinsic_mask):
        merged = nnx.merge(graphdef, state)
        return get_intrinsic_guidance_generator(
            merged,
            z_t,
            t,
            intrinsic_data=intrinsic_data,
            intrinsic_mask=intrinsic_mask,
            attention_mask=None,
            guidance_scale=guidance_scale,
        )

    z_t = {"mod_a": jnp.ones(SHAPE)}
    intrinsic_data = {"mod_a": jnp.zeros(SHAPE)}
    intrinsic_mask = {"mod_a": jnp.ones(SHAPE)}
    score = _scores(
        jitted_recipe(state, z_t, jnp.array(0.5), intrinsic_data, intrinsic_mask)
    )

    assert score["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(score["mod_a"]))


def test_intrinsic_guidance_generator_with_strict_sum_context_encoder():
    """Regression: the intrinsic recipe must never feed ``intrinsic_data`` to
    the network forward. A real ``SumContextEncoder`` with
    ``context_encoders=None`` raises on any non-None context data — so if the
    recipe leaked the intrinsic channel into the forward, the network call would
    crash.
    """

    network = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": IdentityEncoder()},
        backbone=AddContextBackbone(),
        decoders={"mod_a": IdentityDecoder()},
        context_encoder=SumContextEncoder(
            time_encoder=BroadcastTimeEncoder(shape=SHAPE), context_encoders=None
        ),
    )
    gen_model = GuidedVelocityOneSidedGenerativeModel(
        network=network,
        modality_registry=build_one_sided_registry(),
    )

    z_t = {"mod_a": jnp.ones(SHAPE)}
    score = _scores(
        get_intrinsic_guidance_generator(
            gen_model,
            z_t,
            jnp.array(0.5),
            context_data=None,
            intrinsic_data={"mod_a": jnp.zeros(SHAPE)},
            intrinsic_mask={"mod_a": jnp.ones(SHAPE)},
            attention_mask=None,
            guidance_scale={"mod_a": lambda t: jnp.array(1.0)},
        )
    )

    assert score["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(score["mod_a"]))


def test_intrinsic_guidance_generator_threads_gradient_through_network(t_half):
    """Infra invariant: the recipe must differentiate the intrinsic loss through
    the network forward, not just through the loss's direct ``z_t`` dependence.
    """
    network = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": IdentityEncoder()},
        backbone=AddContextBackbone(),
        decoders={"mod_a": IdentityDecoder()},
        context_encoder=SumContextEncoder(
            time_encoder=BroadcastTimeEncoder(shape=SHAPE), context_encoders=None
        ),
    )
    gen_model = GuidedVelocityOneSidedGenerativeModel(
        network=network,
        modality_registry=build_one_sided_registry(),
    )

    z_t = {"mod_a": jnp.ones(SHAPE)}
    intrinsic_data = {"mod_a": jnp.full(SHAPE, 3.0)}
    intrinsic_mask: PyTree[Mask | None] = {"mod_a": jnp.ones(SHAPE)}
    omega = 0.7
    guidance_scale = {"mod_a": lambda _t: jnp.array(omega)}

    net_out = gen_model.get_network_output(z_t, t_half, None, None, None)
    score_uncond = _scores_from_net_out(gen_model, net_out, z_t, t_half)["mod_a"]

    def loss_through_network(z):
        no = gen_model.get_network_output(z, t_half, None, None, None)
        return gen_model.get_guidance_loss(
            no, z, t_half, intrinsic_data, intrinsic_mask
        )

    def loss_network_detached(z):
        no = jax.lax.stop_gradient(
            gen_model.get_network_output(z, t_half, None, None, None)
        )
        return gen_model.get_guidance_loss(
            no, z, t_half, intrinsic_data, intrinsic_mask
        )

    grad_full = jax.grad(loss_through_network)(z_t)["mod_a"]
    grad_detached = jax.grad(loss_network_detached)(z_t)["mod_a"]
    expected_full = score_uncond - omega * grad_full
    expected_detached = score_uncond - omega * grad_detached

    actual = _scores(
        get_intrinsic_guidance_generator(
            gen_model,
            z_t,
            t_half,
            intrinsic_data=intrinsic_data,
            intrinsic_mask=intrinsic_mask,
            attention_mask=None,
            guidance_scale=guidance_scale,
        )
    )["mod_a"]

    assert float(jnp.max(jnp.abs(expected_full - expected_detached))) > 1e-3
    assert jnp.allclose(actual, expected_full)


# ══════════════════════════════════════════════════════════════════════
# get_classifier_free_guidance_generator
# ══════════════════════════════════════════════════════════════════════


def test_cfg_generator_shape(gen_model_uncond_1_cond_2, z_t_all_ones, t_half):
    """Recipe returns one entry per gen_model modality with shape matching ``z_t_all_ones``."""
    context_data = {"context": jnp.zeros(SHAPE)}
    context_mask: PyTree[Mask | None] = {"context": jnp.ones(SHAPE)}
    guidance_scale = {"mod_a": lambda t: jnp.array(1.5)}

    score = _scores(
        get_classifier_free_guidance_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            context_data=context_data,
            context_mask=context_mask,
            attention_mask=None,
            guidance_scale=guidance_scale,
        )
    )

    assert set(score.keys()) == {"mod_a"}
    assert score["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(score["mod_a"]))


def test_cfg_generator_omega_one_recovers_conditional_branch(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """With ``γ=1``, the convex combination collapses to the conditional-branch
    score (forward with ``context_data``).
    """
    context_data = {"context": jnp.zeros(SHAPE)}
    context_mask: PyTree[Mask | None] = {"context": jnp.ones(SHAPE)}

    net_out_cond = gen_model_uncond_1_cond_2.get_network_output(
        z_t_all_ones, t_half, context_data, context_mask, None
    )
    expected = _scores_from_net_out(
        gen_model_uncond_1_cond_2, net_out_cond, z_t_all_ones, t_half
    )

    actual = _scores(
        get_classifier_free_guidance_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            context_data=context_data,
            context_mask=context_mask,
            attention_mask=None,
            guidance_scale={"mod_a": lambda t: jnp.array(1.0)},
        )
    )

    assert jnp.allclose(actual["mod_a"], expected["mod_a"])


def test_cfg_generator_omega_zero_recovers_unconditional_branch(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """With ``γ=0``, the convex combination collapses to the unconditional-branch
    score. The recipe expresses that branch by passing ``context_data=None,
    context_mask=None`` to the network forward.
    """
    context_data = {"context": jnp.zeros(SHAPE)}
    context_mask: PyTree[Mask | None] = {"context": jnp.ones(SHAPE)}

    net_out_uncond = gen_model_uncond_1_cond_2.get_network_output(
        z_t_all_ones, t_half, None, None, None
    )
    expected = _scores_from_net_out(
        gen_model_uncond_1_cond_2, net_out_uncond, z_t_all_ones, t_half
    )

    actual = _scores(
        get_classifier_free_guidance_generator(
            gen_model_uncond_1_cond_2,
            z_t_all_ones,
            t_half,
            context_data=context_data,
            context_mask=context_mask,
            attention_mask=None,
            guidance_scale={"mod_a": lambda t: jnp.array(0.0)},
        )
    )

    assert jnp.allclose(actual["mod_a"], expected["mod_a"])


def test_cfg_generator_jit_compiles(gen_model_uncond_1_cond_2):
    """Recipe traces under ``jax.jit`` via the standard
    ``nnx.split`` / ``nnx.merge`` pattern.
    """
    graphdef, state = nnx.split(gen_model_uncond_1_cond_2)
    guidance_scale = {"mod_a": lambda t: jnp.array(1.5)}

    @jax.jit
    def jitted_recipe(state, z_t, t, context_data, context_mask):
        merged = nnx.merge(graphdef, state)
        return get_classifier_free_guidance_generator(
            merged,
            z_t,
            t,
            context_data=context_data,
            context_mask=context_mask,
            attention_mask=None,
            guidance_scale=guidance_scale,
        )

    z_t = {"mod_a": jnp.ones(SHAPE)}
    context_data = {"context": jnp.zeros(SHAPE)}
    context_mask = {"context": jnp.ones(SHAPE)}
    score = _scores(
        jitted_recipe(state, z_t, jnp.array(0.5), context_data, context_mask)
    )

    assert score["mod_a"].shape == SHAPE
    assert jnp.all(jnp.isfinite(score["mod_a"]))


def test_cfg_generator_integrates_with_real_network(z_t_all_ones, t_half):
    """Integration: CFG recipe + a real ``EncoderBackboneDecoderNetwork``.

    Verifies the recipe and the context encoder compose correctly end-to-end:
    conditional (γ=1) and unconditional (γ=0) forwards produce *different*
    outputs, with exact expected values.
    """

    rngs = nnx.Rngs(0)
    net = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": IdentityEncoder(rngs=rngs)},
        backbone=AddContextBackbone(rngs=rngs),
        decoders={"mod_a": IdentityDecoder(rngs=rngs)},
        context_encoder=SumContextEncoder(
            time_encoder=ConstantContextEncoder(value=1.0, shape=SHAPE, rngs=rngs),
            context_encoders={
                "label": ConstantContextEncoder(value=10.0, shape=SHAPE, rngs=rngs),
            },
        ),
    )
    gen_model = VelocityOneSidedGenerativeModel(
        network=net,
        modality_registry=build_one_sided_registry(),
    )

    context_data = {"label": jnp.zeros(SHAPE)}
    context_mask: PyTree[Mask | None] | None = None

    score_cond = _scores(
        get_classifier_free_guidance_generator(
            gen_model,
            z_t_all_ones,
            t_half,
            context_data=context_data,
            context_mask=context_mask,
            attention_mask=None,
            guidance_scale={"mod_a": lambda t: jnp.array(1.0)},
        )
    )
    score_uncond = _scores(
        get_classifier_free_guidance_generator(
            gen_model,
            z_t_all_ones,
            t_half,
            context_data=context_data,
            context_mask=context_mask,
            attention_mask=None,
            guidance_scale={"mod_a": lambda t: jnp.array(0.0)},
        )
    )

    si = gen_model.modality_registry.registry["mod_a"].interpolant
    expected_cond = si.score_from_velocity(
        jnp.full(SHAPE, 12.0),
        z_t_all_ones["mod_a"],
        t_half,
    )
    expected_uncond = si.score_from_velocity(
        jnp.full(SHAPE, 2.0),
        z_t_all_ones["mod_a"],
        t_half,
    )
    assert score_cond["mod_a"].shape == SHAPE
    assert score_uncond["mod_a"].shape == SHAPE
    assert jnp.allclose(score_cond["mod_a"], expected_cond)
    assert jnp.allclose(score_uncond["mod_a"], expected_uncond)


# ══════════════════════════════════════════════════════════════════════
# _with_conditional_score (score -> VelocityAndScore bridge)
# ══════════════════════════════════════════════════════════════════════


def test_with_conditional_score_matches_si_convert(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """The helper packages a ``VelocityAndScore`` whose velocity is exactly
    ``si.velocity_from_score(...)`` and whose score is the input score.
    """
    conditional_score = jnp.array([0.5, -0.3, 1.2])
    modality = gen_model_uncond_1_cond_2.modality_registry.registry["mod_a"]

    expected_velocity = modality.interpolant.velocity_from_score(
        conditional_score, z_t_all_ones["mod_a"], t_half
    )

    generator = _with_conditional_score(
        modality, conditional_score, z_t_all_ones["mod_a"], t_half
    )

    assert isinstance(generator, VelocityAndScore)
    assert generator.velocity.shape == SHAPE
    assert jnp.allclose(generator.velocity, expected_velocity)
    assert jnp.allclose(generator.score, conditional_score)


def test_with_conditional_score_non_one_sided_raises(z_t_all_ones, t_half):
    """Fail-loud guard: the score->velocity bridge needs ``velocity_from_score``.
    A modality whose interpolant lacks it (a two-sided
    ``LinearStochasticInterpolant``) must raise a clear ``TypeError``.
    """
    modality = Modality(
        shape=SHAPE,
        is_discrete=False,
        interpolant=two_sided_si(),
        embedder=IdentityEmbedder(dm_shape=SHAPE),
    )

    with pytest.raises(TypeError, match="velocity_from_score"):
        _with_conditional_score(
            modality, jnp.ones(SHAPE), z_t_all_ones["mod_a"], t_half
        )


# ══════════════════════════════════════════════════════════════════════
# Intrinsic guidance on a DISCRETE modality (cross-entropy on native logits)
# ══════════════════════════════════════════════════════════════════════


class _UnusedNetwork(Network):
    """Network stand-in: get_guidance_loss consumes net_out directly."""

    def __call__(
        self, z_t, t, context_data=None, context_mask=None, attention_mask=None
    ):
        return z_t


class DiscreteLogitGenerativeModel(GenerativeModel):
    """CDCD-style direct-logit head: ``net_out`` IS the native categorical logits.

    ``get_guidance_loss`` scores the discrete head's ``net_out`` straight against
    the condition with a cross-entropy, and recovers the continuous head's clean
    target the usual way (``velocity -> target -> de-embed``) for an MSE. Sampling
    conversions are unused here.
    """

    def get_loss(self, net_out, t, z_t, epsilon, embedded_pairs, raw_pairs, loss_mask):
        raise NotImplementedError

    def get_generator(self, net_out, z_t, t):
        raise NotImplementedError

    def get_guidance_loss(self, net_out, z_t, t, intrinsic_data, intrinsic_mask):
        filled_intrinsic_mask = fill_intrinsic_mask(intrinsic_data, intrinsic_mask)

        def _loss_per_modality(
            modality, net_out_k, z_t_k, intrinsic_data_k, intrinsic_mask_k
        ):
            if modality.is_discrete:
                return CrossEntropyCriterion()(
                    net_out_k, intrinsic_data_k, t, intrinsic_mask_k
                )
            z1 = modality.interpolant.target_from_velocity(net_out_k, z_t_k, t)
            return MSECriterion()(
                modality.embedder.from_embeddings_to_raw(z1),
                intrinsic_data_k,
                t,
                intrinsic_mask_k,
            )

        losses = self.modality_registry.map(
            _loss_per_modality,
            net_out,
            z_t,
            intrinsic_data,
            filled_intrinsic_mask,
        )
        return jax.tree.reduce(jnp.add, losses, jnp.zeros(()))


class EmbedderLogitsVelocityGenerativeModel(VelocityOneSidedGenerativeModel):
    """Velocity head over both modality kinds, with discrete logits from the embedder."""

    def get_guidance_loss(self, net_out, z_t, t, intrinsic_data, intrinsic_mask):
        filled_intrinsic_mask = fill_intrinsic_mask(intrinsic_data, intrinsic_mask)

        def _loss_per_modality(
            modality, net_out_k, z_t_k, intrinsic_data_k, intrinsic_mask_k
        ):
            z1 = modality.interpolant.target_from_velocity(net_out_k, z_t_k, t)
            if modality.is_discrete:
                return CrossEntropyCriterion()(
                    modality.embedder.from_embeddings_to_logits(z1),
                    intrinsic_data_k,
                    t,
                    intrinsic_mask_k,
                )
            return MSECriterion()(
                modality.embedder.from_embeddings_to_raw(z1),
                intrinsic_data_k,
                t,
                intrinsic_mask_k,
            )

        losses = self.modality_registry.map(
            _loss_per_modality,
            net_out,
            z_t,
            intrinsic_data,
            filled_intrinsic_mask,
        )
        return jax.tree.reduce(jnp.add, losses, jnp.zeros(()))


def _discrete_registry(embedder) -> ModalityRegistry:
    """Single discrete modality ``tok`` with the given embedder."""
    return ModalityRegistry(
        {
            "tok": Modality(
                shape=embedder.dm_shape,
                is_discrete=True,
                interpolant=one_sided_si(),
                embedder=embedder,
                num_categories=embedder.dm_shape[-1],
            )
        }
    )


def test_intrinsic_guidance_mixed_registry():
    """A registry with BOTH a continuous and a discrete modality: the loss sums MSE
    (continuous, on the de-embedded z₁) and cross-entropy (discrete, on native
    logits).
    """
    registry = ModalityRegistry(
        {
            "cts": Modality(
                shape=(2,),
                is_discrete=False,
                interpolant=one_sided_si(),
                embedder=IdentityEmbedder(dm_shape=(2,)),
            ),
            "tok": Modality(
                shape=(4,),
                is_discrete=True,
                interpolant=one_sided_si(),
                embedder=OneHotDiscreteEmbedder(dm_shape=(4,)),
                num_categories=4,
            ),
        }
    )
    model = DiscreteLogitGenerativeModel(
        network=_UnusedNetwork(), modality_registry=registry
    )
    net_out = {
        "cts": jnp.array([[0.3, -0.2]]),
        "tok": jnp.array([[2.0, 0.5, -1.0, 0.0]]),
    }
    z_t = {"cts": jnp.zeros((1, 2)), "tok": jnp.zeros((1, 4))}
    condition = {
        "cts": jnp.array([[0.1, -0.1]]),
        "tok": jnp.array([[1.0, 0.0, 0.0, 0.0]]),
    }
    t = jnp.array(0.5)

    loss = model.get_guidance_loss(
        net_out, z_t, t, condition, {"cts": None, "tok": None}
    )

    z1_cts = registry.registry["cts"].interpolant.target_from_velocity(
        net_out["cts"], z_t["cts"], t
    )
    mse = jnp.sum((z1_cts - condition["cts"]) ** 2) / 2.0
    logits = net_out["tok"][0]
    cross_entropy = jnp.log(jnp.sum(jnp.exp(logits))) - logits[0]
    assert jnp.allclose(loss, mse + cross_entropy)


def test_intrinsic_guidance_mixed_registry_embedder_logits():
    """A velocity head on a mixed registry, where the discrete logits are recovered
    from the *embedder* rather than read off ``net_out``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tok_embedder = LearnedDiscreteEmbedder(
            dm_shape=(4,), embedding_dim=5, propagate_grads=False, rngs=nnx.Rngs(0)
        )
    registry = ModalityRegistry(
        {
            "cts": Modality(
                shape=(2,),
                is_discrete=False,
                interpolant=one_sided_si(),
                embedder=IdentityEmbedder(dm_shape=(2,)),
            ),
            "tok": Modality(
                shape=(4,),
                is_discrete=True,
                interpolant=one_sided_si(),
                embedder=tok_embedder,
                num_categories=4,
            ),
        }
    )
    model = EmbedderLogitsVelocityGenerativeModel(
        network=_UnusedNetwork(), modality_registry=registry
    )
    net_out = {
        "cts": jnp.array([[0.3, -0.2]]),
        "tok": jnp.array([[0.5, -0.1, 0.2, 0.0, 0.4]]),
    }
    z_t = {"cts": jnp.zeros((1, 2)), "tok": jnp.zeros((1, 5))}
    condition = {
        "cts": jnp.array([[0.1, -0.1]]),
        "tok": jnp.array([[1.0, 0.0, 0.0, 0.0]]),
    }
    t = jnp.array(0.5)

    loss = model.get_guidance_loss(
        net_out, z_t, t, condition, {"cts": None, "tok": None}
    )

    z1_cts = registry.registry["cts"].interpolant.target_from_velocity(
        net_out["cts"], z_t["cts"], t
    )
    mse = jnp.sum((z1_cts - condition["cts"]) ** 2) / 2.0
    z1_tok = registry.registry["tok"].interpolant.target_from_velocity(
        net_out["tok"], z_t["tok"], t
    )
    logits = tok_embedder.from_embeddings_to_logits(z1_tok)[0]
    cross_entropy = jnp.log(jnp.sum(jnp.exp(logits))) - logits[0]
    assert jnp.allclose(loss, mse + cross_entropy)


class _LinearLogitNetwork(Network):
    """Tiny param network mapping a categorical modality's z_t to native logits."""

    def __init__(self, in_dim: int, num_categories: int, rngs: nnx.Rngs):
        self.readout = nnx.Linear(in_dim, num_categories, rngs=rngs)

    def __call__(
        self, z_t, t, context_data=None, context_mask=None, attention_mask=None
    ):
        return {"tok": self.readout(z_t["tok"])}


def test_intrinsic_guidance_discrete_gradient_flows_through_network():
    """The discrete guidance loss must be differentiable w.r.t. ``z_t`` THROUGH
    the network (no accidental ``stop_gradient`` on the logits path).
    """
    embedder = OneHotDiscreteEmbedder(dm_shape=(4,))
    model = DiscreteLogitGenerativeModel(
        network=_LinearLogitNetwork(in_dim=4, num_categories=4, rngs=nnx.Rngs(0)),
        modality_registry=_discrete_registry(embedder),
    )
    condition = {"tok": jnp.array([[1.0, 0.0, 0.0, 0.0]])}
    t = jnp.array(0.5)

    def loss_of_z_t(z_t):
        net_out = model.get_network_output(z_t, t, None, None, None)
        return model.get_guidance_loss(net_out, z_t, t, condition, {"tok": None})

    grad = jax.grad(loss_of_z_t)({"tok": jnp.ones((1, 4))})
    assert jnp.all(jnp.isfinite(grad["tok"]))
    assert jnp.any(grad["tok"] != 0.0)


def test_require_score_rejects_velocity_only_generator(
    gen_model_uncond_1_cond_2, z_t_all_ones, t_half
):
    """Score-based guidance on a ``Velocity``-only generator fails loudly.

    CFG combines scores, so a model whose ``get_generator`` returns plain
    ``Velocity`` (no score) must raise a clear ``TypeError``.
    """

    class _VelocityOnlyModel(type(gen_model_uncond_1_cond_2)):
        def get_generator(self, net_out, z_t, t):
            return self.modality_registry.map(
                lambda modality, net_out_k: Velocity(velocity=net_out_k),
                net_out,
            )

    velocity_only = _VelocityOnlyModel(
        network=gen_model_uncond_1_cond_2.network,
        modality_registry=gen_model_uncond_1_cond_2.modality_registry,
    )

    with pytest.raises(TypeError, match="VelocityAndScore"):
        get_classifier_free_guidance_generator(
            velocity_only,
            z_t_all_ones,
            t_half,
            context_data={"context": jnp.zeros(SHAPE)},
            context_mask={"context": jnp.ones(SHAPE)},
            attention_mask=None,
            guidance_scale={"mod_a": lambda t: jnp.array(1.0)},
        )
