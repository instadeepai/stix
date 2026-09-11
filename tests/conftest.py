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

"""Shared test fixtures for the stix test suite.

Defines toy networks, interpolant factories, and model builders used across
multiple test directories.  Import in test files with
``from conftest import ...``.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flax import nnx
from jaxtyping import PyTree

from stix.core.embedder import Embedder, IdentityEmbedder, OneHotDiscreteEmbedder
from stix.core.gen_model.factory import (
    NoiseOneSidedGenerativeModel,
    PosteriorMixtureGenerativeModel,
    VelocityAndPosteriorGenerativeModel,
    VelocityOneSidedGenerativeModel,
)
from stix.core.interpolant.linear_interpolant import (
    LinearDeterministicInterpolant,
    LinearStochasticInterpolant,
)
from stix.core.interpolant.standard_interpolants.discrete_diffusion import (
    MaskDiscreteInterpolant,
    UniformDiscreteInterpolant,
)
from stix.core.interpolant.standard_interpolants.flow_matching import (
    FlowMatchingOneSidedInterpolant,
)
from stix.core.loss import MSECriterion
from stix.core.modality import Modality, ModalityRegistry
from stix.nn import Network
from stix.typing import Mask, RawSourceTargetPair, Shape, Time, Var
from stix.typing.data import Batch

# ── Constants ──

SHAPE = (3,)
BATCH_SIZE = 2
SEQ_LEN = 8
DEFAULT_SEED = 42
DEFAULT_T = 0.3
TOL = 1e-4

# Discrete (mixture-path) toy constants.
DISCRETE_SHAPE = (4,)  # a length-4 sequence of categorical indices
NUM_CATEGORIES = 3


@pytest.fixture
def key():
    return jr.PRNGKey(0)


# ── Toy networks ──


class IdentityNetwork(Network):
    """Minimal network returning input unchanged (no trainable params)."""

    def __call__(
        self, z_t, t, context_data=None, context_mask=None, attention_mask=None
    ):
        return z_t


class ToyConstantNetwork(Network):
    """Network that always returns a constant (broadcast to match input shapes).

    Has a single trainable nnx.Param so that nnx.split/merge works correctly.
    """

    def __init__(self, value: float):
        self.bias = nnx.Param(jnp.array(value))

    def __call__(
        self, z_t, t, context_data=None, context_mask=None, attention_mask=None
    ):
        return {k: jnp.broadcast_to(self.bias[...], v.shape) for k, v in z_t.items()}


class ToyParamNetwork(Network):
    """Network with a single trainable parameter that scales its input."""

    def __init__(self):
        self.scale = nnx.Param(jnp.array(1.0))

    def __call__(
        self, z_t, t, context_data=None, context_mask=None, attention_mask=None
    ):
        return {k: v * self.scale[...] for k, v in z_t.items()}


class ToyLogitsNetwork(Network):
    """Network emitting constant per-category logits, ignoring the input state.

    Each modality's state is a one-hot ``z_t`` of shape ``(*positions,
    num_states)``; the network broadcasts a learnable ``(num_categories,)`` bias
    to ``(*positions, num_categories)`` — i.e. per-position posterior logits over
    the data categories (the one-hot state's trailing axis is dropped). Used by
    the discrete (mixture-path) model tests: setting the bias to favour one
    category makes the terminal denoising posterior (and hence the resolved
    sample) deterministic.
    """

    def __init__(self, logits):
        self.bias = nnx.Param(jnp.asarray(logits, dtype=jnp.float32))

    def __call__(
        self, z_t, t, context_data=None, context_mask=None, attention_mask=None
    ):
        del t, context_data, context_mask, attention_mask
        return {
            k: jnp.broadcast_to(self.bias[...], (*v.shape[:-1], self.bias.shape[-1]))
            for k, v in z_t.items()
        }


class ToyMultimodalNetwork(Network):
    """Network emitting a constant velocity for continuous modalities and constant
    logits for the (one-hot) discrete modalities.

    Both continuous and discrete states are float arrays now (the discrete state
    is a one-hot ``z_t``), so dispatch is by key: a modality in ``discrete_keys``
    is a discrete (mixture-path) modality and gets ``(*positions, num_categories)``
    posterior logits favouring one category (dropping the one-hot trailing axis);
    every other modality is continuous and gets a constant velocity broadcast to
    its shape.
    """

    def __init__(
        self,
        velocity_value: float,
        favoured_category: int,
        num_categories: int,
        discrete_keys: tuple[str, ...] = ("tokens",),
    ):
        self.velocity = nnx.Param(jnp.array(velocity_value, dtype=jnp.float32))
        logits = [0.0] * num_categories
        logits[favoured_category] = 10.0
        self.logits = nnx.Param(jnp.asarray(logits, dtype=jnp.float32))
        self.discrete_keys = discrete_keys

    def __call__(
        self, z_t, t, context_data=None, context_mask=None, attention_mask=None
    ):
        del t, context_data, context_mask, attention_mask
        out = {}
        for k, v in z_t.items():
            if k in self.discrete_keys:
                out[k] = jnp.broadcast_to(
                    self.logits[...], (*v.shape[:-1], self.logits.shape[-1])
                )
            else:
                out[k] = jnp.broadcast_to(self.velocity[...], v.shape)
        return out


@pytest.fixture
def identity_network():
    return IdentityNetwork()


@pytest.fixture
def constant_network():
    return ToyConstantNetwork(value=5.0)


# ── Toy embedders ──


class ScaleEmbedder(Embedder):
    """Continuous embedder carrying a single learnable ``nnx.Param`` (a scale).

    Holds one ``nnx.Param`` of shape ``dm_shape`` and multiplies the raw variable
    by it on encode, so the parameter participates in the embedded pair (and hence
    the loss). Used to probe NNX tracing: the param must show up in
    ``nnx.split(..., nnx.Param)`` state — including when the embedder lives inside a
    registry inside a model — and must receive a real gradient and move under an
    optimizer step.
    """

    def __init__(self, dm_shape: Shape):
        super().__init__(dm_shape=dm_shape, embedding_shape=dm_shape)
        self.scale = nnx.Param(jnp.ones(dm_shape))

    def from_raw_to_embeddings(self, raw_var):
        return raw_var * self.scale[...]

    def from_embeddings_to_raw(self, embedded_var):
        return embedded_var / self.scale[...]


# ── Toy NN slot modules (encoder/backbone/decoder/context-encoder) ──
#
# Used by tests that wire up a real ``EncoderBackboneDecoderNetwork`` +
# ``SumContextEncoder`` and
# want to observe how t / context_data / context_mask thread through.
# All take an optional ``rngs`` (ignored) so they slot into call sites that
# follow the ``Module(rngs=rngs)`` convention.


class IdentityEncoder(nnx.Module):
    """Modality encoder returning ``x`` unchanged."""

    def __init__(self, *, rngs: nnx.Rngs | None = None) -> None:
        del rngs

    def __call__(self, x, t):
        del t
        return x


class IdentityDecoder(nnx.Module):
    """Modality decoder returning its input unchanged."""

    def __init__(self, *, rngs: nnx.Rngs | None = None) -> None:
        del rngs

    def __call__(self, x, context=None):
        del context
        return x


class AddContextBackbone(nnx.Module):
    """Backbone that adds context to each modality's encoded input.

    Pass-through when ``context is None`` so the same backbone serves both
    ``context_encoder=None`` and conditioned wiring.
    """

    def __init__(self, *, rngs: nnx.Rngs | None = None) -> None:
        del rngs

    def __call__(self, encoded, t, context=None, attention_mask=None):
        del t, attention_mask
        if context is None:
            return encoded
        return {k: v + context for k, v in encoded.items()}


class ConstantContextEncoder(nnx.Module):
    """SumContextEncoder slot returning ``jnp.full(shape, value)``, ignoring input.

    Serves as either a constant time encoder or a constant condition encoder
    — the SumContextEncoder calls both with a single positional arg.
    """

    def __init__(
        self,
        value: float,
        shape: tuple[int, ...],
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        del rngs
        self.value = nnx.Param(jnp.full(shape, value, dtype=jnp.float32))

    def __call__(self, _arg):
        return self.value[...]


class BroadcastTimeEncoder(nnx.Module):
    """Time encoder broadcasting scalar ``t`` to ``shape``."""

    def __init__(self, shape: tuple[int, ...], *, rngs: nnx.Rngs | None = None) -> None:
        del rngs
        self.shape = shape

    def __call__(self, t):
        return jnp.broadcast_to(jnp.asarray(t, jnp.float32), self.shape)


class EmbedConditionEncoder(nnx.Module):
    """Categorical condition encoder mapping ``idx ∈ {0, 1}`` to per-class vectors."""

    def __init__(
        self,
        values: tuple[float, float],
        shape: tuple[int, ...],
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        del rngs
        v0 = jnp.full(shape, values[0], dtype=jnp.float32)
        v1 = jnp.full(shape, values[1], dtype=jnp.float32)
        self.table = nnx.Param(jnp.stack([v0, v1], axis=0))

    def __call__(self, idx):
        return self.table[...][idx]


# ── Interpolant factories ──


def one_sided_si() -> FlowMatchingOneSidedInterpolant:
    """One-sided FM: gamma(t) = 1-t, beta(t) = t."""
    return FlowMatchingOneSidedInterpolant()


def two_sided_si() -> LinearStochasticInterpolant:
    """Two-sided FM: gamma(t) = sqrt(t(1-t)), alpha(t) = 1-t, beta(t) = t."""
    return LinearStochasticInterpolant(
        gamma_fn=lambda t: (t * (1 - t)) ** 0.5,
        alpha_fn=lambda t: 1 - t,
        beta_fn=lambda t: t,
    )


def two_sided_deterministic_si() -> LinearDeterministicInterpolant:
    """Deterministic FM: alpha(t) = 1-t, beta(t) = t."""
    return LinearDeterministicInterpolant(alpha_fn=lambda t: 1 - t, beta_fn=lambda t: t)


def mask_mixture_interpolant(
    *, num_categories: int = NUM_CATEGORIES
) -> MaskDiscreteInterpolant:
    """Masked-diffusion path with a linear ``kappa`` schedule."""
    return MaskDiscreteInterpolant(num_categories=num_categories, kappa_fn=lambda t: t)


def uniform_mixture_interpolant(
    *, num_categories: int = NUM_CATEGORIES
) -> UniformDiscreteInterpolant:
    """Uniform-diffusion path with a linear ``kappa`` schedule."""
    return UniformDiscreteInterpolant(
        num_categories=num_categories, kappa_fn=lambda t: t
    )


# ── Shared gen-model builders ──


def build_one_sided_registry(*, shape: Shape = SHAPE) -> ModalityRegistry:
    """Single-modality (``mod_a``) one-sided modality_registry used by the toy gen-models.

    An ``IdentityEmbedder`` keeps embedded space == raw space. The training loss
    now lives on the paired ``GenerativeModel.get_loss``, not the registry.
    """
    return ModalityRegistry(
        {
            "mod_a": Modality(
                shape=shape,
                is_discrete=False,
                interpolant=one_sided_si(),
                embedder=IdentityEmbedder(dm_shape=shape),
            )
        }
    )


def build_learnable_embedder_registry(*, shape: Shape = SHAPE) -> ModalityRegistry:
    """Single continuous modality whose embedder carries a learnable ``nnx.Param``.

    Mirrors :func:`build_one_sided_registry` but swaps the ``IdentityEmbedder`` for a
    :class:`ScaleEmbedder`, so the registry holds a trainable parameter. Used to
    assert a registry-held param is captured by ``nnx.split``, receives a gradient
    through the real loss, and moves under an optimizer step. Paired with the
    velocity-MSE :class:`VelocityOneSidedGenerativeModel` so the embedder participates in
    the loss.
    """
    return ModalityRegistry(
        {
            "mod_a": Modality(
                shape=shape,
                is_discrete=False,
                interpolant=one_sided_si(),
                embedder=ScaleEmbedder(dm_shape=shape),
            )
        }
    )


def toy_learnable_embedder_gen_model() -> VelocityOneSidedGenerativeModel:
    """Velocity model whose *only* trainable param is a registry-held embedder scale.

    ``IdentityNetwork`` contributes no parameters, so ``nnx.split(model, nnx.Param)``
    captures exactly the registry-held :class:`ScaleEmbedder` scale — making the
    trainability guards unambiguous.
    """
    return VelocityOneSidedGenerativeModel(
        network=IdentityNetwork(),
        modality_registry=build_learnable_embedder_registry(),
    )


def toy_constant_velocity_gen_model(value: float) -> VelocityOneSidedGenerativeModel:
    """One-sided velocity-prediction model whose network emits a constant velocity.

    ``ToyConstantNetwork`` + ``IdentityEmbedder`` so the network output is used
    verbatim as the velocity field. Used by solver and guidance tests.
    """
    return VelocityOneSidedGenerativeModel(
        network=ToyConstantNetwork(value),
        modality_registry=build_one_sided_registry(),
    )


# ── Intrinsic-guidance-capable gen-models ──
#
# ``get_guidance_loss`` is an opt-in hook: the base ``GenerativeModel`` raises
# ``NotImplementedError`` and the library ships no implementation, so every model
# driving the intrinsic-guidance recipe defines its own loss. The model below
# provides one so guidance and solver tests can exercise that recipe.


def fill_intrinsic_mask(
    intrinsic_data: PyTree[Var], intrinsic_mask: PyTree[Mask | None] | None
) -> PyTree[Mask]:
    """Resolve the ``None``-means-fully-conditional mask convention to all-ones masks.

    A whole-tree ``None`` and a per-entry ``None`` both mean "no masking". The
    criteria require real arrays, so both are materialised as ``ones_like`` the
    target. ``intrinsic_data`` must lead the ``jax.tree.map`` so that a ``None``
    mask entry is handed to the function rather than treated as an empty subtree.

    Each model implementing ``get_guidance_loss`` owns this convention, so the toy
    guidance models share this helper.
    """
    if intrinsic_mask is None:
        intrinsic_mask = jax.tree.map(lambda _: None, intrinsic_data)
    return jax.tree.map(
        lambda intrinsic_data_k, intrinsic_mask_k: (
            jnp.ones_like(intrinsic_data_k)
            if intrinsic_mask_k is None
            else intrinsic_mask_k
        ),
        intrinsic_data,
        intrinsic_mask,
    )


class GuidedVelocityOneSidedGenerativeModel(VelocityOneSidedGenerativeModel):
    """Velocity model that opts into intrinsic guidance via ``get_guidance_loss``.

    The canonical shape of the hook for a velocity head over continuous
    modalities: recover the clean target (``velocity -> target -> de-embed``) and
    score it against the condition with an MSE.
    """

    def get_guidance_loss(self, net_out, z_t, t, intrinsic_data, intrinsic_mask):
        filled_intrinsic_mask = fill_intrinsic_mask(intrinsic_data, intrinsic_mask)

        def _loss_per_modality(
            modality, net_out_k, z_t_k, intrinsic_data_k, intrinsic_mask_k
        ):
            z1 = modality.interpolant.target_from_velocity(net_out_k, z_t_k, t)
            raw_prediction = modality.embedder.from_embeddings_to_raw(z1)
            return MSECriterion()(raw_prediction, intrinsic_data_k, t, intrinsic_mask_k)

        losses = self.modality_registry.map(
            _loss_per_modality,
            net_out,
            z_t,
            intrinsic_data,
            filled_intrinsic_mask,
        )
        return jax.tree.reduce(jnp.add, losses, jnp.zeros(()))


def toy_guided_constant_velocity_gen_model(
    value: float,
) -> GuidedVelocityOneSidedGenerativeModel:
    """:func:`toy_constant_velocity_gen_model`, plus an intrinsic-guidance loss.

    Used by the tests that drive ``get_intrinsic_guidance_score``, which needs a
    model overriding ``get_guidance_loss``.
    """
    return GuidedVelocityOneSidedGenerativeModel(
        network=ToyConstantNetwork(value),
        modality_registry=build_one_sided_registry(),
    )


def toy_one_sided_gen_model(
    network: Network | None = None,
) -> NoiseOneSidedGenerativeModel:
    """One-sided noise-prediction model over a single modality.

    Defaults to a :class:`ToyParamNetwork`. Used as the ``gen_model`` argument
    to the training loop in training/checkpointer tests.
    """
    return NoiseOneSidedGenerativeModel(
        network=network if network is not None else ToyParamNetwork(),
        modality_registry=build_one_sided_registry(),
    )


def build_mixture_registry(
    *,
    source: str = "mask",
    shape: Shape = DISCRETE_SHAPE,
    num_categories: int = NUM_CATEGORIES,
) -> ModalityRegistry:
    """Single discrete modality on the mixture path (masked or uniform).

    Index-valued raw space with a ``OneHotDiscreteEmbedder`` (the embedded/sampling
    state ``z_t`` is a one-hot vector over the source's ``num_states`` states),
    CTMC sampling via the interpolant's inferred ``TransitionRates`` generator type.
    """
    interpolant = (
        mask_mixture_interpolant(num_categories=num_categories)
        if source == "mask"
        else uniform_mixture_interpolant(num_categories=num_categories)
    )
    return ModalityRegistry(
        {
            "tokens": Modality(
                shape=shape,
                is_discrete=True,
                interpolant=interpolant,
                embedder=OneHotDiscreteEmbedder(
                    dm_shape=shape, num_states=interpolant.num_states
                ),
                num_categories=num_categories,
            )
        }
    )


def toy_posterior_mixture_gen_model(
    *,
    source: str = "mask",
    favoured_category: int = 0,
    num_categories: int = NUM_CATEGORIES,
) -> PosteriorMixtureGenerativeModel:
    """Posterior mixture model whose network always favours ``favoured_category``.

    The constant logits make the terminal denoising posterior a near-point-mass on
    ``favoured_category``, so masked-diffusion sampling should resolve every
    position to that category (a deterministic target for the sampling tests).
    """
    logits = [0.0] * num_categories
    logits[favoured_category] = 10.0
    return PosteriorMixtureGenerativeModel(
        network=ToyLogitsNetwork(logits),
        modality_registry=build_mixture_registry(
            source=source, num_categories=num_categories
        ),
    )


def build_multimodal_registry(
    *,
    continuous_shape: Shape = SHAPE,
    discrete_shape: Shape = DISCRETE_SHAPE,
    num_categories: int = NUM_CATEGORIES,
) -> ModalityRegistry:
    """A continuous one-sided modality (``cont``) plus a masked discrete one (``tokens``)."""
    token_interpolant = mask_mixture_interpolant(num_categories=num_categories)
    return ModalityRegistry(
        {
            "cont": Modality(
                shape=continuous_shape,
                is_discrete=False,
                interpolant=one_sided_si(),
                embedder=IdentityEmbedder(dm_shape=continuous_shape),
            ),
            "tokens": Modality(
                shape=discrete_shape,
                is_discrete=True,
                interpolant=token_interpolant,
                embedder=OneHotDiscreteEmbedder(
                    dm_shape=discrete_shape,
                    num_states=token_interpolant.num_states,
                ),
                num_categories=num_categories,
            ),
        }
    )


def toy_multimodal_gen_model(
    *,
    velocity_value: float = 1.0,
    favoured_category: int = 0,
    num_categories: int = NUM_CATEGORIES,
) -> VelocityAndPosteriorGenerativeModel:
    """Joint continuous (velocity) + discrete DFM toy model."""
    return VelocityAndPosteriorGenerativeModel(
        network=ToyMultimodalNetwork(
            velocity_value=velocity_value,
            favoured_category=favoured_category,
            num_categories=num_categories,
        ),
        modality_registry=build_multimodal_registry(num_categories=num_categories),
    )


def toy_batch() -> Batch:
    """Single training batch of ones with the default ``BATCH_SIZE``."""
    return Batch(
        raw_batch={
            "mod_a": RawSourceTargetPair(
                target=jnp.ones((BATCH_SIZE, *SHAPE)), source=None
            )
        },
        context_data=None,
        context_mask=None,
        attn_mask=None,
        loss_mask=None,
    )


# ── Shared test inputs ──


def random_inputs_one_sided(
    shape: tuple[int, ...] = SHAPE,
    seed: int = DEFAULT_SEED,
    t: float = DEFAULT_T,
) -> tuple[Var, Var, Time]:
    """Deterministic (var, z_t, t) triplet for conversion-style tests."""
    k1, k2 = jr.split(jr.PRNGKey(seed))
    var = jr.normal(k1, shape)
    z_t = jr.normal(k2, shape)
    return var, z_t, jnp.array(t)
