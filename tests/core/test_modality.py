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

"""Tests for ``stix.core.modality``: the ``Modality`` config and ``ModalityRegistry``.

Covers construction defaults, the NNX-tracing guarantee (params of a stored
``embedder``/``interpolant`` are visible to ``nnx.split`` — including when the
attribute is assigned *after* a ``None`` init, which is the ``set``
mutation path), registry leaf-extraction, structure inference from a batch, and
the bulk in-place ``set`` mutator (broadcasting and per-modality pytrees).
"""

import inspect

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import BATCH_SIZE, SHAPE, ScaleEmbedder, two_sided_si
from flax import nnx

from stix.core.embedder import (
    IdentityEmbedder,
    OneHotDiscreteEmbedder,
)
from stix.core.interpolant.interpolant import Interpolant
from stix.core.modality import Modality, ModalityRegistry, _is_field_value
from stix.typing.data import Batch
from stix.typing.variables import RawSourceTargetPair


def _batch(raw_batch, is_discrete=None) -> Batch:
    """Wrap a ``raw_batch`` pytree in an otherwise-empty :class:`Batch`."""
    return Batch(
        raw_batch=raw_batch,
        context_data=None,
        context_mask=None,
        attn_mask=None,
        loss_mask=None,
        is_discrete=is_discrete,
    )


# ── Modality ──


def test_embedder_params_visible_after_inplace_replace():
    """Assigning an embedder *after* a ``None`` init is still traced by NNX.

    This is the ``set`` mutation path, and the reason ``embedder``
    is wrapped in ``nnx.data`` in ``Modality.__init__``.
    """
    modality = Modality(shape=SHAPE)
    assert modality.embedder is None
    modality.embedder = ScaleEmbedder(SHAPE)
    _, state = nnx.split(modality)
    leaves = jax.tree.leaves(state)
    assert any(jnp.array_equal(leaf, jnp.ones(SHAPE)) for leaf in leaves)


def test_embedder_params_visible_through_registry_in_module():
    """A registry-held embedder's params are visible to ``nnx.split`` on the owner.

    ``ModalityRegistry`` is an ``nnx.Module`` and wraps its ``registry`` in
    ``nnx.data``, so NNX descends into the registry rather than treating it as an
    opaque static leaf. This is what makes a learnable embedder living *inside the
    registry inside a model* trainable: its param must appear in the
    ``nnx.Param`` state of the enclosing module (mirroring how ``GenerativeModel``
    holds ``self.modality_registry``), otherwise it is silently frozen at init.
    """

    class _Wrapper(nnx.Module):
        """Minimal owner holding a registry, as ``GenerativeModel`` does."""

        def __init__(self, modality_registry):
            self.modality_registry = modality_registry

    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE, embedder=ScaleEmbedder(SHAPE))}
    )
    wrapper = _Wrapper(modality_registry)
    _, state = nnx.split(wrapper, nnx.Param)
    leaves = jax.tree.leaves(state)
    assert any(jnp.array_equal(leaf, jnp.ones(SHAPE)) for leaf in leaves)


def test_fully_built_modality_splits_cleanly_on_param():
    """A bare ``Modality(interpolant, embedder)`` splits on ``nnx.Param`` cleanly.

    Invariant: ``interpolant`` is a plain *static* attribute, so a fully-built
    ``Modality`` has no non-``Param`` state leaf — ``nnx.split(modality, nnx.Param)``
    succeeds and the interpolant never appears as a state leaf (it lives in the
    graphdef). Prevents the latent defect of wrapping the param-less interpolant in
    ``nnx.data``, which turns it into a non-array data leaf that makes
    ``nnx.split(..., nnx.Param)`` raise a "non-empty remainder" error and breaks JIT
    tracing of any model holding the modality.
    """
    modality = Modality(
        interpolant=two_sided_si(),
        embedder=IdentityEmbedder(dm_shape=SHAPE),
    )
    # Must not raise "Non-exhaustive filters, got a non-empty remainder".
    nnx.split(modality, nnx.Param)
    # The interpolant must be static (in the graphdef), never a state data leaf.
    _, full_state = nnx.split(modality)
    state_leaves = jax.tree.leaves(
        full_state, is_leaf=lambda x: isinstance(x, Interpolant)
    )
    assert not any(isinstance(leaf, Interpolant) for leaf in state_leaves)


# ── ModalityRegistry._modality_leaves ──


def test_modality_leaves_flat():
    """Leaves of a flat registry are the ``Modality`` objects themselves."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    leaves = modality_registry._modality_leaves()
    assert len(leaves) == 2
    assert all(isinstance(modality, Modality) for modality in leaves)


def test_modality_leaves_nested_stops_at_modality():
    """Traversal stops at ``Modality`` and does not descend into its NNX params.

    Even with a nested registry and a param-carrying embedder, the leaves are
    ``Modality`` objects (not parameter arrays).
    """
    modality_registry = ModalityRegistry(
        {
            "group": {"a": Modality(shape=SHAPE, embedder=ScaleEmbedder(SHAPE))},
            "b": Modality(shape=(5,)),
        }
    )
    leaves = modality_registry._modality_leaves()
    assert len(leaves) == 2
    assert all(isinstance(modality, Modality) for modality in leaves)


# ── ModalityRegistry.assert_compatible / map ──


def test_assert_compatible_accepts_matching_skeleton():
    """A tree sharing the registry's modality skeleton is compatible."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    # Same keys, plain array leaves -> registry treedef is an exact prefix.
    modality_registry.assert_compatible({"a": jnp.zeros(SHAPE), "b": jnp.zeros((5,))})


def test_assert_compatible_accepts_higher_resolution_tree():
    """A tree that resolves *finer* below each modality is still compatible.

    This is the prefix property: the registry (``Modality`` leaves) is a prefix
    of a batch whose leaves are ``RawSourceTargetPair`` nodes, even with an
    asymmetric ``None`` source.
    """
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    modality_registry.assert_compatible(
        {
            "a": RawSourceTargetPair(target=jnp.zeros(SHAPE), source=None),
            "b": RawSourceTargetPair(target=jnp.zeros((5,)), source=jnp.zeros((5,))),
        }
    )


def test_assert_compatible_raises_with_readable_error_on_key_mismatch():
    """A missing modality key raises a registry-specific ``ValueError``."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    with pytest.raises(ValueError, match="not compatible with the modality registry"):
        modality_registry.assert_compatible({"a": jnp.zeros(SHAPE)})  # missing "b"


def test_map_passes_modality_and_flattens_higher_resolution_leaf():
    """``map`` calls ``f(modality, *leaves)`` with leaves flattened up to it.

    A ``RawSourceTargetPair`` under each modality is delivered to ``f`` as a
    single leaf (the whole pair), confirming the registry is the prefix.
    """
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    batch = {
        "a": RawSourceTargetPair(target=jnp.full(SHAPE, 2.0), source=None),
        "b": RawSourceTargetPair(target=jnp.full((5,), 3.0), source=None),
    }
    result = modality_registry.map(lambda modality, pair: pair.target * 2, batch)
    assert jnp.array_equal(result["a"], jnp.full(SHAPE, 4.0))
    assert jnp.array_equal(result["b"], jnp.full((5,), 6.0))


def test_map_raises_on_incompatible_tree():
    """``map`` front-runs the compatibility check before mapping."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    with pytest.raises(ValueError, match="not compatible with the modality registry"):
        modality_registry.map(lambda modality, leaf: leaf, {"a": jnp.zeros(SHAPE)})


# ── ModalityRegistry.broadcast ──


def test_broadcast_single_value_to_every_modality():
    """A single leaf broadcasts to every modality of a flat registry."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    result = modality_registry.broadcast(2.0)
    assert result == {"a": 2.0, "b": 2.0}


def test_broadcast_prefix_per_group_to_nested_modalities():
    """A coarser per-group prefix broadcasts down to each modality in the group."""
    modality_registry = ModalityRegistry(
        {
            "group": {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))},
            "c": Modality(shape=(3,)),
        }
    )
    result = modality_registry.broadcast({"group": 1.0, "c": 2.0})
    assert result == {"group": {"a": 1.0, "b": 1.0}, "c": 2.0}


def test_broadcast_leaf_is_shared_not_copied():
    """A broadcast non-array leaf is the *same* object at every modality."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    sentinel = object()
    result = modality_registry.broadcast(sentinel)
    assert result["a"] is sentinel and result["b"] is sentinel


# ── ModalityRegistry.split_and_project_key ──


def test_split_and_project_key_matches_registry_structure():
    """The key-tree mirrors the registry's (possibly nested) modality structure."""
    modality_registry = ModalityRegistry(
        {
            "group": {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))},
            "c": Modality(shape=(3,)),
        }
    )
    keys = modality_registry.split_and_project_key(jax.random.PRNGKey(0))
    assert set(keys) == {"group", "c"}
    assert set(keys["group"]) == {"a", "b"}


def test_split_and_project_key_gives_distinct_key_per_modality():
    """Each modality gets an independent key, not a shared one."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    keys = modality_registry.split_and_project_key(jax.random.PRNGKey(0))
    assert not jnp.array_equal(keys["a"], keys["b"])


def test_split_and_project_key_is_deterministic():
    """The same input key yields the same per-modality key-tree."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    first = modality_registry.split_and_project_key(jax.random.PRNGKey(0))
    second = modality_registry.split_and_project_key(jax.random.PRNGKey(0))
    assert jnp.array_equal(first["a"], second["a"])
    assert jnp.array_equal(first["b"], second["b"])


# ── ModalityRegistry.from_batch ──


def test_from_batch_preserves_structure():
    """The registry mirrors the multi-modality structure of the batch."""
    raw_batch = {
        "x": RawSourceTargetPair(
            target=jnp.full((BATCH_SIZE, *SHAPE), 0.5), source=None
        ),
        "y": RawSourceTargetPair(target=jnp.full((BATCH_SIZE, 5), 0.5), source=None),
    }
    modality_registry = ModalityRegistry.from_batch(
        _batch(raw_batch, is_discrete={"x": False, "y": True})
    )
    assert set(modality_registry.registry) == {"x", "y"}
    assert modality_registry.registry["x"].shape == SHAPE
    assert modality_registry.registry["y"].shape == (5,)
    assert modality_registry.registry["x"].is_discrete is False
    assert modality_registry.registry["y"].is_discrete is True


def test_from_batch_requires_is_discrete():
    """from_batch raises when the batch does not state discreteness."""
    raw_batch = {
        "x": RawSourceTargetPair(
            target=jnp.full((BATCH_SIZE, *SHAPE), 0.5), source=None
        ),
    }
    with pytest.raises(ValueError, match="is_discrete"):
        ModalityRegistry.from_batch(_batch(raw_batch))


# ── ModalityRegistry.set ──


def test_set_broadcasts_single_interpolant():
    """A single interpolant is broadcast to every modality leaf.

    Uses the default (``use_deepcopy=True``): each modality
    receives an independent copy of the right type, not a shared instance.
    """
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=SHAPE)}
    )
    interpolant = two_sided_si()
    modality_registry.set("interpolant", interpolant)  # default: use_deepcopy=True
    leaves = modality_registry._modality_leaves()
    assert all(isinstance(m.interpolant, type(interpolant)) for m in leaves)
    assert all(m.interpolant is not interpolant for m in leaves)  # independent copies
    assert leaves[0].interpolant is not leaves[1].interpolant


def test_set_assigns_per_modality_pytree():
    """A registry-shaped pytree assigns a distinct value to each modality."""
    modality_registry = ModalityRegistry(
        {
            "cts": Modality(shape=SHAPE, is_discrete=False),
            "disc": Modality(shape=(4,), is_discrete=True),
        }
    )
    modality_registry.set(
        "embedder",
        {
            "cts": IdentityEmbedder(dm_shape=SHAPE),
            "disc": OneHotDiscreteEmbedder(dm_shape=(4,)),
        },
    )
    assert isinstance(modality_registry.registry["cts"].embedder, IdentityEmbedder)
    assert isinstance(
        modality_registry.registry["disc"].embedder, OneHotDiscreteEmbedder
    )


def test_set_embedder_deepcopy_gives_independent_params():
    """Default deep-copy gives every modality an embedder with *independent* params.

    The point of deep-copying shape/param-bearing values: mutating one modality's
    embedder must not touch the others (or the source), so per-modality params can
    diverge under training. A param-carrying embedder makes the aliasing visible.
    """
    modality_registry = ModalityRegistry(
        {
            "a": Modality(shape=SHAPE),
            "b": Modality(shape=SHAPE),
            "c": Modality(shape=SHAPE),
        }
    )
    embedder = ScaleEmbedder(SHAPE)  # carries an nnx.Param of ones
    modality_registry.set("embedder", embedder)  # default: use_deepcopy=True

    copies = [
        leaf_embedder
        for modality in modality_registry._modality_leaves()
        if isinstance(leaf_embedder := modality.embedder, ScaleEmbedder)
    ]
    assert len(copies) == 3  # every leaf got a ScaleEmbedder
    assert all(copy is not embedder for copy in copies)  # distinct from the source
    assert copies[0] is not copies[1] and copies[1] is not copies[2]  # and each other

    # Mutate one copy; the others and the source stay put -> params are not aliased.
    copies[0].scale = nnx.Param(jnp.full(SHAPE, 5.0))
    assert jnp.array_equal(copies[1].scale[...], jnp.ones(SHAPE))
    assert jnp.array_equal(copies[2].scale[...], jnp.ones(SHAPE))
    assert jnp.array_equal(embedder.scale[...], jnp.ones(SHAPE))


def test_set_broadcasts_prefix_over_nested_registry():
    """A coarse prefix replicates each leaf across the registry subtree it covers."""
    interpolant = two_sided_si()
    modality_registry = ModalityRegistry(
        {
            "a": Modality(shape=SHAPE),
            "group": {"b": Modality(shape=SHAPE), "c": Modality(shape=SHAPE)},
        }
    )
    other = two_sided_si()
    # Prefix stops at the "group" node: its single interpolant fans out to b and c.
    # use_deepcopy=False so identity pins which source each modality received.
    modality_registry.set(
        "interpolant", {"a": interpolant, "group": other}, use_deepcopy=False
    )
    assert modality_registry.registry["a"].interpolant is interpolant
    assert modality_registry.registry["group"]["b"].interpolant is other
    assert modality_registry.registry["group"]["c"].interpolant is other


def test_set_raises_on_incompatible_pytree():
    """A pytree that is not a valid prefix of the registry raises."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=SHAPE)}
    )
    with pytest.raises(ValueError):
        modality_registry.set("interpolant", {"a": two_sided_si()})  # missing "b"


def test_set_filter_only_affects_matching_modalities():
    """``filter`` restricts a broadcast to modalities for which it returns True."""
    modality_registry = ModalityRegistry(
        {
            "cts": Modality(shape=SHAPE, is_discrete=False),
            "disc": Modality(shape=(4,), is_discrete=True),
        }
    )
    embedder = IdentityEmbedder(dm_shape=SHAPE)
    modality_registry.set(
        "embedder",
        embedder,
        filter_fn=lambda m: not m.is_discrete,
        use_deepcopy=False,
    )
    assert modality_registry.registry["cts"].embedder is embedder
    assert modality_registry.registry["disc"].embedder is None


def test_set_filter_applies_within_broadcast_prefix():
    """``filter`` is honoured even when a prefix leaf fans out over a subtree."""
    interpolant = two_sided_si()
    modality_registry = ModalityRegistry(
        {
            "a": Modality(shape=SHAPE, is_discrete=False),
            "group": {
                "b": Modality(shape=SHAPE, is_discrete=True),
                "c": Modality(shape=SHAPE, is_discrete=False),
            },
        }
    )
    modality_registry.set(
        "interpolant",
        interpolant,
        filter_fn=lambda m: not m.is_discrete,
        use_deepcopy=False,
    )
    assert modality_registry.registry["a"].interpolant is interpolant
    assert modality_registry.registry["group"]["b"].interpolant is None
    assert modality_registry.registry["group"]["c"].interpolant is interpolant


# ── ModalityRegistry.set: is_factory ──


def test_set_factory_broadcast_builds_per_modality_value():
    """A single factory broadcasts and builds a distinct value per modality.

    The factory reads each modality's ``shape`` to size the embedder, so the two
    modalities end up with independent, correctly-shaped embedders.
    """
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=(5,))}
    )
    modality_registry.set(
        "embedder", lambda m: IdentityEmbedder(dm_shape=m.shape), is_factory=True
    )
    embedder_a = modality_registry.registry["a"].embedder
    embedder_b = modality_registry.registry["b"].embedder
    assert isinstance(embedder_a, IdentityEmbedder)
    assert isinstance(embedder_b, IdentityEmbedder)
    assert embedder_a is not embedder_b
    assert embedder_a.dm_shape == SHAPE
    assert embedder_b.dm_shape == (5,)


def test_set_factory_per_modality_pytree_maps_each_callable():
    """A registry-shaped pytree of factories maps one factory per modality."""
    modality_registry = ModalityRegistry(
        {
            "cts": Modality(shape=SHAPE, is_discrete=False),
            "disc": Modality(shape=(4,), is_discrete=True),
        }
    )
    modality_registry.set(
        "embedder",
        {
            "cts": lambda m: IdentityEmbedder(dm_shape=m.shape),
            "disc": lambda m: OneHotDiscreteEmbedder(dm_shape=m.shape),
        },
        is_factory=True,
    )
    assert isinstance(modality_registry.registry["cts"].embedder, IdentityEmbedder)
    assert isinstance(
        modality_registry.registry["disc"].embedder, OneHotDiscreteEmbedder
    )


def test_set_factory_honours_filter_fn():
    """``filter_fn`` restricts a factory broadcast to matching modalities."""
    modality_registry = ModalityRegistry(
        {
            "cts": Modality(shape=SHAPE, is_discrete=False),
            "disc": Modality(shape=(4,), is_discrete=True),
        }
    )
    modality_registry.set(
        "embedder",
        lambda m: IdentityEmbedder(dm_shape=m.shape),
        is_factory=True,
        filter_fn=lambda m: not m.is_discrete,
    )
    assert isinstance(modality_registry.registry["cts"].embedder, IdentityEmbedder)
    assert modality_registry.registry["disc"].embedder is None


def test_set_factory_non_callable_leaf_raises():
    """``is_factory=True`` with a non-callable leaf raises ``TypeError``."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=SHAPE)}
    )
    with pytest.raises(TypeError, match="requires callable leaves"):
        # A bare scalar is a non-callable leaf: the factory contract is violated.
        modality_registry.set("interpolant", 1.0, is_factory=True)


def test_set_factory_builds_per_modality_value():
    """``is_factory=True`` invokes each leaf with its ``Modality`` and stores the
    *result* — here a shape-dependent embedder built per modality."""
    modality_registry = ModalityRegistry(
        {
            "a": Modality(shape=SHAPE, is_discrete=False),
            "b": Modality(shape=(5,), is_discrete=True),
        }
    )
    modality_registry.set(
        "embedder",
        lambda m: IdentityEmbedder(dm_shape=m.shape),
        is_factory=True,
    )
    assert isinstance(modality_registry.registry["a"].embedder, IdentityEmbedder)
    assert modality_registry.registry["a"].embedder.embedding_shape == SHAPE
    assert modality_registry.registry["b"].embedder.embedding_shape == (5,)


# ── ModalityRegistry.set: use_deepcopy ──


def test_set_use_deepcopy_gives_distinct_objects_per_modality():
    """``use_deepcopy=True`` broadcasts an independent copy to each modality."""
    modality_registry = ModalityRegistry(
        {"a": Modality(shape=SHAPE), "b": Modality(shape=SHAPE)}
    )
    embedder = IdentityEmbedder(dm_shape=SHAPE)
    modality_registry.set("embedder", embedder, use_deepcopy=True)
    embedder_a = modality_registry.registry["a"].embedder
    embedder_b = modality_registry.registry["b"].embedder
    assert embedder_a is not embedder
    assert embedder_b is not embedder
    assert embedder_a is not embedder_b
    assert isinstance(embedder_a, IdentityEmbedder)
    assert embedder_a.dm_shape == SHAPE


def test_is_field_value_handles_every_settable_modality_field():
    """Guard against a ``Modality`` field drifting out of sync with ``_is_field_value``.

    Every constructor field a user can broadcast via ``ModalityRegistry.set`` must be
    classifiable by ``_is_field_value``; otherwise ``set`` hits its ``case _`` guard and
    raises ``Unknown modality field`` at runtime. ``shape`` and ``is_discrete`` are inferred
    by ``from_batch`` (set directly, never broadcast), so they are intentionally excluded.

    Adding a new ``Modality`` field forces the author to either handle it in
    ``_is_field_value`` or add it to the excluded-metadata set below.
    """
    inferred_metadata_fields = {"shape", "is_discrete"}
    constructor_fields = set(inspect.signature(Modality.__init__).parameters) - {"self"}

    # The exclusion set itself cannot rot: it must name real constructor fields.
    assert inferred_metadata_fields <= constructor_fields

    sentinel = object()
    for field in constructor_fields - inferred_metadata_fields:
        # A handled field returns a bool; an unhandled one raises via ``case _``.
        assert isinstance(_is_field_value(field, sentinel), bool), field


def test_is_field_value_raises_on_unknown_field():
    """The ``case _`` guard is what makes the sync test above meaningful."""
    with pytest.raises(ValueError, match="Unknown modality field"):
        _is_field_value("definitely_not_a_field", object())


def test_embedded_source_prior_rejected_on_one_sided_interpolant():
    """``embedded_source_prior`` is only valid with a two-sided interpolant."""
    from stix.core.interpolant.standard_interpolants.flow_matching import (
        FlowMatchingOneSidedInterpolant,
    )

    with pytest.raises(
        ValueError,
        match="embedded_source_prior is only valid with a two-sided interpolant",
    ):
        Modality(
            shape=SHAPE,
            interpolant=FlowMatchingOneSidedInterpolant(),
            embedder=IdentityEmbedder(dm_shape=SHAPE),
            embedded_source_prior=jr.normal,
        )


def test_registry_set_embedded_source_prior_broadcasts():
    """``registry.set("embedded_source_prior", jr.normal)`` fans out to all leaves."""
    modality_registry = ModalityRegistry(
        {
            "a": Modality(
                shape=SHAPE,
                interpolant=two_sided_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
            ),
            "b": Modality(
                shape=(5,),
                interpolant=two_sided_si(),
                embedder=IdentityEmbedder(dm_shape=(5,)),
            ),
        }
    )
    modality_registry.set("embedded_source_prior", jr.normal)
    assert modality_registry.registry["a"].embedded_source_prior is jr.normal
    assert modality_registry.registry["b"].embedded_source_prior is jr.normal


def test_set_embedded_source_prior_copies_a_stateful_callable():
    """A prior obeys ``use_deepcopy`` exactly as any other field value does.

    A plain function is untouched by the copy — ``copy.deepcopy`` handles functions
    atomically, which is why ``jr.normal`` survives as *the same object* in the test
    above. A **stateful** callable is instead snapshotted per modality, so state no
    longer flows back to the caller's object; ``use_deepcopy=False`` shares one live
    prior across modalities.
    """

    class StatefulPrior:
        """Callable prior holding mutable state, to make the copy observable."""

        def __init__(self):
            self.num_calls = 0

        def __call__(self, key, shape):
            self.num_calls += 1
            return jr.normal(key, shape)

    def two_sided_registry() -> ModalityRegistry:
        return ModalityRegistry(
            {
                "a": Modality(
                    shape=SHAPE,
                    interpolant=two_sided_si(),
                    embedder=IdentityEmbedder(dm_shape=SHAPE),
                ),
                "b": Modality(
                    shape=SHAPE,
                    interpolant=two_sided_si(),
                    embedder=IdentityEmbedder(dm_shape=SHAPE),
                ),
            }
        )

    prior = StatefulPrior()

    modality_registry = two_sided_registry()
    modality_registry.set("embedded_source_prior", prior)
    prior_a = modality_registry.registry["a"].embedded_source_prior
    prior_b = modality_registry.registry["b"].embedded_source_prior
    assert prior_a is not prior
    assert prior_a is not prior_b
    # Each modality owns its state: mutating a copy leaves the caller's prior alone.
    prior_a.num_calls += 1
    assert prior_b.num_calls == 0
    assert prior.num_calls == 0

    modality_registry = two_sided_registry()
    modality_registry.set("embedded_source_prior", prior, use_deepcopy=False)
    assert modality_registry.registry["a"].embedded_source_prior is prior
    assert modality_registry.registry["b"].embedded_source_prior is prior


def test_registry_set_prior_on_one_sided_raises():
    """Setting a prior after a one-sided interpolant raises the same error."""
    from stix.core.interpolant.standard_interpolants.flow_matching import (
        FlowMatchingOneSidedInterpolant,
    )

    modality_registry = ModalityRegistry(
        {
            "a": Modality(
                shape=SHAPE,
                interpolant=FlowMatchingOneSidedInterpolant(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
            )
        }
    )
    with pytest.raises(
        ValueError,
        match="embedded_source_prior is only valid with a two-sided interpolant",
    ):
        modality_registry.set("embedded_source_prior", jr.normal)
