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

"""Tests for the fuse/unfuse round trip used by the fusion backbones."""

from collections import OrderedDict

import jax
import jax.numpy as jnp
import pytest

from stix.nn.fusion import fuse, fuse_mask, unfuse

_TREES = {
    "dict": {
        "mod_a": jnp.arange(2 * 4, dtype=jnp.float32).reshape(2, 4),
        "mod_b": jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4) + 100.0,
    },
    "ordered_dict": OrderedDict(
        {"mod_a": jnp.ones((2, 4), jnp.float32), "mod_b": jnp.full((3, 4), 2.0)}
    ),
    "list": [jnp.ones((2, 4), jnp.float32), jnp.full((3, 4), 2.0)],
    "nested": {
        "group": {"mod_a": jnp.ones((2, 4), jnp.float32)},
        "mod_b": jnp.full((3, 4), 2.0),
    },
}


@pytest.mark.parametrize("name", list(_TREES), ids=list(_TREES))
def test_round_trip_restores_the_structure_and_values(name):
    """Unfusing a fused tree gives back the same structure and the same values."""
    tree = _TREES[name]
    fused, treedef, sizes = fuse(tree)
    assert fused.shape == (5, 4)
    assert sizes == [2, 3]

    restored = unfuse(fused, treedef, sizes)
    assert jax.tree.structure(restored) == jax.tree.structure(tree)
    assert type(restored) is type(tree)
    for original, back in zip(jax.tree.leaves(tree), jax.tree.leaves(restored)):
        assert jnp.array_equal(original, back)


def test_fuse_orders_leaves_independently_of_dict_insertion_order():
    """Two dicts with the same modalities fuse identically however they were built."""
    a = {"mod_a": jnp.zeros((2, 4)), "mod_b": jnp.ones((3, 4))}
    b = {"mod_b": jnp.ones((3, 4)), "mod_a": jnp.zeros((2, 4))}
    assert jnp.array_equal(fuse(a)[0], fuse(b)[0])


def test_fuse_on_the_feature_axis():
    """``axis`` selects which axis is concatenated and split."""
    tree = {"mod_a": jnp.ones((4, 3)), "mod_b": jnp.full((4, 5), 2.0)}
    fused, treedef, sizes = fuse(tree, axis=-1)
    assert fused.shape == (4, 8)
    assert sizes == [3, 5]
    restored = unfuse(fused, treedef, sizes, axis=-1)
    assert jnp.array_equal(restored["mod_a"], tree["mod_a"])
    assert jnp.array_equal(restored["mod_b"], tree["mod_b"])


def test_fuse_mask_whole_none_is_no_mask():
    """A whole ``None`` mask means no masking at all."""
    _, treedef, sizes = fuse(_TREES["dict"])
    assert fuse_mask(None, treedef, sizes) is None


def test_fuse_mask_fills_a_missing_modality_with_ones():
    """A ``None`` in place of one modality's mask leaves it fully visible."""
    _, treedef, sizes = fuse(_TREES["dict"])
    partial = fuse_mask(
        {"mod_a": jnp.array([True, False]), "mod_b": None}, treedef, sizes
    )
    assert partial is not None
    assert partial.shape == (5,)
    assert partial.tolist() == [True, False, True, True, True]


def test_fuse_mask_accepts_a_non_boolean_mask():
    """A mask given as floats gates the same positions as the boolean form."""
    _, treedef, sizes = fuse(_TREES["dict"])
    as_float = {"mod_a": jnp.array([1.0, 0.0]), "mod_b": jnp.ones((3,))}
    as_bool = jax.tree.map(lambda m: m.astype(jnp.bool_), as_float)
    from_float = fuse_mask(as_float, treedef, sizes)
    from_bool = fuse_mask(as_bool, treedef, sizes)
    assert from_float is not None and from_bool is not None
    assert jnp.array_equal(from_float, from_bool)


@pytest.mark.parametrize(
    "bad", [(4,), (), (2, 1)], ids=["wrong_length", "zero_dim", "two_dim"]
)
def test_fuse_mask_rejects_a_mask_that_is_not_one_boolean_per_position(bad):
    """A mask must cover its own modality exactly.

    The wrong-length case still concatenates to the full stream length, so it
    would gate the wrong positions rather than failing on shape.
    """
    _, treedef, sizes = fuse(_TREES["dict"])
    mask = {"mod_a": jnp.ones(bad, dtype=jnp.bool_), "mod_b": jnp.ones((3,), jnp.bool_)}
    with pytest.raises(ValueError, match="expected a mask of shape"):
        fuse_mask(mask, treedef, sizes)


def test_fuse_mask_rejects_a_mask_missing_a_modality():
    """A mask whose structure does not cover the modalities raises."""
    _, treedef, sizes = fuse(_TREES["dict"])
    with pytest.raises(ValueError, match="key mismatch"):
        fuse_mask({"mod_a": jnp.ones((2,), jnp.bool_)}, treedef, sizes)
