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

"""Tests for the embedder module."""

import jax
import jax.numpy as jnp
import jax.random as jnr
import pytest
from conftest import BATCH_SIZE, SEQ_LEN
from flax import nnx

from stix.core.embedder import (
    IdentityEmbedder,
    LearnedDiscreteEmbedder,
    OneHotDiscreteEmbedder,
)

# Global Variables
FEATURE_DIM = 8
NUM_CATEGORIES = 5
EMBEDDING_DIM = 16


@pytest.fixture
def identity_embedder():
    return IdentityEmbedder(dm_shape=(SEQ_LEN, FEATURE_DIM))


def make_discrete_embedder(propagate_grads: bool = False) -> LearnedDiscreteEmbedder:
    with pytest.warns(
        UserWarning, match="LearnedDiscreteEmbedder uses learned embeddings"
    ):
        return LearnedDiscreteEmbedder(
            dm_shape=(SEQ_LEN, NUM_CATEGORIES),
            embedding_dim=EMBEDDING_DIM,
            propagate_grads=propagate_grads,
            rngs=nnx.Rngs(0),
        )


@pytest.fixture
def discrete_embedder():
    return make_discrete_embedder()


@pytest.fixture
def one_hot_embedder():
    return OneHotDiscreteEmbedder(dm_shape=(SEQ_LEN, NUM_CATEGORIES))


# ══════════════════════════════════════════════════════════════════════
# IdentityEmbedder
# ══════════════════════════════════════════════════════════════════════


def test_identity_acts_as_identity(identity_embedder, key):
    """Output must be exactly equal to input."""
    x = jnr.normal(key, shape=(BATCH_SIZE, SEQ_LEN, FEATURE_DIM))
    z = identity_embedder.from_raw_to_embeddings(x)
    assert jnp.array_equal(z, x)


def test_identity_roundtrip(identity_embedder, key):
    """Encode then decode must recover the original input exactly."""
    x = jnr.normal(key, shape=(BATCH_SIZE, SEQ_LEN, FEATURE_DIM))
    z = identity_embedder.from_raw_to_embeddings(x)
    x_recovered = identity_embedder.from_embeddings_to_raw(z)
    assert jnp.array_equal(x_recovered, x)


# ══════════════════════════════════════════════════════════════════════
# LearnedDiscreteEmbedder — shape & determinism
# ══════════════════════════════════════════════════════════════════════


def test_discrete_output_shape_from_indices(discrete_embedder, key):
    """Embedding integer indices must produce (batch, seq_len, embedding_dim)."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), minval=0, maxval=NUM_CATEGORIES)
    z = discrete_embedder.from_raw_to_embeddings(indices)
    assert z.shape == (BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)


def test_discrete_output_shape_from_one_hot(discrete_embedder, key):
    """Embedding one-hot vectors must produce (batch, seq_len, embedding_dim)."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), minval=0, maxval=NUM_CATEGORIES)
    one_hot = jax.nn.one_hot(indices, NUM_CATEGORIES)
    z = discrete_embedder.from_raw_to_embeddings(one_hot)
    assert z.shape == (BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)


def test_discrete_is_deterministic(discrete_embedder, key):
    """Embedding the same input twice must give identical results."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), minval=0, maxval=NUM_CATEGORIES)
    z_a = discrete_embedder.from_raw_to_embeddings(indices)
    z_b = discrete_embedder.from_raw_to_embeddings(indices)
    assert jnp.array_equal(z_a, z_b)


# ══════════════════════════════════════════════════════════════════════
# LearnedDiscreteEmbedder — indices vs one-hot agreement
# ══════════════════════════════════════════════════════════════════════


def test_discrete_indices_and_one_hot_agree(discrete_embedder, key):
    """Integer indices and their one-hot encoding must yield the same embeddings."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), 0, NUM_CATEGORIES)
    one_hot = jax.nn.one_hot(indices, NUM_CATEGORIES)

    z_from_indices = discrete_embedder.from_raw_to_embeddings(indices)
    z_from_one_hot = discrete_embedder.from_raw_to_embeddings(one_hot)
    assert jnp.allclose(z_from_indices, z_from_one_hot)


# ══════════════════════════════════════════════════════════════════════
# LearnedDiscreteEmbedder — stop_gradient behaviour
# ══════════════════════════════════════════════════════════════════════


def _embedding_grad_leaves(embedder, indices):
    """Compute gradient leaves of the embedding params w.r.t. a sum loss."""
    graphdef, params, rest = nnx.split(embedder, nnx.Param, ...)

    def loss_fn(params):
        model = nnx.merge(graphdef, params, rest)
        return model.from_raw_to_embeddings(indices).sum()

    grads = jax.grad(loss_fn)(params)
    return jax.tree.leaves(grads)


def test_discrete_stops_gradient(discrete_embedder):
    """With propagate_grads=False, gradients through the embedding must be zero."""
    grad_leaves = _embedding_grad_leaves(discrete_embedder, jnp.array([0, 1, 2]))
    assert all(jnp.allclose(g, 0.0) for g in grad_leaves)


def test_discrete_propagates_gradient():
    """With propagate_grads=True, gradients through the embedding must be non-zero."""
    embedder = make_discrete_embedder(propagate_grads=True)
    grad_leaves = _embedding_grad_leaves(embedder, jnp.array([0, 1, 2]))
    assert any(not jnp.allclose(g, 0.0) for g in grad_leaves)


# ══════════════════════════════════════════════════════════════════════
# LearnedDiscreteEmbedder — roundtrip
# ══════════════════════════════════════════════════════════════════════


def test_discrete_probs_are_valid_distribution(discrete_embedder, key):
    """``from_embeddings_to_probs`` must produce valid distributions over categories.

    The category axis is the table's, so it is ``NUM_CATEGORIES`` wide even though
    the embedding is ``EMBEDDING_DIM`` wide.
    """
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), 0, NUM_CATEGORIES)
    z = discrete_embedder.from_raw_to_embeddings(indices)
    probs = discrete_embedder.from_embeddings_to_probs(z)

    assert probs.shape == (BATCH_SIZE, SEQ_LEN, NUM_CATEGORIES)
    assert jnp.all(probs >= 0.0)
    assert jnp.allclose(probs.sum(axis=-1), 1.0)


def test_discrete_probs_match_softmax_of_logits(discrete_embedder, key):
    """``from_embeddings_to_probs`` is the softmax of ``from_embeddings_to_logits``."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), 0, NUM_CATEGORIES)
    z = discrete_embedder.from_raw_to_embeddings(indices)

    logits = discrete_embedder.from_embeddings_to_logits(z)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, NUM_CATEGORIES)
    assert jnp.allclose(
        discrete_embedder.from_embeddings_to_probs(z), jax.nn.softmax(logits, axis=-1)
    )


def test_discrete_roundtrip_recovers_indices(discrete_embedder, key):
    """Encode then decode must recover the original indices.

    The raw space for a discrete modality is the integer indices themselves, so
    the decode argmaxes internally rather than handing back a distribution.
    """
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), 0, NUM_CATEGORIES)
    z = discrete_embedder.from_raw_to_embeddings(indices)
    recovered = discrete_embedder.from_embeddings_to_raw(z)

    assert recovered.shape == (BATCH_SIZE, SEQ_LEN)
    assert jnp.array_equal(recovered, indices)


# ══════════════════════════════════════════════════════════════════════
# OneHotDiscreteEmbedder
# ══════════════════════════════════════════════════════════════════════


def test_one_hot_output_shape_from_indices(one_hot_embedder, key):
    """Embedding integer indices must produce (batch, seq_len, num_categories)."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), minval=0, maxval=NUM_CATEGORIES)
    z = one_hot_embedder.from_raw_to_embeddings(indices)
    assert z.shape == (BATCH_SIZE, SEQ_LEN, NUM_CATEGORIES)


def test_one_hot_passthrough_existing_one_hot(one_hot_embedder, key):
    """Existing one-hot input must pass through unchanged."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), minval=0, maxval=NUM_CATEGORIES)
    one_hot = jax.nn.one_hot(indices, NUM_CATEGORIES)
    z = one_hot_embedder.from_raw_to_embeddings(one_hot)
    assert jnp.array_equal(z, one_hot)


def test_one_hot_encodes_indices_correctly(one_hot_embedder, key):
    """Encoding indices must match jax.nn.one_hot."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), minval=0, maxval=NUM_CATEGORIES)
    z = one_hot_embedder.from_raw_to_embeddings(indices)
    assert jnp.array_equal(z, jax.nn.one_hot(indices, NUM_CATEGORIES))


def test_one_hot_has_no_learnable_params(one_hot_embedder):
    """OneHotDiscreteEmbedder must expose no nnx.Param leaves."""
    _, params, _ = nnx.split(one_hot_embedder, nnx.Param, ...)
    assert len(jax.tree.leaves(params)) == 0


def test_one_hot_decode_reads_embeddings_as_logits(one_hot_embedder, key):
    """Decoding treats the embedded variable as logits over categories.

    Fed a *soft* embedding (not a one-hot vertex), the decode must return the
    argmax category — pinning the documented "embeddings are logits" contract,
    which the roundtrip below cannot distinguish from any order-preserving map.
    """
    soft = jnr.normal(key, (BATCH_SIZE, SEQ_LEN, NUM_CATEGORIES))
    recovered = one_hot_embedder.from_embeddings_to_raw(soft)

    assert recovered.shape == (BATCH_SIZE, SEQ_LEN)
    assert jnp.array_equal(recovered, jnp.argmax(soft, axis=-1))


def test_one_hot_roundtrip_recovers_indices(one_hot_embedder, key):
    """Encode then decode must recover the original indices."""
    indices = jnr.randint(key, (BATCH_SIZE, SEQ_LEN), 0, NUM_CATEGORIES)
    z = one_hot_embedder.from_raw_to_embeddings(indices)
    recovered = one_hot_embedder.from_embeddings_to_raw(z)

    assert recovered.shape == (BATCH_SIZE, SEQ_LEN)
    assert jnp.array_equal(recovered, indices)


def test_one_hot_embedder_rejects_num_states_on_one_hot_raw_data():
    """Declaring ``num_states`` for already-one-hot data is a silent shape bug.

    ``dm_shape=(L, K)`` with ``num_states=K + 1`` — the natural reach for one-hot
    raw data under masked diffusion — declares ``(L, K, K + 1)`` but emits width
    ``K``, since one-hot input is passed through unchanged. Caught at first use,
    where the raw dtype settles whether the data are indices or one-hot.
    """
    num_tokens, num_categories = 4, 3
    embedder = OneHotDiscreteEmbedder(
        dm_shape=(num_tokens, num_categories), num_states=num_categories + 1
    )
    one_hot_raw = jax.nn.one_hot(jnp.zeros((2, num_tokens), jnp.int32), num_categories)
    with pytest.raises(ValueError, match="already-one-hot"):
        embedder.from_raw_to_embeddings(one_hot_raw)


def test_one_hot_embedder_allows_index_length_equal_to_num_states():
    """``dm_shape=(L,)`` with ``L == num_states`` is a coincidence, not a mistake."""
    embedder = OneHotDiscreteEmbedder(dm_shape=(4,), num_states=4)
    out = embedder.from_raw_to_embeddings(jnp.zeros((2, 4), jnp.int32))
    assert out.shape == (2, 4, 4)
