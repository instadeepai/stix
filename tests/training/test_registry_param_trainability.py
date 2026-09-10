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

"""End-to-end trainability guards for a registry-held learnable parameter.

A learnable embedder (or, in future, a learnable interpolant) stored *inside the
``ModalityRegistry`` inside a ``GenerativeModel``* must be reachable by the real
training machinery: ``nnx.split(gen_model, nnx.Param)`` must capture it, it must
receive a gradient through the ``LossPipeline``, and it must move under an
optimizer step. If the registry ever stops being an ``nnx.Module`` (or its
container stops being ``nnx.data``), such params are silently frozen at init — no
error, they simply never train.

These tests mirror the real gradient path
(``nnx.split`` -> ``nnx.merge`` -> ``loss_pipeline`` -> ``optax.apply_updates``;
see ``stix.training.training_step``) and also guard the dual defect: a param-less
object wrongly wrapped in ``nnx.data`` (as ``interpolant`` once was) becomes a
non-array data leaf that makes ``nnx.split(..., nnx.Param)`` raise and breaks
``jax.jit`` tracing of the merged model.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from conftest import BATCH_SIZE, SHAPE, toy_learnable_embedder_gen_model
from flax import nnx
from jaxtyping import PyTree

from stix.training.loss_pipeline import LossPipeline
from stix.typing import RawSourceTargetPair
from stix.typing.data import Batch

# The exact key path at which the registry-held embedder scale must appear in the
# ``nnx.Param`` state of the fully-built model. Asserting on the path (not merely
# "some leaf exists") ties the guard to the registry -> modality -> embedder route.
_SCALE_PATH = "['modality_registry']['registry']['mod_a']['embedder']['scale']"


def _batch() -> Batch:
    """Single all-ones training batch for the one continuous modality."""
    return Batch(
        raw_batch={
            "mod_a": RawSourceTargetPair(
                source=None, target=jnp.ones((BATCH_SIZE, *SHAPE))
            )
        },
        context_data=None,
        context_mask=None,
        attn_mask=None,
        loss_mask=None,
    )


def _find_scale_leaf(state: PyTree) -> jax.Array:
    """Return the leaf whose key path is the registry-held embedder scale."""
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        if _SCALE_PATH in jax.tree_util.keystr(path):
            return leaf
    raise AssertionError(
        f"No leaf at {_SCALE_PATH!r}; registry-held param is invisible to nnx.split."
    )


def test_registry_held_embedder_param_appears_in_param_split():
    """The registry-held embedder scale lands in ``nnx.split(model, nnx.Param)`` state.

    Guards the primary regression: if the registry is not an ``nnx.Module`` (or its
    container not ``nnx.data``), the param is dropped from the split state and thus
    silently frozen — never handed to the optimizer.
    """
    model = toy_learnable_embedder_gen_model()
    _, params = nnx.split(model, nnx.Param)
    paths = {
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_leaves_with_path(params)
    }
    assert any(_SCALE_PATH in path for path in paths), (
        f"Expected a Param at {_SCALE_PATH!r}, got paths: {sorted(paths)}"
    )


def test_registry_held_embedder_param_receives_finite_nonzero_gradient():
    """The registry-held embedder scale gets a finite, nonzero gradient through the loss.

    Differentiates the real ``LossPipeline`` loss w.r.t. the split ``nnx.Param``
    state exactly as ``training_step._loss_fn`` does (merge then call the pipeline,
    ``has_aux=True``). A frozen/invisible param would carry no gradient at all.
    """
    model = toy_learnable_embedder_gen_model()
    graphdef, params = nnx.split(model, nnx.Param)
    pipeline = LossPipeline()
    batch = _batch()

    def loss_fn(params, key):
        gen_model = nnx.merge(graphdef, params)
        return pipeline(gen_model, batch, key)  # (loss, metrics)

    (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, jr.PRNGKey(0))
    scale_grad = _find_scale_leaf(grads)
    assert jnp.all(jnp.isfinite(scale_grad))
    assert jnp.any(scale_grad != 0.0)


def test_registry_held_embedder_param_moves_under_optimizer_step():
    """After one SGD step the registry-held embedder scale actually changes value.

    The full loop: split -> grad through the pipeline -> ``optax.apply_updates``.
    A param that is captured but somehow disconnected from the loss (or frozen)
    would not move; this asserts a genuine value delta.
    """
    model = toy_learnable_embedder_gen_model()
    graphdef, params = nnx.split(model, nnx.Param)
    pipeline = LossPipeline()
    batch = _batch()

    def loss_fn(params, key):
        gen_model = nnx.merge(graphdef, params)
        return pipeline(gen_model, batch, key)

    (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, jr.PRNGKey(0))

    optimizer_tx = optax.sgd(learning_rate=0.1)
    opt_state = optimizer_tx.init(params)
    updates, _ = optimizer_tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    before = _find_scale_leaf(params)
    after = _find_scale_leaf(new_params)
    # ``after`` must be finite: a NaN/inf update would satisfy ``not allclose``
    # (NaN compares unequal to everything) and pass vacuously, so assert it first.
    assert jnp.all(jnp.isfinite(after)), f"param went non-finite after step: {after}"
    assert not jnp.allclose(before, after)


def test_fully_built_gen_model_has_no_non_param_state_leak():
    """A fully-built model has no non-``Param`` state leaf.

    ``training_loop`` trains via ``nnx.split(gen_model, nnx.Param)``; anything that
    is not an ``nnx.Param`` must live in the graphdef, not the state. A param-less
    config object wrongly wrapped in ``nnx.data`` (as ``interpolant`` once was)
    becomes a stray non-``Param`` leaf that makes ``nnx.split(model, nnx.Param)``
    raise and breaks jit tracing. Asserting an empty non-``Param`` remainder catches
    that class generally — whatever object leaked, and independent of whether the
    interpolant is later made an ``nnx.Module``.
    """
    model = toy_learnable_embedder_gen_model()
    _, params, rest = nnx.split(model, nnx.Param, ...)
    assert jax.tree.leaves(params), (
        "registry-held embedder scale missing from Param state"
    )
    assert not jax.tree.leaves(rest), (
        f"non-Param state leaked into the model: {jax.tree.leaves(rest)}"
    )


def test_merged_gen_model_is_jit_traceable():
    """A merged model's state jit-traces without a non-array-leaf error.

    Splits the *full* state (no filter) and threads it through ``jax.jit`` after
    ``nnx.merge``. Guards the sampling-side symptom of the ``nnx.data(interpolant)``
    defect: an interpolant living in the state would be a non-array leaf and abort
    jit tracing. With the interpolant static, the only state leaf is the embedder
    scale (an array), so tracing succeeds.
    """
    model = toy_learnable_embedder_gen_model()
    graphdef, state = nnx.split(model)
    batch = _batch()

    @jax.jit
    def embed(state):
        gen_model = nnx.merge(graphdef, state)
        return gen_model.get_embeddings(batch.raw_batch)

    embedded = embed(state)  # must not raise a non-array-leaf tracing error
    assert embedded["mod_a"].target.shape == (BATCH_SIZE, *SHAPE)
