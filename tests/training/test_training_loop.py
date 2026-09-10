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

"""Tests for the training loop, EMA helpers, and TrainingLoopConfig."""

from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pydantic
import pytest
from conftest import toy_batch, toy_one_sided_gen_model
from flax import nnx
from jax import Array

from stix.training.ema import _ema_update, get_debiased_ema
from stix.training.loss_pipeline import LossPipeline
from stix.training.training_io_handler import LoggingCategory, TrainingIOHandler
from stix.training.training_loop import TrainingLoop, TrainingLoopConfig


# ── Helpers ──
def toy_training_loop(
    num_steps=1,
    eval_every_n_steps=None,
    run_eval_at_start=False,
    logger_fn=None,
    val_data=None,
    optimizer_tx=None,
) -> TrainingLoop:
    """Build a minimal ``TrainingLoop`` with a one-sided generative model and MSE loss."""
    io_handler = None
    if logger_fn is not None:
        io_handler = TrainingIOHandler()
        io_handler.attach_logger(logger_fn)
    return TrainingLoop(
        train_data=iter([toy_batch() for _ in range(num_steps)]),
        val_data=val_data,
        loss_pipeline=LossPipeline(),
        gen_model=toy_one_sided_gen_model(),
        optimizer_tx=optimizer_tx if optimizer_tx is not None else optax.adam(1e-2),
        config=TrainingLoopConfig(
            num_steps=num_steps,
            eval_every_n_steps=eval_every_n_steps,
            run_eval_at_start=run_eval_at_start,
        ),
        io_handler=io_handler,
    )


# ══════════════════════════════════════════════════════════════════════
# _ema_update
# ══════════════════════════════════════════════════════════════════════


def test_ema_update_formula():
    """``_ema_update`` should implement the standard EMA formula
    ``ema = decay * ema + (1 - decay) * new``.
    """
    ema = nnx.State({"a": jnp.array(10.0)})
    new = nnx.State({"a": jnp.array(0.0)})
    result = _ema_update(ema, new, decay=0.9)
    assert jnp.allclose(cast(Array, result["a"]), 9.0)


def test_ema_update_decay_one_is_identity():
    """With ``decay=1.0``, the EMA update should leave the running average
    unchanged regardless of the new parameters.
    """
    ema = nnx.State({"a": jnp.array(5.0)})
    new = nnx.State({"a": jnp.array(999.0)})
    result = _ema_update(ema, new, decay=1.0)
    assert jnp.allclose(cast(Array, result["a"]), 5.0)


def test_ema_update_decay_zero_replaces():
    """With ``decay=0.0``, the EMA update should replace the running average
    with the new parameters entirely.
    """
    ema = nnx.State({"a": jnp.array(5.0)})
    new = nnx.State({"a": jnp.array(42.0)})
    result = _ema_update(ema, new, decay=0.0)
    assert jnp.allclose(cast(Array, result["a"]), 42.0)


# ══════════════════════════════════════════════════════════════════════
# get_debiased_ema
# ══════════════════════════════════════════════════════════════════════


def test_debiased_ema_formula():
    """``get_debiased_ema`` applies ``ema / (1 - decay^num_updates)``
    (Adam-style bias correction for a zero-initialised running average).
    """
    ema = nnx.State({"a": jnp.array(4.0)})
    decay = 0.5
    num_updates = 3
    result = get_debiased_ema(ema, decay, num_updates)
    correction = 1.0 - 0.5**3  # = 0.875
    assert jnp.allclose(cast(Array, result["a"]), 4.0 / correction)


def test_debiased_ema_after_one_update():
    """After a single update, the correction should be ``1 - decay``."""
    ema = nnx.State({"a": jnp.array(1.0)})
    result = get_debiased_ema(ema, decay=0.9, num_updates=1)
    assert jnp.allclose(cast(Array, result["a"]), 1.0 / 0.1)


# ══════════════════════════════════════════════════════════════════════
# TrainingLoopConfig
# ══════════════════════════════════════════════════════════════════════


def test_config_rejects_non_positive_num_steps():
    """``TrainingLoopConfig`` should reject non-positive ``num_steps`` via a
    pydantic ``ValidationError``.
    """
    with pytest.raises(pydantic.ValidationError):
        TrainingLoopConfig(num_steps=0)


@pytest.mark.parametrize("bad_decay", [0.0, 1.5])
def test_config_rejects_invalid_ema_decay(bad_decay):
    """``TrainingLoopConfig`` should reject ``ema_decay`` values outside
    ``(0, 1]`` via a pydantic ``ValidationError``.
    """
    with pytest.raises(pydantic.ValidationError):
        TrainingLoopConfig(num_steps=1, ema_decay=bad_decay)


# ══════════════════════════════════════════════════════════════════════
# TrainingLoop: construction and accessors
# ══════════════════════════════════════════════════════════════════════


def test_training_loop_init_state():
    """After construction, ``training_state.num_steps`` should be zero."""
    loop = toy_training_loop()
    assert loop.training_state.num_steps == 0


def test_restore_model_without_eval_raises():
    """``restore_model`` should raise a ``ValueError`` when called before any
    evaluation has produced a set of parameters to restore from.
    """
    loop = toy_training_loop(run_eval_at_start=False)
    with pytest.raises(ValueError, match="No parameters to restore"):
        loop.restore_model()


def test_eval_params_uses_raw_before_any_step():
    """Before any training step has run, ``_eval_params`` should return the raw
    model parameters rather than the EMA parameters (which are still in their
    initial uninformative state).
    """
    loop = toy_training_loop()
    eval_p = loop._eval_params()
    raw_p = loop.training_state.params
    for leaf_e, leaf_r in zip(jax.tree.leaves(eval_p), jax.tree.leaves(raw_p)):
        assert jnp.array_equal(leaf_e, leaf_r)


# ══════════════════════════════════════════════════════════════════════
# TrainingLoop: smoke test
# ══════════════════════════════════════════════════════════════════════


def test_training_loop_runs_one_step():
    """Running the training loop for one step should complete without error,
    advance the state to ``num_steps=1``, and produce parameters that differ
    from the initial values after the gradient step.
    """
    loop = toy_training_loop(num_steps=1)
    params_before = jax.tree.map(jnp.copy, loop.training_state.params)

    loop.run()
    state = loop.training_state

    assert state.num_steps == 1
    params_changed = any(
        not jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(params_before), jax.tree.leaves(state.params))
    )
    assert params_changed


def test_training_loop_logger_called():
    """The user-supplied logger attached to the IO handler should be invoked at
    the end of training (after the final step) for each mlip-style category
    (TRAIN/EVAL/SYSTEM/BEST_MODEL), and TRAIN/EVAL payloads should each carry
    a ``loss`` key.
    """
    logged: list[tuple[LoggingCategory, dict, int]] = []

    loop = toy_training_loop(
        logger_fn=lambda category, to_log, step: logged.append((category, to_log, step))
    )
    loop.run()

    by_category = {category: (to_log, step) for category, to_log, step in logged}
    assert LoggingCategory.TRAIN_METRICS in by_category
    assert LoggingCategory.EVAL_METRICS in by_category
    assert LoggingCategory.SYSTEM_METRICS in by_category
    assert LoggingCategory.BEST_MODEL in by_category

    train_payload, train_step = by_category[LoggingCategory.TRAIN_METRICS]
    eval_payload, eval_step = by_category[LoggingCategory.EVAL_METRICS]
    assert train_step == 1
    assert eval_step == 1
    assert "loss" in train_payload
    assert "loss" in eval_payload


# ══════════════════════════════════════════════════════════════════════
# rng_key: seeding, resume guard, JIT inertness
# ══════════════════════════════════════════════════════════════════════


def test_initial_state_has_seeded_rng_key():
    """The initial state carries a uint32[2] PRNG key seeded from random_seed."""
    loop = toy_training_loop()
    rng_key = loop.training_state.rng_key
    assert rng_key.shape == (2,)
    assert rng_key.dtype == jnp.uint32
    assert jnp.array_equal(rng_key, jr.PRNGKey(loop.config.random_seed))


def test_run_eval_at_start_skipped_on_resume():
    """When resuming (num_steps > 0), the run_eval_at_start block is skipped so
    the extra key split does not diverge the resumed stream.
    """
    logged: list[tuple[LoggingCategory, int]] = []
    loop = toy_training_loop(
        num_steps=1,
        run_eval_at_start=True,
        logger_fn=lambda category, to_log, step: logged.append((category, step)),
    )
    # Simulate a resume: advance num_steps past zero before running.
    loop.training_state = loop.training_state.replace(
        num_steps=jnp.array(1, dtype=jnp.int32)
    )
    loop.run()

    # No step-0 initial-eval entry: the start-of-run eval block must be skipped
    # on resume. Any logs emitted should be at the resumed step (>= 1).
    assert all(step >= 1 for _, step in logged)
    assert (LoggingCategory.EVAL_METRICS, 0) not in logged


def test_rng_key_is_inert_in_train_step():
    """The train step must use its explicit key argument, never ``state.rng_key``.

    ``rng_key`` rides on ``TrainingState`` only so it is checkpointed; the JIT'd
    step must ignore it. We check both directions: the step genuinely depends on
    its explicit key (a different key gives a different loss), yet changing only
    ``state.rng_key`` — with the explicit key held fixed — leaves the update
    identical.
    """
    loop = toy_training_loop()
    state, batch = loop.training_state, toy_batch()

    # The step genuinely consumes its explicit key: different key -> different loss.
    _, metrics_a = loop._train_step(state, batch, jr.PRNGKey(1))
    _, metrics_b = loop._train_step(state, batch, jr.PRNGKey(2))
    assert not jnp.array_equal(metrics_a["loss"], metrics_b["loss"])

    # Inertness: same explicit key, different state.rng_key -> identical new params.
    new_default, _ = loop._train_step(state, batch, jr.PRNGKey(1))
    new_other, _ = loop._train_step(
        state.replace(rng_key=jr.PRNGKey(999)),
        batch,
        jr.PRNGKey(1),
    )
    for a, b in zip(
        jax.tree.leaves(new_default.params), jax.tree.leaves(new_other.params)
    ):
        assert jnp.array_equal(a, b)
