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

"""Tests for the Orbax checkpointer and training-loop resume.

Covers the checkpointer round-trip, the latest-survives-``BestN``-culling
retention guarantee (the core resume-correctness fix), restore flags,
end-to-end bit-identical resume of the training loop, and ``get_state``-based
dataset-state checkpointing (capture, persistence, and cursor restoration).
"""

from typing import Iterator

import grain
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from conftest import BATCH_SIZE, SHAPE, toy_batch, toy_one_sided_gen_model

from stix.training.checkpointer import (
    Checkpointer,
    CheckpointerConfig,
    read_run_metadata,
    save_run_metadata,
)
from stix.training.loss_pipeline import LossPipeline
from stix.training.training_io_handler import TrainingIOHandler, TrainingIOHandlerConfig
from stix.training.training_loop import (
    TrainingLoop,
    TrainingLoopConfig,
    TrainingState,
    get_debiased_ema,
)
from stix.typing import RawSourceTargetPair
from stix.typing.data import Batch


# ── Helpers ──
def _build_loop(
    num_steps: int,
    eval_every: int | None,
    *,
    train_data: Iterator | None = None,
    io_handler: TrainingIOHandler | None = None,
    seed: int = 42,
) -> TrainingLoop:
    """Deterministic toy loop with a finite constant eval (replayable sampler).

    ``train_data`` defaults to a plain replayable iterator of constant batches;
    pass a grain iterator to exercise dataset-state checkpointing.
    """
    if train_data is None:
        train_data = iter([toy_batch() for _ in range(num_steps + 5)])
    return TrainingLoop(
        train_data=train_data,
        val_data=None,
        loss_pipeline=LossPipeline(),
        gen_model=toy_one_sided_gen_model(),
        optimizer_tx=optax.adam(1e-2),
        config=TrainingLoopConfig(
            num_steps=num_steps,
            eval_every_n_steps=eval_every,
            run_eval_at_start=False,
            random_seed=seed,
        ),
        eval_fn=lambda _model: 0.5,
        io_handler=io_handler,
    )


def _io_handler(
    tmp_path, *, save_interval_steps: int = 2, restore: bool = False
) -> TrainingIOHandler:
    """Saving (and optionally restoring) IO handler rooted at ``tmp_path``."""
    return TrainingIOHandler(
        TrainingIOHandlerConfig(
            checkpoint_dir=str(tmp_path),
            save_interval_steps=save_interval_steps,
            restore_checkpoint_if_exists=restore,
        )
    )


def _random_batch(_index: int, rng: np.random.Generator) -> Batch:
    """A toy batch whose target is drawn from the per-element ``rng``."""
    target = rng.standard_normal((BATCH_SIZE, *SHAPE), dtype=np.float32)
    return Batch(
        raw_batch={
            "mod_a": RawSourceTargetPair(target=jnp.asarray(target), source=None)
        },
        context_data=None,
        context_mask=None,
        attn_mask=None,
        loss_mask=None,
    )


def _grain_batch_iterator(seed: int = 0) -> grain.DatasetIterator:
    """A real grain iterator yielding random toy batches seeded by ``seed``.

    ``random_map`` draws each element from a per-index seeded RNG, so the
    stream is deterministic given ``seed`` yet varies step to step. Grain
    iterators expose ``get_state``/``set_state`` (state is
    ``{"next_index": <int>}``), so the cursor resumes exactly on restore.
    """
    dataset = (
        grain.MapDataset.source([0])
        .repeat()
        .random_map(_random_batch, seed=seed)
        .to_iter_dataset()
    )
    return iter(dataset)


# ══════════════════════════════════════════════════════════════════════
# Round-trip: shapes, dtypes, values
# ══════════════════════════════════════════════════════════════════════


def test_round_trip_preserves_leaves(tmp_path):
    """Save then restore reproduces every leaf, with rng_key uint32[2] and
    num_steps int32 preserved.
    """
    loop = _build_loop(num_steps=2, eval_every=2)
    loop.run()
    state = loop.training_state

    ckpt = Checkpointer(CheckpointerConfig(checkpoint_dir=str(tmp_path)))
    ckpt.save(state, int(state.num_steps), eval_loss=0.5)
    ckpt.wait_until_finished()

    template = _build_loop(num_steps=2, eval_every=2).training_state
    restored_states = ckpt.restore(template)
    assert restored_states is not None
    restored, _ = restored_states

    assert restored.rng_key.shape == (2,)
    assert restored.rng_key.dtype == jnp.uint32
    assert restored.num_steps.dtype == jnp.int32
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
        assert jnp.array_equal(a, b)


# ══════════════════════════════════════════════════════════════════════
# Check the latest checkpoint survives BestN culling
# ══════════════════════════════════════════════════════════════════════


def test_latest_survives_best_culling(tmp_path):
    """A later step with a worse eval_loss than ``max_to_keep`` earlier steps
    must still be retained (LatestN) so ``latest_step()`` is resumable, while
    ``best_step()`` still points at the genuine minimum.
    """
    loop = _build_loop(num_steps=1, eval_every=1)
    loop.run()
    base = loop.training_state

    ckpt = Checkpointer(
        CheckpointerConfig(
            checkpoint_dir=str(tmp_path), max_to_keep=2, save_interval_steps=1
        )
    )
    # Latest step (3) has a WORSE loss than the two best (steps 1, 2).
    losses = {0: 1.0, 1: 0.5, 2: 0.4, 3: 0.9}
    for step, loss in losses.items():
        state = base.replace(num_steps=jnp.array(step, jnp.int32))
        ckpt.save(state, step, eval_loss=loss)
        ckpt.wait_until_finished()

    assert ckpt.latest_step() == 3  # survives despite not being top-2
    assert ckpt.best_step() == 2  # genuine minimum

    template = _build_loop(num_steps=1, eval_every=1).training_state
    restored_states = ckpt.restore(template)  # defaults to latest
    assert restored_states is not None
    restored, _ = restored_states
    assert int(restored.num_steps) == 3


def test_keep_best_false_keeps_latest_n(tmp_path):
    """With keep_best=False, best tracking is disabled (best_step falls back to
    latest) and the latest N are kept by recency.
    """
    loop = _build_loop(num_steps=1, eval_every=1)
    loop.run()
    base = loop.training_state

    ckpt = Checkpointer(
        CheckpointerConfig(
            checkpoint_dir=str(tmp_path),
            max_to_keep=2,
            keep_best=False,
            save_interval_steps=1,
        )
    )
    for step in range(4):
        state = base.replace(num_steps=jnp.array(step, jnp.int32))
        ckpt.save(state, step, eval_loss=1.0)
        ckpt.wait_until_finished()

    assert ckpt.latest_step() == 3
    # Without best tracking, orbax's best_step() falls back to the latest.
    assert ckpt.best_step() == ckpt.latest_step()


# ══════════════════════════════════════════════════════════════════════
# Restore flags
# ══════════════════════════════════════════════════════════════════════


def test_restore_flags_off_reset_from_template(tmp_path):
    """Restoring with a flag off resets that field from the template instead of
    the checkpoint.
    """
    loop = _build_loop(num_steps=2, eval_every=2, seed=100)
    loop.run()
    state = loop.training_state

    ckpt = Checkpointer(CheckpointerConfig(checkpoint_dir=str(tmp_path)))
    ckpt.save(state, int(state.num_steps), eval_loss=0.5)
    ckpt.wait_until_finished()

    # Template with a distinct rng_key so the reset is observable.
    template = _build_loop(num_steps=2, eval_every=2, seed=123).training_state
    template = template.replace(rng_key=jr.PRNGKey(999))

    def restore_state(**flags) -> TrainingState:
        restored = ckpt.restore(template, **flags)
        assert restored is not None
        return restored[0]

    keep = restore_state(restore_rng_key=True)
    reset = restore_state(restore_rng_key=False)

    assert jnp.array_equal(keep.rng_key, state.rng_key)
    assert jnp.array_equal(reset.rng_key, template.rng_key)
    assert not jnp.array_equal(reset.rng_key, state.rng_key)

    # Optimizer-state reset: template's opt_state differs (fresh init) only if
    # values differ; assert the flag routes to the template object's values.
    reset_opt = restore_state(restore_optimizer_state=False)
    for a, b in zip(
        jax.tree.leaves(reset_opt.opt_state), jax.tree.leaves(template.opt_state)
    ):
        assert jnp.array_equal(a, b)


# ══════════════════════════════════════════════════════════════════════
# Debiased EMA restore
# ══════════════════════════════════════════════════════════════════════


def test_restore_ema_matches_loop_debiasing(tmp_path):
    """``restore_ema`` reproduces the debiased EMA the live loop uses for
    evaluation (``get_debiased_ema(ema_params, ema_decay, num_steps)``).
    """
    loop = _build_loop(num_steps=4, eval_every=4)
    loop.run()
    state = loop.training_state
    expected = get_debiased_ema(
        state.ema_params, loop.config.ema_decay, state.num_steps
    )

    ckpt = Checkpointer(CheckpointerConfig(checkpoint_dir=str(tmp_path)))
    ckpt.save(state, int(state.num_steps), eval_loss=0.5)
    ckpt.wait_until_finished()

    # restore_ema needs only a params-shaped template, no optimizer/full state.
    template = _build_loop(num_steps=4, eval_every=4).training_state.ema_params
    ema_params = ckpt.restore_ema(template)

    # allclose, not array_equal: save debiases with the float32-stored ema_decay
    # while the loop uses the float64 config value, so they differ by float ulps.
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(ema_params)):
        assert b.dtype == a.dtype
        assert jnp.allclose(a, b)


# ══════════════════════════════════════════════════════════════════════
# Restore errors
# ══════════════════════════════════════════════════════════════════════


def test_restore_no_checkpoint_returns_none(tmp_path):
    """``restore()`` returns ``None`` on an empty dir (the signal callers use to
    fall back to a fresh start) rather than raising.
    """
    template = _build_loop(
        num_steps=1,
        eval_every=1,
        io_handler=TrainingIOHandler(
            TrainingIOHandlerConfig(restore_checkpoint_if_exists=False)
        ),
    ).training_state
    ckpt = Checkpointer(CheckpointerConfig(checkpoint_dir=str(tmp_path)))
    assert ckpt.restore(template) is None


def test_restore_ema_no_checkpoint_raises(tmp_path):
    ckpt = Checkpointer(CheckpointerConfig(checkpoint_dir=str(tmp_path)))
    template = _build_loop(num_steps=1, eval_every=1).training_state.ema_params
    with pytest.raises(ValueError, match="No checkpoint"):
        ckpt.restore_ema(template)


# ══════════════════════════════════════════════════════════════════════
# End-to-end with resume
# ══════════════════════════════════════════════════════════════════════


def test_resume_is_bit_identical(tmp_path):
    """An interrupted+resumed run reproduces the uninterrupted run's final
    params, opt_state, ema_params, num_steps and rng_key exactly.
    """
    # Uninterrupted reference: 6 steps, eval/checkpoint boundary at 3 and 6.
    ref = _build_loop(num_steps=6, eval_every=3)
    ref.run()
    ref_state = ref.training_state

    # Interrupted: run 3 steps with checkpointing (checkpoint at step 3).
    save_handler = TrainingIOHandler(
        TrainingIOHandlerConfig(
            checkpoint_dir=str(tmp_path),
            save_interval_steps=3,
        )
    )
    first = _build_loop(num_steps=3, eval_every=3, io_handler=save_handler)
    first.run()
    assert (
        Checkpointer(
            CheckpointerConfig(checkpoint_dir=str(tmp_path), max_to_keep=None)
        ).latest_step()
        == 3
    )

    # Resume in a fresh loop, continue to step 6.
    restore_handler = TrainingIOHandler(
        TrainingIOHandlerConfig(
            checkpoint_dir=str(tmp_path),
            save_interval_steps=3,
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=True,
        )
    )
    second = _build_loop(num_steps=6, eval_every=3, io_handler=restore_handler)
    second.run()

    assert int(second.training_state.num_steps) == 6 == int(ref_state.num_steps)
    for a, b in zip(jax.tree.leaves(ref_state), jax.tree.leaves(second.training_state)):
        assert jnp.array_equal(a, b)


def test_no_checkpoint_falls_back_to_fresh_training(tmp_path):
    """``restore_checkpoint_if_exists=True`` against an *empty* checkpoint dir
    must fall back to a fresh start (step 0) and train to completion, rather
    than crashing because there is nothing to restore.
    """
    io_handler = TrainingIOHandler(
        TrainingIOHandlerConfig(
            checkpoint_dir=str(tmp_path),
            save_interval_steps=2,
            restore_checkpoint_if_exists=True,
        )
    )
    loop = _build_loop(num_steps=2, eval_every=2, io_handler=io_handler)

    # No checkpoint existed, so restoration is a no-op: training starts at step 0.
    assert int(loop.training_state.num_steps) == 0

    loop.run()
    assert int(loop.training_state.num_steps) == 2


# ══════════════════════════════════════════════════════════════════════
# Run-level metadata
# ══════════════════════════════════════════════════════════════════════


def test_run_metadata_round_trip(tmp_path):
    """save_run_metadata then read_run_metadata reproduces the dict, and the
    file lands at the run root (sibling of the checkpoints dir).
    """
    checkpoint_dir = tmp_path / "checkpoints"
    run_dir = str(tmp_path)

    metadata = {
        "label": "cifar-baseline",
        "patch_size": 1,
        "embedding_dim": 256,
        "nested": {"num_layers": 12, "heads": 4},
    }
    save_run_metadata(run_dir, metadata)

    assert read_run_metadata(run_dir) == metadata
    # Sibling of checkpoints/, not inside it.
    assert (tmp_path / "run_metadata.json").exists()
    assert not (checkpoint_dir / "run_metadata.json").exists()


def test_read_run_metadata_absent_returns_none(tmp_path):
    """read_run_metadata returns None when no metadata file has been written."""
    assert read_run_metadata(str(tmp_path)) is None


# ══════════════════════════════════════════════════════════════════════
# Dataset-state checkpointing (get_state / set_state)
# ══════════════════════════════════════════════════════════════════════


def test_dataset_without_get_state_is_not_checkpointed(tmp_path):
    """A plain iterator (no ``get_state``) leaves ``dataset_state`` as ``None``,
    the checkpoint omits the ``dataset_state`` item, and training still runs.
    """
    plain_data = iter([toy_batch() for _ in range(5)])
    loop = _build_loop(2, 2, train_data=plain_data, io_handler=_io_handler(tmp_path))

    assert loop.dataset_state is None

    loop.run()  # run() flushes pending checkpoints before returning

    step = int(loop.training_state.num_steps)
    assert step == 2
    # Other items are written, but no dataset_state item exists.
    assert (tmp_path / str(step) / "params").exists()
    assert not (tmp_path / str(step) / "dataset_state").exists()


def test_dataset_state_is_checkpointed_and_restorable(tmp_path):
    """A grain iterator's cursor is checkpointed at the eval boundary and
    round-trips through restore, tracking the batches consumed so far.
    """
    loop = _build_loop(
        2, 2, train_data=_grain_batch_iterator(), io_handler=_io_handler(tmp_path)
    )
    # A fresh grain iterator starts at cursor zero.
    (init_index,) = jax.tree.leaves(loop.dataset_state)
    assert int(init_index) == 0

    loop.run()  # run() flushes pending checkpoints before returning

    step = int(loop.training_state.num_steps)
    assert step == 2
    assert (tmp_path / str(step) / "dataset_state").exists()

    # The persisted cursor reflects the two consumed batches and round-trips.
    ckpt = Checkpointer(
        CheckpointerConfig(checkpoint_dir=str(tmp_path), max_to_keep=None)
    )
    template = {"next_index": np.asarray(0, dtype=np.int64)}
    restored = ckpt.restore(loop.training_state, dataset_state=template)
    assert restored is not None
    _, restored_dataset_state = restored
    # leaves() sidesteps the loosely-typed DatasetState alias while still
    # asserting the persisted cursor round-trips.
    (restored_index,) = jax.tree.leaves(restored_dataset_state)
    assert int(restored_index) == 2


def test_resume_with_dataset_state_is_bit_identical(tmp_path):
    """An interrupted+resumed run reproduces the uninterrupted run exactly.

    Two independent stochastic sources must resume correctly: the rng_key
    stream that drives the train step's noise/time sampling, and the random
    grain data stream. Because the data varies every step (asserted below), a
    bit-identical step-6 state is only possible if the dataset cursor is
    restored on resume — otherwise steps 4-6 would re-see the first three
    batches and diverge.
    """
    # The data really is random/varied (consecutive batches differ).
    probe = _grain_batch_iterator()
    b0, b1 = next(probe), next(probe)
    assert any(
        not jnp.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(b0), jax.tree.leaves(b1))
    )

    # Uninterrupted reference: 6 steps, checkpoint boundaries at 3 and 6.
    ref = _build_loop(6, 3, train_data=_grain_batch_iterator())
    ref.run()
    ref_state = ref.training_state

    # Interrupt: run 3 steps, writing a checkpoint at step 3.
    first = _build_loop(
        3,
        3,
        train_data=_grain_batch_iterator(),
        io_handler=_io_handler(tmp_path, save_interval_steps=3),
    )
    first.run()  # run() flushes the step-3 checkpoint before returning

    # Resume in a fresh loop and continue to step 6.
    second = _build_loop(
        6,
        3,
        train_data=_grain_batch_iterator(),
        io_handler=_io_handler(tmp_path, save_interval_steps=3, restore=True),
    )
    second.run()

    assert int(second.training_state.num_steps) == 6 == int(ref_state.num_steps)
    for a, b in zip(jax.tree.leaves(ref_state), jax.tree.leaves(second.training_state)):
        assert jnp.array_equal(a, b)
