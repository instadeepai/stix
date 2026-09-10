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

"""Tests for per-modality criteria."""

import jax
import jax.numpy as jnp
from conftest import DEFAULT_T

from stix.core.loss.criterion import CrossEntropyCriterion, MSECriterion

# ── Helpers ──
t = jnp.array(
    DEFAULT_T
)  # time is unused by both criteria but required by the signature

# ══════════════════════════════════════════════════════════════════════
# MSECriterion
# ══════════════════════════════════════════════════════════════════════


def test_mse_zero_for_identical_inputs():
    """``MSECriterion`` should return exactly zero when the prediction equals the
    ground truth.
    """
    criterion = MSECriterion()
    x = jnp.array([1.0, 2.0, 3.0])
    mask = jnp.ones_like(x)
    assert jnp.allclose(criterion(x, x, t, mask), 0.0)


def test_mse_known_value():
    """``MSECriterion`` should reproduce a hand-computed MSE value for a simple
    prediction/ground-truth pair with a fully-active mask.
    """
    criterion = MSECriterion()
    pred = jnp.array([1.0, 2.0, 3.0])
    gt = jnp.array([2.0, 2.0, 2.0])
    mask = jnp.ones(3)
    # residual^2 = [1, 0, 1], sum = 2, mask.sum = 3 -> 2/3
    expected = 2.0 / 3.0
    assert jnp.allclose(criterion(pred, gt, t, mask), expected)


def test_mse_with_mask():
    """``MSECriterion`` should exclude masked positions from the loss, with the
    denominator normalising by the number of active positions.
    """
    criterion = MSECriterion()
    pred = jnp.array([0.0, 0.0, 0.0])
    gt = jnp.array([2.0, 3.0, 4.0])
    mask = jnp.array([1.0, 1.0, 0.0])  # only the first two positions count
    # residual^2 * mask = [4, 9, 0], sum = 13, mask.sum = 2 -> 13/2 = 6.5
    assert jnp.allclose(criterion(pred, gt, t, mask), 6.5)


def test_mse_fully_masked_returns_zero():
    """``MSECriterion`` should return zero when the mask is fully zero; the
    denominator is clamped to 1 to avoid a division-by-zero.
    """
    criterion = MSECriterion()
    pred = jnp.array([999.0, 999.0])
    gt = jnp.zeros(2)
    mask = jnp.zeros(2)
    assert jnp.allclose(criterion(pred, gt, t, mask), 0.0)


# ══════════════════════════════════════════════════════════════════════
# CrossEntropyCriterion
# ══════════════════════════════════════════════════════════════════════


def test_ce_known_value():
    """``CrossEntropyCriterion`` should reproduce a hand-computed cross-entropy
    value from **logits** and a one-hot target.
    """
    criterion = CrossEntropyCriterion()
    logits = jnp.array([0.7, 0.2, 0.1])
    gt = jnp.array([1.0, 0.0, 0.0])
    mask = jnp.ones(3)
    # CE against the one-hot at index 0 is logsumexp(logits) - logits[0].
    expected = jnp.log(jnp.sum(jnp.exp(logits))) - logits[0]
    assert jnp.allclose(criterion(logits, gt, t, mask), expected)


def test_ce_with_mask():
    """``CrossEntropyCriterion`` should exclude fully-masked examples, with the
    denominator normalising by the number of active examples.

    The last axis is the category axis, so each row is one example; the mask is
    reduced over that axis with ``jnp.max``, so an example counts if *any* of its
    categories is active — a partially-masked row is still a full example.
    """
    criterion = CrossEntropyCriterion()
    logits = jnp.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    gt = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Row 0 fully active, row 1 fully masked, row 2 partially masked. The partial
    # row must count as active (max, not min/mean) — so denominator = 2 examples.
    mask = jnp.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    ce_row0 = jnp.log(jnp.sum(jnp.exp(jnp.array([2.0, 1.0, 0.0])))) - 2.0
    ce_row2 = jnp.log(jnp.sum(jnp.exp(jnp.array([1.0, 2.0, 3.0])))) - 1.0
    expected = (ce_row0 + ce_row2) / 2.0
    assert jnp.allclose(criterion(logits, gt, t, mask), expected)


def test_ce_fully_masked_returns_zero():
    """``CrossEntropyCriterion`` should return zero when the mask is fully zero."""
    criterion = CrossEntropyCriterion()
    logits = jnp.array([0.5, 0.5])
    gt = jnp.array([1.0, 0.0])
    mask = jnp.zeros(2)
    assert jnp.allclose(criterion(logits, gt, t, mask), 0.0)


def test_ce_perfect_prediction_lowerthan_random():
    """``CrossEntropyCriterion`` on logits peaked at the true class should yield a
    lower loss than uniform logits against the same target.
    """
    criterion = CrossEntropyCriterion()
    gt = jnp.array([1.0, 0.0, 0.0])
    mask = jnp.ones(3)
    near_perfect = jnp.array([10.0, 0.0, 0.0])  # softmax mass concentrates on class 0
    uniform = jnp.array([0.0, 0.0, 0.0])  # equal logits -> uniform softmax
    loss_perfect = criterion(near_perfect, gt, t, mask)
    loss_uniform = criterion(uniform, gt, t, mask)
    assert loss_perfect < loss_uniform


def test_ce_gradient_finite_at_saturated_logits():
    """The value *and* gradient must stay finite for extreme logits. This is the
    point of computing the loss in logit space: a naive ``-log(softmax(logits))``
    would take ``log`` of an underflowed ``0.0`` and produce ``NaN`` gradients.
    """
    criterion = CrossEntropyCriterion()
    gt = jnp.array([1.0, 0.0, 0.0])
    mask = jnp.ones(3)
    saturated_logits = jnp.array([1e4, -1e4, -1e4])

    loss, grad = jax.value_and_grad(lambda logits: criterion(logits, gt, t, mask))(
        saturated_logits
    )
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(grad))
